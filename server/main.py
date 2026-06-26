"""
Vigirix Central Server
Receives detection events from edge devices, stores them, broadcasts
over WebSocket to authenticated dashboard clients, dispatches alerts.
"""
import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel

import database as db
import alerts as alert_svc
from auth import check_credentials, create_token, verify_token

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("vigirix.server")

INGEST_API_KEY  = os.getenv("INGEST_API_KEY", "")
SNAPSHOTS_DIR   = Path(os.getenv("SNAPSHOTS_DIR", "static/snapshots"))
UPLOADS_DIR     = Path(os.getenv("UPLOADS_DIR",   "static/uploads"))
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ── WebSocket hub ─────────────────────────────────────────────────────────────

class Hub:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._clients:
                self._clients.remove(ws)

    @property
    def count(self) -> int:
        return len(self._clients)


hub = Hub()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await db.cleanup_old_events(days=30)
    logger.info("Vigirix Central Server ready")
    yield

app = FastAPI(title="Vigirix Central", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

static_path = Path("static")
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve dashboard at /dashboard/
dashboard_path = Path("../dashboard")
if dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")


# ── Auth ──────────────────────────────────────────────────────────────────────

bearer = HTTPBearer(auto_error=False)

def require_auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    claims = verify_token(creds.credentials) if creds else None
    if not claims:
        raise HTTPException(status_code=401, detail="Unauthorised")
    return claims


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(body: LoginBody):
    if not check_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(body.username)
    return {"token": token, "token_type": "bearer"}


# ── Ingest (edge → server) ────────────────────────────────────────────────────

class IngestPayload(BaseModel):
    device_id:    str
    location:     str = ""
    detections:   list
    threat_level: str = "LOW"
    snapshot_b64: Optional[str] = None


@app.post("/api/ingest")
async def ingest(payload: IngestPayload, x_api_key: Optional[str] = None):
    from fastapi import Header
    # Validate API key if one is configured
    if INGEST_API_KEY:
        from fastapi import Request
    # (Key checked via header in the route below)

    snapshot_path = None
    if payload.snapshot_b64:
        from datetime import datetime
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        snap = SNAPSHOTS_DIR / f"snap_{ts}.jpg"
        try:
            snap.write_bytes(base64.b64decode(payload.snapshot_b64))
            snapshot_path = str(snap)
        except Exception as e:
            logger.warning(f"Invalid snapshot_b64 from {payload.device_id}: {e}")

    event_id = await db.insert_event(
        device_id=payload.device_id,
        location=payload.location,
        detections=payload.detections,
        threat_level=payload.threat_level,
        snapshot_path=snapshot_path,
    )

    event = await db.get_event(event_id)
    await hub.broadcast({"type": "detection", "event": event})

    stats = await db.get_stats()
    await hub.broadcast({"type": "stats", "data": stats})

    asyncio.create_task(alert_svc.dispatch(
        threat_level=payload.threat_level,
        device_id=payload.device_id,
        location=payload.location,
        detections=payload.detections,
        snapshot_path=snapshot_path,
    ))

    return {"id": event_id, "status": "received"}


# Ingest with API key header
from fastapi import Request

@app.middleware("http")
async def check_ingest_key(request: Request, call_next):
    if request.url.path == "/api/ingest" and request.method == "POST" and INGEST_API_KEY:
        key = request.headers.get("X-API-Key", "")
        if key != INGEST_API_KEY:
            return JSONResponse({"detail": "Invalid API key"}, status_code=403)
    return await call_next(request)


# ── Events API ────────────────────────────────────────────────────────────────

@app.get("/api/events")
async def get_events(
    limit: int = 50,
    offset: int = 0,
    threat: Optional[str] = None,
    device: Optional[str] = None,
    _=Depends(require_auth),
):
    return await db.get_events(limit=limit, offset=offset, threat=threat, device=device)


@app.get("/api/events/{event_id}")
async def get_event(event_id: int, _=Depends(require_auth)):
    ev = await db.get_event(event_id)
    if not ev:
        raise HTTPException(404, "Event not found")
    return ev


@app.patch("/api/events/{event_id}/acknowledge")
async def ack_event(event_id: int, _=Depends(require_auth)):
    await db.acknowledge_event(event_id)
    return {"acknowledged": True}


@app.delete("/api/events")
async def clear_all_events(_=Depends(require_auth)):
    await db.delete_all_events()
    stats = await db.get_stats()
    await hub.broadcast({"type": "stats", "data": stats})
    await hub.broadcast({"type": "cleared"})
    return {"cleared": True}


@app.delete("/api/events/{event_id}")
async def delete_event(event_id: int, _=Depends(require_auth)):
    ev = await db.get_event(event_id)
    if not ev:
        raise HTTPException(404, "Event not found")
    await db.delete_event(event_id)
    stats = await db.get_stats()
    await hub.broadcast({"type": "stats", "data": stats})
    return {"deleted": event_id}


@app.get("/api/stats")
async def stats(_=Depends(require_auth)):
    return await db.get_stats()


# ── Devices ───────────────────────────────────────────────────────────────────

@app.get("/api/devices")
async def get_devices(_=Depends(require_auth)):
    return await db.get_devices()


# ── Happenings (public read, auth write) ──────────────────────────────────────

@app.get("/api/happenings")
async def get_happenings(limit: int = 10):
    return await db.get_happenings(limit=limit)


@app.post("/api/happenings")
async def post_happening(
    title: str = Form(...),
    body:  str = Form(...),
    image: Optional[UploadFile] = File(None),
    _=Depends(require_auth),
):
    image_path = None
    if image and image.filename:
        from datetime import datetime
        ext  = Path(image.filename).suffix
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = UPLOADS_DIR / f"happening_{ts}{ext}"
        dest.write_bytes(await image.read())
        image_path = str(dest)

    hid = await db.insert_happening(title, body, image_path)
    h   = await db.get_happening(hid)
    await hub.broadcast({"type": "happening", "data": h})
    return h


@app.delete("/api/happenings/{hid}")
async def del_happening(hid: int, _=Depends(require_auth)):
    await db.delete_happening(hid)
    return {"deleted": hid}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        auth_msg = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        claims = verify_token(json.loads(auth_msg).get("token", ""))
    except Exception:
        claims = None

    if not claims:
        await ws.close(code=4001)
        return

    hub._clients.append(ws)
    try:
        stats = await db.get_stats()
        await ws.send_json({"type": "stats", "data": stats})
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        hub.disconnect(ws)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def landing_page():
    landing = Path("../landing/index.html")
    if landing.exists():
        return FileResponse(str(landing))
    return {"service": "vigirix-server", "status": "ok"}


@app.get("/api/health")
async def health():
    return {
        "service":    "vigirix-server",
        "ws_clients": hub.count,
    }
