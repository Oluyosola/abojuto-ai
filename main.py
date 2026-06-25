"""
AbojutoAI — Main FastAPI Application
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from detector import CameraDetector
from alerts import dispatch_alerts
from database import log_event, get_recent_events, get_stats

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("abojuto.main")

# ── Global state ─────────────────────────────────────────────────────────────

camera = CameraDetector(
    camera_index=int(os.getenv("CAMERA_INDEX", "0")),
    detection_interval=float(os.getenv("DETECTION_INTERVAL", "1.0")),
    alert_cooldown=float(os.getenv("ALERT_COOLDOWN", "15.0")),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
)

ws_clients: list[WebSocket] = []


# ── Detection callback ────────────────────────────────────────────────────────

async def on_person_detected(snapshot_path: str, count: int, confidences: list):
    # 1. Log to DB
    event = log_event(snapshot_path, count, confidences)

    # 2. Push to all connected WebSocket clients
    msg = json.dumps({"type": "detection", "event": event})
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        ws_clients.remove(d)

    # 3. Send external alerts (email + telegram)
    asyncio.create_task(dispatch_alerts(snapshot_path, count, confidences))
    logger.info(f"Detection broadcast: {count} person(s), snapshot={snapshot_path}")


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    camera.on_detection.append(on_person_detected)
    camera.start(loop)
    logger.info("AbojutoAI started 🚀")
    yield
    camera.stop()
    logger.info("AbojutoAI stopped")


app = FastAPI(title="AbojutoAI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve snapshots and static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing():
    with open("static/landing.html", encoding="utf-8") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
async def video_feed():
    """MJPEG live camera stream."""
    async def generate() -> AsyncGenerator[bytes, None]:
        while True:
            frame = camera.get_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.04)   # ~25 fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(ws_clients)} total)")
    try:
        # Send current stats on connect
        await ws.send_text(json.dumps({"type": "stats", "data": get_stats()}))
        while True:
            await ws.receive_text()   # keep alive / ping-pong
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


@app.get("/api/events")
async def api_events(limit: int = 50):
    return get_recent_events(limit)


@app.get("/api/stats")
async def api_stats():
    return get_stats()


@app.get("/api/health")
async def health():
    return {"status": "ok", "camera_running": camera._running, "camera_paused": camera.paused}


@app.post("/api/camera/off")
async def camera_off():
    camera.pause()
    return {"camera_paused": True}


@app.post("/api/camera/on")
async def camera_on():
    camera.resume()
    return {"camera_paused": False}


@app.post("/api/ingest")
async def ingest_alert(request: Request):
    """Receive detections from remote AbojutoAI field devices."""
    api_key = os.getenv("INGEST_API_KEY", "")
    if api_key:
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    event = {
        "type":       "ingest",
        "device_id":  body.get("device_id", "Unknown Device"),
        "location":   body.get("location", ""),
        "count":      int(body.get("count", 1)),
        "confidence": float(body.get("confidence", 0.0)),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    msg = json.dumps({"type": "ingest", "event": event})
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        ws_clients.remove(d)

    logger.info(f"Ingest from {event['device_id']}: {event['count']} person(s)")
    return {"status": "ok", "broadcasted_to": len(ws_clients)}
