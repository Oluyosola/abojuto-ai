"""
Vigirix Edge Service
Runs on the camera device. Streams MJPEG locally and POSTs detection
events to the Vigirix central server.
"""
import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from detector import CameraDetector

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("vigirix.edge")

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX         = int(os.getenv("CAMERA_INDEX", "0"))
DETECTION_INTERVAL   = float(os.getenv("DETECTION_INTERVAL", "1.0"))
ALERT_COOLDOWN       = float(os.getenv("ALERT_COOLDOWN", "15.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
LOITERING_SECONDS    = float(os.getenv("LOITERING_SECONDS", "30.0"))
MOTION_FILTER        = os.getenv("MOTION_FILTER", "true").lower() == "true"

CENTRAL_URL  = os.getenv("CENTRAL_URL", "http://localhost:9001")
INGEST_KEY   = os.getenv("INGEST_API_KEY", "")
DEVICE_ID    = os.getenv("DEVICE_ID", "edge-01")
DEVICE_LOC   = os.getenv("DEVICE_LOCATION", "Main Entrance")

SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR", "static/snapshots"))
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

camera = CameraDetector(
    camera_index=CAMERA_INDEX,
    detection_interval=DETECTION_INTERVAL,
    alert_cooldown=ALERT_COOLDOWN,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    loitering_seconds=LOITERING_SECONDS,
    motion_filter=MOTION_FILTER,
)


# ── Alert forwarding ─────────────────────────────────────────────────────────

async def forward_to_central(snapshot_path: str, detections: list, threat_level: str):
    """POST the detection event to the central server."""
    snapshot_b64 = None
    p = Path(snapshot_path)
    if p.exists():
        snapshot_b64 = base64.b64encode(p.read_bytes()).decode()

    payload = {
        "device_id":    DEVICE_ID,
        "location":     DEVICE_LOC,
        "detections":   detections,
        "threat_level": threat_level,
        "snapshot_b64": snapshot_b64,
    }
    headers = {"X-API-Key": INGEST_KEY} if INGEST_KEY else {}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(f"{CENTRAL_URL}/api/ingest", json=payload, headers=headers)
        if r.status_code == 200:
            logger.info(f"Alert forwarded → central [{threat_level}]")
        else:
            logger.warning(f"Central rejected alert: {r.status_code}")
    except Exception as e:
        logger.error(f"Could not reach central server: {e}")


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    camera.on_detection.append(forward_to_central)
    camera.start(loop)
    logger.info(f"Vigirix Edge [{DEVICE_ID}] started — streaming to {CENTRAL_URL}")
    yield
    camera.stop()
    logger.info("Edge service stopped")


app = FastAPI(title="Vigirix Edge", lifespan=lifespan)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

def _placeholder_frame() -> bytes:
    """Single JPEG frame shown before the first camera frame arrives."""
    import numpy as np
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Camera starting...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


_PLACEHOLDER = _placeholder_frame()


def _mjpeg_generator():
    import time
    while True:
        frame = camera.get_frame() or _PLACEHOLDER
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)  # ~25 fps cap


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/camera/on")
def camera_on():
    camera.resume()
    return {"paused": False}


@app.post("/api/camera/off")
def camera_off():
    camera.pause()
    return {"paused": True}


@app.get("/api/health")
def health():
    return {
        "service":    "vigirix-edge",
        "device_id":  DEVICE_ID,
        "location":   DEVICE_LOC,
        "running":    camera._running,
        "paused":     camera.paused,
        "central_url": CENTRAL_URL,
    }
