"""
Vigirix Edge Service
Runs on the camera device. Streams MJPEG locally and POSTs detection
events to the Vigirix central server.
"""
import asyncio
import base64
import csv
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from detector import CameraDetector

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("vigirix.edge")

# ── Config ────────────────────────────────────────────────────────────────────
# A number selects a webcam; anything else is treated as a video file path
_camera_src          = os.getenv("CAMERA_INDEX", "0").strip()
CAMERA_INDEX         = int(_camera_src) if _camera_src.isdigit() else _camera_src
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

# One CSV row per alert, for offline analysis of latency per configuration
ALERT_LOG = Path(os.getenv("ALERT_LOG", "metrics/alerts.csv"))
ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
RUN_LABEL = os.getenv("RUN_LABEL", "live")   # e.g. "baseline" / "proposed"

camera = CameraDetector(
    camera_index=CAMERA_INDEX,
    detection_interval=DETECTION_INTERVAL,
    alert_cooldown=ALERT_COOLDOWN,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    loitering_seconds=LOITERING_SECONDS,
    motion_filter=MOTION_FILTER,
)


# ── Alert forwarding ─────────────────────────────────────────────────────────

def _log_alert(captured_at: float, threat_level: str, detections: list,
               latency_ms: Optional[float], status: str):
    new_file = not ALERT_LOG.exists()
    with ALERT_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run_label", "source", "captured_at", "threat_level",
                        "labels", "latency_ms", "status"])
        w.writerow([RUN_LABEL, CAMERA_INDEX,
                    datetime.fromtimestamp(captured_at).isoformat(timespec="milliseconds"),
                    threat_level, "|".join(d["label"] for d in detections),
                    "" if latency_ms is None else round(latency_ms, 1), status])


async def forward_to_central(snapshot_path: str, detections: list, threat_level: str,
                             captured_at: float):
    """POST the detection event to the central server.

    Alert latency = frame capture → central server acknowledges the event.
    """
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
            latency_ms = (time.time() - captured_at) * 1000
            camera.record_alert_latency(latency_ms)
            _log_alert(captured_at, threat_level, detections, latency_ms, "ok")
            logger.info(f"Alert forwarded → central [{threat_level}] in {latency_ms:.0f} ms")
        else:
            _log_alert(captured_at, threat_level, detections, None, f"http_{r.status_code}")
            logger.warning(f"Central rejected alert: {r.status_code}")
    except Exception as e:
        _log_alert(captured_at, threat_level, detections, None, "unreachable")
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
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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


@app.get("/api/energy")
def energy():
    return camera.get_energy_stats()


@app.get("/api/latency")
def latency():
    return camera.get_latency_stats()
