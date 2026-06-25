"""
AbojutoAI — Detection Engine
Runs YOLOv8-nano person detection on webcam frames.
Falls back to OpenCV HOG if ultralytics is unavailable.
"""

import cv2
import time
import threading
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("abojuto.detector")

SNAPSHOT_DIR = Path("static/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Try to load YOLOv8 ──────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")          # downloads ~6 MB on first run
    _use_yolo = True
    logger.info("YOLOv8-nano loaded ✓")
except Exception as e:
    logger.warning(f"YOLOv8 unavailable ({e}), falling back to HOG detector")
    _use_yolo = False
    _hog = cv2.HOGDescriptor()
    _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people(frame):
    """
    Returns (annotated_frame, person_count, confidence_list).
    """
    if _use_yolo:
        results = _model(frame, classes=[0], verbose=False)[0]  # class 0 = person
        boxes = results.boxes
        count = len(boxes)
        confidences = [float(b.conf) for b in boxes]
        annotated = results.plot()
        return annotated, count, confidences
    else:
        # HOG fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = _hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        annotated = frame.copy()
        confidences = []
        for (x, y, w, h), weight in zip(rects, weights):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            conf = float(weight[0]) if hasattr(weight, '__len__') else float(weight)
            confidences.append(conf)
            cv2.putText(annotated, f"Person {conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated, len(rects), confidences


def save_snapshot(frame) -> str:
    """Save a JPEG snapshot and return the relative URL path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"snapshot_{ts}.jpg"
    path = SNAPSHOT_DIR / filename
    cv2.imwrite(str(path), frame)
    return f"static/snapshots/{filename}"


# ── Camera streamer ──────────────────────────────────────────────────────────

class CameraDetector:
    """
    Runs detection in a background thread.
    Provides:
      - `get_frame()` → latest annotated JPEG bytes for MJPEG stream
      - `on_detection` callback list for alert subscribers
    """

    def __init__(
        self,
        camera_index: int = 0,
        detection_interval: float = 1.0,    # seconds between detection runs
        alert_cooldown: float = 15.0,       # min seconds between alerts
        confidence_threshold: float = 0.5,
    ):
        self.camera_index = camera_index
        self.detection_interval = detection_interval
        self.alert_cooldown = alert_cooldown
        self.confidence_threshold = confidence_threshold

        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._last_alert_time: float = 0.0

        # Subscribers: async callables (snapshot_path, count, confidences) → None
        self.on_detection: list[Callable] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Camera detector stopped")

    def pause(self):
        self._paused = True
        with self._lock:
            self._latest_frame = None
        logger.info("Camera paused")

    def resume(self):
        self._paused = False
        logger.info("Camera resumed")

    @property
    def paused(self) -> bool:
        return self._paused

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    # ── Internal loop ────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        last_detect = 0.0

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame read failed, retrying…")
                time.sleep(0.1)
                continue

            now = time.time()
            run_detect = (now - last_detect) >= self.detection_interval

            if run_detect:
                last_detect = now
                annotated, count, confs = detect_people(frame)

                # Filter by confidence (HOG weights are >1, so skip threshold for HOG)
                qualified = [c for c in confs if not _use_yolo or c >= self.confidence_threshold]

                if qualified and (now - self._last_alert_time) >= self.alert_cooldown:
                    self._last_alert_time = now
                    snapshot_path = save_snapshot(annotated)
                    self._fire_detection(snapshot_path, len(qualified), qualified)

                # Encode annotated frame for MJPEG
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                jpeg = buf.tobytes()
            else:
                # Just encode raw frame (no detection overlay)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                jpeg = buf.tobytes()

            with self._lock:
                self._latest_frame = jpeg

        cap.release()

    def _fire_detection(self, snapshot_path: str, count: int, confidences: list):
        if not self._loop or not self.on_detection:
            return
        for cb in self.on_detection:
            asyncio.run_coroutine_threadsafe(
                cb(snapshot_path, count, confidences), self._loop
            )
