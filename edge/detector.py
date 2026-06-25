"""
Vigirix Edge — Detection Engine
Detects: persons, knives (COCO), firearms/weapons (optional weapons model)
Energy-saving motion pre-filter: YOLOv8 only runs when pixels change.
"""
import cv2
import time
import threading
import logging
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Callable

import numpy as np
from dotenv import load_dotenv

from loitering import LoiteringTracker

load_dotenv()
logger = logging.getLogger("vigirix.detector")

# ── Threat config ─────────────────────────────────────────────────────────────
# Each detected class maps to a severity level and BGR draw colour.
THREAT_MAP: Dict[str, tuple] = {
    "person":     ("LOW",      (0, 200, 80)),
    "loitering":  ("MEDIUM",   (0, 165, 255)),
    "knife":      ("HIGH",     (0, 60, 255)),
    "scissors":   ("HIGH",     (0, 60, 255)),
    # Weapons model classes
    "pistol":     ("CRITICAL", (0, 0, 220)),
    "gun":        ("CRITICAL", (0, 0, 220)),
    "rifle":      ("CRITICAL", (0, 0, 220)),
    "firearm":    ("CRITICAL", (0, 0, 220)),
    "weapon":     ("CRITICAL", (0, 0, 220)),
    "ammunition": ("CRITICAL", (0, 0, 220)),
    "grenade":    ("CRITICAL", (0, 0, 220)),
    "explosive":  ("CRITICAL", (0, 0, 220)),
}

SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# COCO class IDs we care about
COCO_CLASSES = {0: "person", 43: "knife", 76: "scissors"}

SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR", "static/snapshots"))
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────
_coco_model    = None
_weapons_model = None

try:
    from ultralytics import YOLO
    _coco_model = YOLO(os.getenv("YOLO_MODEL", "../yolov8n.pt"))
    logger.info("COCO YOLOv8 loaded ✓")
except Exception as e:
    logger.warning(f"YOLOv8 unavailable ({e}) — HOG fallback active")

_weapons_path = os.getenv("WEAPONS_MODEL_PATH", "")
if _weapons_path and Path(_weapons_path).exists():
    try:
        from ultralytics import YOLO as _W
        _weapons_model = _W(_weapons_path)
        logger.info(f"Weapons model loaded: {_weapons_path} ✓")
    except Exception as e:
        logger.warning(f"Weapons model failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_snapshot(frame: np.ndarray) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = SNAPSHOTS_DIR / f"snap_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return str(path)


def _motion_detected(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.015) -> bool:
    """Pixel-diff pre-filter. Saves ~70% AI compute on static scenes."""
    diff  = cv2.absdiff(prev, curr)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    return (np.count_nonzero(th) / th.size) > threshold


def _top_threat(detections: list) -> str:
    if not detections:
        return "NONE"
    return max((d["threat"] for d in detections), key=lambda x: SEVERITY_RANK.get(x, 0))


def _hog_detect(frame: np.ndarray):
    """HOG fallback — only person detection, lower accuracy."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
    dets = []
    for (x, y, w, h) in boxes:
        dets.append({"label": "person", "confidence": 0.6,
                     "bbox": [x, y, x+w, y+h], "threat": "LOW"})
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 80), 2)
    return frame, dets


def detect_frame(
    frame: np.ndarray,
    confidence_threshold: float = 0.45,
    loitering_tracker: Optional[LoiteringTracker] = None,
) -> tuple:
    """
    Returns (annotated_frame, detections, threat_level)
    detections = [{"label", "confidence", "bbox", "threat"}, ...]
    """
    if _coco_model is None:
        annotated, dets = _hog_detect(frame)
        return annotated, dets, _top_threat(dets)

    annotated  = frame.copy()
    detections = []
    person_boxes = []

    # ── COCO model (person + knife) ───────────────────────────────────────────
    results = _coco_model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in COCO_CLASSES:
            continue
        conf  = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        label = COCO_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        threat, color = THREAT_MAP.get(label, ("LOW", (0,200,80)))
        detections.append({"label": label, "confidence": round(conf, 3),
                           "bbox": [x1,y1,x2,y2], "threat": threat})
        if label == "person":
            person_boxes.append([x1, y1, x2, y2])
        _draw_box(annotated, x1, y1, x2, y2, label, conf, color)

    # ── Weapons model ─────────────────────────────────────────────────────────
    if _weapons_model:
        w_results = _weapons_model(frame, verbose=False)[0]
        for box in w_results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            cls_name = _weapons_model.names[int(box.cls[0])].lower()
            threat, color = THREAT_MAP.get(cls_name, ("HIGH", (0, 60, 255)))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({"label": cls_name, "confidence": round(conf, 3),
                               "bbox": [x1,y1,x2,y2], "threat": threat})
            _draw_box(annotated, x1, y1, x2, y2, cls_name, conf, color)

    # ── Loitering check ───────────────────────────────────────────────────────
    if loitering_tracker and person_boxes:
        loitering_ids = loitering_tracker.update(person_boxes)
        if loitering_ids:
            for det in detections:
                if det["label"] == "person":
                    det["label"]  = "loitering"
                    det["threat"] = "MEDIUM"
            n = len(loitering_ids)
            cv2.putText(annotated, f"LOITERING x{n}",
                        (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)

    threat_level = _top_threat(detections)

    # ── Overlay: timestamp + threat ───────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    if threat_level not in ("NONE", "LOW"):
        _draw_threat_banner(annotated, threat_level)

    return annotated, detections, threat_level


def _draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)


def _draw_threat_banner(frame, threat_level: str):
    colours = {"MEDIUM": (0,165,255), "HIGH": (0,60,255), "CRITICAL": (0,0,200)}
    color = colours.get(threat_level, (0,60,255))
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"⚠ THREAT: {threat_level}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)


# ── CameraDetector ────────────────────────────────────────────────────────────

class CameraDetector:
    def __init__(
        self,
        camera_index:         int   = 0,
        detection_interval:   float = 1.0,
        alert_cooldown:       float = 15.0,
        confidence_threshold: float = 0.45,
        loitering_seconds:    float = 30.0,
        motion_filter:        bool  = True,
    ):
        self.camera_index         = camera_index
        self.detection_interval   = detection_interval
        self.alert_cooldown       = alert_cooldown
        self.confidence_threshold = confidence_threshold
        self.motion_filter        = motion_filter

        self._loitering = LoiteringTracker(dwell_seconds=loitering_seconds)
        self._lock          = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._running       = False
        self._paused        = False
        self._thread: Optional[threading.Thread] = None
        self._last_alert    = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.on_detection: List[Callable] = []

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop    = loop
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def pause(self):
        self._paused = True
        with self._lock:
            self._latest_frame = None

    def resume(self):
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_frame   = None
        last_detect  = 0.0

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            now        = time.time()
            run_detect = (now - last_detect) >= self.detection_interval

            if run_detect:
                # Motion pre-filter — save energy on static scenes
                if self.motion_filter and prev_frame is not None:
                    if not _motion_detected(prev_frame, frame):
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                        with self._lock:
                            self._latest_frame = buf.tobytes()
                        prev_frame = frame.copy()
                        continue

                last_detect = now
                annotated, detections, threat = detect_frame(
                    frame,
                    confidence_threshold=self.confidence_threshold,
                    loitering_tracker=self._loitering,
                )
                prev_frame = frame.copy()

                if detections and (now - self._last_alert) >= self.alert_cooldown:
                    self._last_alert = now
                    snapshot = save_snapshot(annotated)
                    self._fire(snapshot, detections, threat)

                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            else:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])

            with self._lock:
                self._latest_frame = buf.tobytes()

        cap.release()

    def _fire(self, snapshot: str, detections: list, threat: str):
        if not self._loop or not self.on_detection:
            return
        for cb in self.on_detection:
            asyncio.run_coroutine_threadsafe(
                cb(snapshot, detections, threat), self._loop
            )
