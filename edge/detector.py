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

try:
    from codecarbon import EmissionsTracker as _ETracker
    _CODECARBON_OK = True
except ImportError:
    _CODECARBON_OK = False

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
    "handgun":    ("CRITICAL", (0, 0, 220)),
    "ammunition": ("CRITICAL", (0, 0, 220)),
    "grenade":    ("CRITICAL", (0, 0, 220)),
    "explosive":  ("CRITICAL", (0, 0, 220)),
    "explosion":  ("CRITICAL", (0, 0, 220)),
    "machete":    ("HIGH",     (0, 60, 255)),
    "blade":      ("HIGH",     (0, 60, 255)),
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

LOW_LIGHT_THRESHOLD  = int(os.getenv("LOW_LIGHT_THRESHOLD", "80"))
WEAPON_HELD_OVERLAP  = 0.30
# Weapons model uses a lower threshold — missing a gun is worse than a false positive
WEAPONS_CONF         = float(os.getenv("WEAPONS_CONF_THRESHOLD", "0.25"))


def _enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE if mean brightness is below LOW_LIGHT_THRESHOLD."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) >= LOW_LIGHT_THRESHOLD:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _weapon_held_by_person(weapon_bbox, person_bboxes) -> bool:
    """True if the weapon bbox overlaps any person bbox by > WEAPON_HELD_OVERLAP."""
    wx1, wy1, wx2, wy2 = weapon_bbox
    w_area = max(1e-6, (wx2 - wx1) * (wy2 - wy1))
    for px1, py1, px2, py2 in person_bboxes:
        ix = max(0.0, min(wx2, px2) - max(wx1, px1))
        iy = max(0.0, min(wy2, py2) - max(wy1, py1))
        if (ix * iy) / w_area > WEAPON_HELD_OVERLAP:
            return True
    return False


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
    frame = _enhance_low_light(frame)

    if _coco_model is None:
        annotated, dets = _hog_detect(frame)
        return annotated, dets, _top_threat(dets)

    annotated    = frame.copy()
    detections   = []
    person_boxes = []
    coco_weapons = []  # non-person COCO detections held until person_boxes is complete

    # ── COCO model — pass 1: collect persons ─────────────────────────────────
    results = _coco_model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in COCO_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        label = COCO_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "person":
            person_boxes.append([x1, y1, x2, y2])
            threat, color = THREAT_MAP.get("person", ("LOW", (0, 200, 80)))
            detections.append({"label": "person", "confidence": round(conf, 3),
                               "bbox": [x1, y1, x2, y2], "threat": threat})
            _draw_box(annotated, x1, y1, x2, y2, "person", conf, color)
        else:
            coco_weapons.append((label, conf, x1, y1, x2, y2))

    # ── COCO model — pass 2: weapons with held-weapon check ──────────────────
    for label, conf, x1, y1, x2, y2 in coco_weapons:
        threat, color = THREAT_MAP.get(label, ("HIGH", (0, 60, 255)))
        if _weapon_held_by_person([x1, y1, x2, y2], person_boxes):
            threat = "CRITICAL"
            color  = (0, 0, 200)
        detections.append({"label": label, "confidence": round(conf, 3),
                           "bbox": [x1, y1, x2, y2], "threat": threat})
        _draw_box(annotated, x1, y1, x2, y2, label, conf, color)

    # ── Weapons model ─────────────────────────────────────────────────────────
    if _weapons_model:
        w_results = _weapons_model(frame, verbose=False)[0]
        for box in w_results.boxes:
            conf = float(box.conf[0])
            if conf < WEAPONS_CONF:
                continue
            cls_name = _weapons_model.names[int(box.cls[0])].lower()
            threat, color = THREAT_MAP.get(cls_name, ("HIGH", (0, 60, 255)))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if _weapon_held_by_person([x1, y1, x2, y2], person_boxes):
                threat = "CRITICAL"
                color  = (0, 0, 200)
            detections.append({"label": cls_name, "confidence": round(conf, 3),
                               "bbox": [x1, y1, x2, y2], "threat": threat})
            _draw_box(annotated, x1, y1, x2, y2, cls_name, conf, color)

    # ── Loitering check ───────────────────────────────────────────────────────
    if loitering_tracker and person_boxes:
        loitering_indices = set(loitering_tracker.update(person_boxes))
        if loitering_indices:
            person_idx = 0
            for det in detections:
                if det["label"] == "person":
                    if person_idx in loitering_indices:
                        det["label"]  = "loitering"
                        det["threat"] = "MEDIUM"
                    person_idx += 1
            n = len(loitering_indices)
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


def _annotate_live(frame: np.ndarray, detections: list) -> None:
    """Draw cached detection boxes on the current live frame (in-place)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label  = det["label"]
        conf   = det["confidence"]
        threat = det["threat"]
        _, color = THREAT_MAP.get(label, ("LOW", (0, 200, 80)))
        if threat == "CRITICAL":
            color = (0, 0, 200)
        _draw_box(frame, x1, y1, x2, y2, label, conf, color)
    top = _top_threat(detections)
    if top not in ("NONE", "LOW"):
        _draw_threat_banner(frame, top)


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
        self._last_threat   = "NONE"   # track severity of last fired alert
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.on_detection: List[Callable] = []

        # ── Energy tracking ───────────────────────────────────────────────────
        self._tracker       = None
        self._session_start = 0.0
        self._frames_read      = 0   # every frame from camera
        self._frames_skipped   = 0   # skipped by motion filter
        self._inferences_run   = 0   # actual AI inference calls
        self._alerts_fired     = 0

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop         = loop
        self._running      = True
        self._session_start = time.time()
        if _CODECARBON_OK:
            self._tracker = _ETracker(
                project_name="vigirix-edge",
                log_level="error",
                save_to_file=False,
                measure_power_secs=2,
                allow_multiple_runs=True,
            )
            self._tracker.start()
            logger.info("CodeCarbon energy tracking active")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._tracker:
            try:
                self._tracker.stop()
            except Exception:
                pass

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

    def get_energy_stats(self) -> dict:
        duration_s  = time.time() - self._session_start if self._session_start else 0
        kwh = co2_g = cpu_kwh = gpu_kwh = ram_kwh = 0.0
        if self._tracker:
            try:
                kwh     = self._tracker._total_energy.kWh
                cpu_kwh = self._tracker._total_cpu_energy.kWh
                gpu_kwh = self._tracker._total_gpu_energy.kWh
                ram_kwh = self._tracker._total_ram_energy.kWh
                # use tracker emissions if available, else estimate at 0.5 kg CO2/kWh
                raw_co2 = float(self._tracker._total_emissions or 0)
                co2_g   = raw_co2 * 1000 if raw_co2 else kwh * 0.5 * 1000
            except Exception:
                pass

        skipped_pct = 0.0
        if self._frames_read > 0:
            skipped_pct = round(self._frames_skipped / self._frames_read * 100, 1)

        return {
            "session_seconds":    round(duration_s, 1),
            "energy_kwh":         round(kwh, 8),
            "energy_wh":          round(kwh * 1000, 5),
            "co2_grams":          round(co2_g, 4),
            "cpu_kwh":            round(cpu_kwh, 8),
            "gpu_kwh":            round(gpu_kwh, 8),
            "ram_kwh":            round(ram_kwh, 8),
            "frames_read":        self._frames_read,
            "inferences_run":     self._inferences_run,
            "frames_skipped_motion": self._frames_skipped,
            "motion_filter_savings_pct": skipped_pct,
            "alerts_fired":       self._alerts_fired,
            "tracker_active":     self._tracker is not None,
        }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_frame  = None
        last_detect = 0.0
        last_dets: list = []   # cached detections drawn on every live frame

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            self._frames_read += 1
            now        = time.time()
            run_detect = (now - last_detect) >= self.detection_interval

            if run_detect:
                last_detect   = now
                should_detect = True
                if self.motion_filter and prev_frame is not None:
                    if not _motion_detected(prev_frame, frame):
                        should_detect = False
                        self._frames_skipped += 1
                prev_frame = frame.copy()

                if should_detect:
                    self._inferences_run += 1
                    annotated, detections, threat = detect_frame(
                        frame,
                        confidence_threshold=self.confidence_threshold,
                        loitering_tracker=self._loitering,
                    )
                    last_dets = detections

                    if detections:
                        top          = _top_threat(detections)
                        cooldown_ok  = (now - self._last_alert) >= self.alert_cooldown
                        # Always fire if new threat is more severe than last alert
                        escalation   = (SEVERITY_RANK.get(top, 0) >
                                        SEVERITY_RANK.get(self._last_threat, 0))
                        if cooldown_ok or escalation:
                            self._last_alert  = now
                            self._last_threat = top
                            self._alerts_fired += 1
                            snapshot = save_snapshot(annotated)
                            self._fire(snapshot, detections, threat)

            # Always render live frame with cached detection overlay
            display = frame.copy()
            _annotate_live(display, last_dets)
            _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
