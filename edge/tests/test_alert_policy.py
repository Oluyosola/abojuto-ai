"""
Functional test of the alert policy in CameraDetector:
  - a 15 s cooldown suppresses repeated alerts
  - an event more severe than the last alert is sent immediately

Scripted detections are fed through the real capture loop (detect_frame is
replaced), using a short synthetic video played at native speed.

Run from the edge/ folder:  python tests/test_alert_policy.py
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

EDGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE))
import detector as det  # noqa: E402

# second of video → threat returned by the scripted detector
SCRIPT = [(0, "LOW"), (3, "LOW"), (5, "HIGH"), (7, "HIGH"), (9, "CRITICAL"), (11, "LOW"), (13, "CRITICAL")]
EXPECTED = {0: "LOW", 5: "HIGH", 9: "CRITICAL"}   # fired; the rest are inside the cooldown


def scripted(start):
    def fake_detect(frame, **kw):
        t = time.time() - start
        threat = "LOW"
        for s, lvl in SCRIPT:
            if t >= s:
                threat = lvl
        label = {"LOW": "person", "HIGH": "knife", "CRITICAL": "handgun"}[threat]
        return frame, [{"label": label, "confidence": 0.9, "bbox": [0, 0, 10, 10], "threat": threat}], threat
    return fake_detect


def main():
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "blank.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(150):                       # 15 s
        w.write(np.zeros((120, 160, 3), np.uint8))
    w.release()
    det.SNAPSHOTS_DIR = tmp

    cam = det.CameraDetector(camera_index=str(video), detection_interval=1.0,
                             alert_cooldown=15.0, motion_filter=False)
    fired = []
    cam.on_detection.append(lambda *a: None)   # _fire needs a callback to be registered
    start = time.time()
    cam._fire = lambda snap, dets, threat, captured_at: fired.append((round(captured_at - start), threat))
    det.detect_frame = scripted(start)
    cam._tracker = None
    det._CODECARBON_OK = False
    cam.start(asyncio.new_event_loop())
    while not cam.finished:
        time.sleep(0.2)

    got = {t: lvl for t, lvl in fired}
    print("alerts fired (second → level):", got)
    ok = len(fired) == len(EXPECTED) and all(
        any(abs(t - et) <= 1 and lvl == el for t, lvl in fired) for et, el in EXPECTED.items())
    print("PASS" if ok else f"FAIL — expected {EXPECTED}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
