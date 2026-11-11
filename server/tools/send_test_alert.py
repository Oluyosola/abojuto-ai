"""
Send a synthetic event through the full alert path (ingest → database →
dashboard → email / Telegram) to check that notification channels work.

Usage (from the server/ folder):
    python tools/send_test_alert.py CRITICAL
    python tools/send_test_alert.py HIGH --url http://127.0.0.1:9010
"""
import argparse
import base64
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_DIR / ".env")

SAMPLE_DETECTIONS = {
    "LOW":      [{"label": "person",  "confidence": 0.91, "bbox": [100, 80, 260, 420], "threat": "LOW"}],
    "MEDIUM":   [{"label": "loitering", "confidence": 0.88, "bbox": [100, 80, 260, 420], "threat": "MEDIUM"}],
    "HIGH":     [{"label": "knife",   "confidence": 0.72, "bbox": [400, 300, 450, 360], "threat": "HIGH"}],
    "CRITICAL": [{"label": "person",  "confidence": 0.90, "bbox": [100, 80, 260, 420], "threat": "LOW"},
                 {"label": "handgun", "confidence": 0.81, "bbox": [210, 220, 260, 260], "threat": "CRITICAL"}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", choices=list(SAMPLE_DETECTIONS))
    ap.add_argument("--url", default=os.getenv("TEST_SERVER_URL", "http://127.0.0.1:9010"))
    args = ap.parse_args()

    snaps = sorted((SERVER_DIR / "static" / "snapshots").glob("*.jpg"))
    snapshot_b64 = base64.b64encode(snaps[-1].read_bytes()).decode() if snaps else None

    key = os.getenv("INGEST_API_KEY", "")
    r = httpx.post(
        f"{args.url}/api/ingest",
        json={
            "device_id":    "test-harness",
            "location":     "Alert channel test",
            "detections":   SAMPLE_DETECTIONS[args.level],
            "threat_level": args.level,
            "snapshot_b64": snapshot_b64,
        },
        headers={"X-API-Key": key} if key else {},
        timeout=10,
    )
    print(r.status_code, r.text)
    sys.exit(0 if r.status_code == 200 else 1)


if __name__ == "__main__":
    main()
