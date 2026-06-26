# Vigirix

AI-powered threat detection and surveillance.
Detects weapons, flags suspicious behaviour, and fires alerts
to a live dashboard, email, and Telegram — all on-device.

---

## What it does

Vigirix watches camera feeds in real time using a fine-tuned
YOLOv8 model. When it spots a threat it saves a snapshot,
logs the event, and pushes an alert instantly.

Two services work together:

| Service | What it runs | Port |
|---|---|---|
| **Edge** | Camera capture + AI inference | 8000 |
| **Server** | Event storage + dashboard + alerts | 8002 |

---

## Threat levels

| Level | Triggers |
|---|---|
| LOW | Person visible |
| MEDIUM | Loitering, erratic movement |
| HIGH | Unattached knife / gun / grenade |
| CRITICAL | Weapon held by a person (30 % bbox overlap) |

---

## Key features

- Fine-tuned weapon model — Gun, knife, grenade, explosion
- COCO fallback — knife and scissors via standard classes
- Held-weapon check — weapon overlapping a person → CRITICAL
- CLAHE low-light boost — better detection in dark frames
- Motion pre-filter — skips inference on static scenes (~70 % less CPU)
- Loitering tracker — centroid-based dwell-time detection
- Real-time WebSocket dashboard — no polling
- Email alerts via Mailtrap (test) or any SMTP (production)
- Telegram alerts with snapshot photo
- Multi-camera support via device IDs

---

## Project layout

```
Abojuto-ai/
├── edge/
│   ├── main.py              FastAPI app, MJPEG stream
│   ├── detector.py          Inference, CLAHE, held-weapon check
│   ├── loitering.py         Dwell-time tracker
│   ├── models/
│   │   └── weapon_model.pt  Fine-tuned weights
│   └── .env
│
├── server/
│   ├── main.py              FastAPI, WebSocket hub, REST API
│   ├── database.py          Async SQLite (aiosqlite)
│   ├── alerts.py            Email + Telegram dispatcher
│   ├── auth.py              JWT (HMAC-SHA256, no external lib)
│   └── .env
│
├── dashboard/
│   └── index.html           Single-file SPA
│
└── yolov8n.pt               COCO weights (auto-downloaded)
```

---

## Quick start

You need two terminal windows.

**Terminal 1 — Server**
```bash
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8002 --reload
```

**Terminal 2 — Edge**
```bash
cd edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

**Dashboard**
```
http://localhost:8002/dashboard/
```

Login: `admin` / `vigirix2026`

---

## Configuration

### edge/.env

```env
CAMERA_INDEX=0
DETECTION_INTERVAL=1.0        # seconds between inference runs
ALERT_COOLDOWN=15.0           # seconds between forwarded alerts
CONFIDENCE_THRESHOLD=0.35     # COCO model threshold
WEAPONS_CONF_THRESHOLD=0.25   # weapons model (lower = catch more)
LOITERING_SECONDS=30.0
MOTION_FILTER=true

YOLO_MODEL=../yolov8n.pt
WEAPONS_MODEL_PATH=models/weapon_model.pt

CENTRAL_URL=http://localhost:8002
INGEST_API_KEY=change-me-secret

DEVICE_ID=edge-01
DEVICE_LOCATION=Main Entrance
```

### server/.env

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=vigirix2026
JWT_SECRET=change-me

INGEST_API_KEY=change-me-secret

DB_PATH=db/vigirix.db
SNAPSHOTS_DIR=static/snapshots
UPLOADS_DIR=static/uploads

SMTP_HOST=smtp.mailtrap.io
SMTP_PORT=2525
SMTP_USER=your-mailtrap-user
SMTP_PASS=your-mailtrap-password
ALERT_EMAIL=recipient@example.com

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## Email alerts (Mailtrap)

Mailtrap catches outgoing email in a safe test inbox
without delivering to real addresses — ideal for demos.

1. Sign up at [mailtrap.io](https://mailtrap.io) (free tier)
2. Go to **Email Testing → Inboxes → SMTP Settings**
3. Copy your username and password
4. Paste into `SMTP_USER` and `SMTP_PASS` in `server/.env`

Emails fire on **HIGH** and **CRITICAL** events only.

---

## Telegram alerts

1. Message **@BotFather** → `/newbot` → copy the token
2. Start a chat with your bot
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
   and copy the `chat_id`
4. Add both to `server/.env`

Telegram fires on **CRITICAL** events with a snapshot photo.

---

## API

All routes except login and `/api/ingest` require
`Authorization: Bearer <token>`.

| Method | Path | Notes |
|---|---|---|
| POST | `/api/auth/login` | Returns JWT |
| GET | `/api/events` | `limit`, `offset`, `threat`, `device` filters |
| GET | `/api/events/{id}` | Single event |
| PATCH | `/api/events/{id}/acknowledge` | Mark as seen |
| DELETE | `/api/events/{id}` | Remove event |
| GET | `/api/stats` | Dashboard counters |
| GET | `/api/devices` | Registered edge devices |
| POST | `/api/ingest` | Edge → server event ingestion |
| GET | `/video_feed` | MJPEG stream *(edge only)* |
| WS | `/ws` | Real-time push *(server only)* |

---

## Tech stack

| | |
|---|---|
| Detection | Ultralytics YOLOv8, OpenCV |
| Backend | Python, FastAPI, aiosqlite |
| Auth | HMAC-SHA256 JWT |
| Real-time | FastAPI WebSocket |
| Alerts | Mailtrap SMTP, Telegram Bot API |
| Frontend | Vanilla HTML / CSS / JS |
| Database | SQLite |

---

*Vigirix — Built for the Sustainable Security Monitoring Hackathon*
