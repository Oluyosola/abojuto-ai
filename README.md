# 🛡️ AbojutoAI — AI-Powered Surveillance System

AbojutoAI is a prototype surveillance system combining **YOLOv8 computer vision**, a **FastAPI backend**, and a **real-time web dashboard**. It detects people via webcam, captures snapshots, and fires instant alerts to a web dashboard, email, and Telegram.

---

## Quick Start (< 5 minutes)

### 1. Install dependencies

```bash
cd path/to/Abojuto-ai
pip install -r requirements.txt
```

> YOLOv8-nano model (~6 MB) downloads automatically on first run.

### 2. Configure alerts

```bash
cp .env.example .env
# Open .env and fill in your email + Telegram credentials
```

> The system works without credentials — alerts simply go to the dashboard only.

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Open the dashboard

Go to **http://localhost:8000** in your browser.

---

## Features

| Feature | Details |
|---|---|
| Person detection | YOLOv8-nano (falls back to OpenCV HOG) |
| Live feed | MJPEG stream at ~25 fps |
| Real-time alerts | WebSocket push to dashboard |
| Email alerts | SMTP with snapshot attachment |
| Telegram alerts | Bot message with snapshot photo |
| Event history | SQLite — all events stored locally |
| Snapshot gallery | Clickable thumbnail grid |
| Browser notifications | Optional push notification on detection |

---

## Project Structure

```
Abojuto-ai/
├── main.py          ← FastAPI app, routes, WebSocket
├── detector.py      ← YOLOv8 / HOG detection engine
├── alerts.py        ← Email + Telegram alert senders
├── database.py      ← SQLite event logger
├── requirements.txt
├── .env.example     ← Copy to .env and configure
├── static/
│   ├── index.html   ← Dashboard (single-page)
│   └── snapshots/   ← Auto-saved detection snapshots
└── db/
    └── events.db    ← Auto-created SQLite database
```

---

## Telegram Setup

1. Message **@BotFather** → `/newbot` → copy your token
2. Add the bot to a group or send it a message
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your `chat_id`
4. Paste both into `.env`

## Email Setup (Gmail)

1. Enable 2-Factor Authentication on your Google account
2. Go to **myaccount.google.com/apppasswords**
3. Generate an App Password for "Mail"
4. Use that 16-character password as `SMTP_PASS` in `.env`

---

## Tuning Detection

| Setting | Effect |
|---|---|
| `DETECTION_INTERVAL=1.0` | Lower = more CPU, faster detection |
| `ALERT_COOLDOWN=15.0` | Prevents alert spam; raise for quiet zones |
| `CONFIDENCE_THRESHOLD=0.5` | Lower = more sensitive, more false positives |

---

*AbojutoAI Prototype — Built for rapid deployment in under 24 hours.*
