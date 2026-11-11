"""
Vigirix Server — Alert Dispatcher
Fires alerts based on threat level:
  LOW      → dashboard only
  MEDIUM   → dashboard only
  HIGH     → email + dashboard
  CRITICAL → email + Telegram + dashboard
"""
import asyncio
import logging
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import httpx
import aiosmtplib
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("vigirix.alerts")

SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.mailtrap.io")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

TG_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT     = os.getenv("TELEGRAM_CHAT_ID", "")
RETRIES     = 3   # per channel; a brief network drop must not lose a CRITICAL alert

THREAT_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🔴",
    "CRITICAL": "🚨",
}


async def dispatch(
    threat_level: str,
    device_id:    str,
    location:     str,
    detections:   list,
    snapshot_path: Optional[str] = None,
):
    rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    level = rank.get(threat_level, 1)

    tasks = []
    if level >= 3:
        tasks.append(_send_email(threat_level, device_id, location, detections, snapshot_path))
    if level >= 4:
        tasks.append(_send_telegram(threat_level, device_id, location, detections, snapshot_path))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_email(threat_level, device_id, location, detections, snapshot_path):
    if not all([SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
        return
    emoji = THREAT_EMOJI.get(threat_level, "⚠️")
    detected_labels = ", ".join(set(d["label"] for d in detections))
    body = (
        f"{emoji} Vigirix — Threat Level: {threat_level}\n\n"
        f"Device   : {device_id}\n"
        f"Location : {location}\n"
        f"Detected : {detected_labels}\n"
        f"Objects  : {len(detections)}\n\n"
        "This is an automated alert from Vigirix Surveillance."
    )
    msg = EmailMessage()
    msg["Subject"] = f"{emoji} Vigirix [{threat_level}] — {detected_labels} at {location}"
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_EMAIL
    msg.set_content(body)

    snap = Path(snapshot_path) if snapshot_path else None
    if snap and snap.exists():
        with open(snap, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                               filename=snap.name)
    for attempt in range(1, RETRIES + 1):
        try:
            await aiosmtplib.send(
                msg, hostname=SMTP_HOST, port=SMTP_PORT,
                start_tls=True, username=SMTP_USER, password=SMTP_PASS, timeout=20,
            )
            logger.info(f"Email sent [{threat_level}] → {ALERT_EMAIL} (attempt {attempt})")
            return
        except Exception as e:
            logger.warning(f"Email attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e!r}")
            if attempt < RETRIES:
                await asyncio.sleep(2 * attempt)
    logger.error(f"Email failed [{threat_level}] after {RETRIES} attempts")


async def _send_telegram(threat_level, device_id, location, detections, snapshot_path):
    if not all([TG_TOKEN, TG_CHAT]):
        return
    emoji = THREAT_EMOJI.get(threat_level, "⚠️")
    detected_labels = ", ".join(set(d["label"] for d in detections))
    caption = (
        f"{emoji} *Vigirix Alert — {threat_level}*\n"
        f"📍 {location} (`{device_id}`)\n"
        f"🔍 Detected: *{detected_labels}*"
    )
    url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    snap = Path(snapshot_path) if snapshot_path else None
    for attempt in range(1, RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                if snap and snap.exists():
                    r = await client.post(url, data={
                        "chat_id": TG_CHAT, "caption": caption, "parse_mode": "Markdown"
                    }, files={"photo": (snap.name, snap.read_bytes(), "image/jpeg")})
                else:
                    r = await client.post(
                        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                        json={"chat_id": TG_CHAT, "text": caption, "parse_mode": "Markdown"},
                    )
            if r.status_code == 200:
                logger.info(f"Telegram sent [{threat_level}] (attempt {attempt})")
                return
            if r.status_code < 500 and r.status_code != 429:
                # Bad request or bad credentials: retrying will not help
                logger.error(f"Telegram rejected [{threat_level}]: {r.status_code} {r.text[:200]}")
                return
            logger.warning(f"Telegram attempt {attempt}/{RETRIES}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Telegram attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e!r}")
        if attempt < RETRIES:
            await asyncio.sleep(2 * attempt)
    logger.error(f"Telegram failed [{threat_level}] after {RETRIES} attempts")
