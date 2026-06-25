"""
AbojutoAI — Alert System
Sends Email (SMTP) and Telegram notifications with snapshot on detection.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("abojuto.alerts")


# ── Configuration (from .env) ────────────────────────────────────────────────

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASS     = os.getenv("SMTP_PASS", "")
ALERT_EMAIL   = os.getenv("ALERT_EMAIL", "")          # recipient

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ── Email ────────────────────────────────────────────────────────────────────

async def send_email_alert(snapshot_path: str, count: int, confidences: list):
    if not all([SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
        logger.warning("Email credentials not configured — skipping email alert")
        return

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    body = (
        f"⚠️ AbojutoAI ALERT\n\n"
        f"Persons detected : {count}\n"
        f"Avg confidence   : {avg_conf:.1%}\n"
        f"Snapshot         : {snapshot_path}\n\n"
        "This is an automated alert from AbojutoAI Surveillance System."
    )

    msg = EmailMessage()
    msg["Subject"] = f"🚨 AbojutoAI: {count} person(s) detected"
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_EMAIL
    msg.set_content(body)

    # Attach snapshot
    snap = Path(snapshot_path)
    if snap.exists():
        with open(snap, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                               filename=snap.name)

    try:
        await aiosmtplib.send(
            msg,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            start_tls=True,
            username=SMTP_USER,
            password=SMTP_PASS,
        )
        logger.info(f"Email alert sent to {ALERT_EMAIL}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")


# ── Telegram ─────────────────────────────────────────────────────────────────

async def send_telegram_alert(snapshot_path: str, count: int, confidences: list):
    if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
        logger.warning("Telegram credentials not configured — skipping Telegram alert")
        return

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    caption = (
        f"🚨 *AbojutoAI Alert*\n"
        f"👤 Persons detected: *{count}*\n"
        f"📊 Avg confidence: *{avg_conf:.1%}*"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    snap = Path(snapshot_path)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if snap.exists():
                with open(snap, "rb") as f:
                    resp = await client.post(url, data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    }, files={"photo": (snap.name, f, "image/jpeg")})
            else:
                resp = await client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": caption,
                    "parse_mode": "Markdown",
                })

        if resp.status_code == 200:
            logger.info("Telegram alert sent ✓")
        else:
            logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# ── Combined dispatcher ──────────────────────────────────────────────────────

async def dispatch_alerts(snapshot_path: str, count: int, confidences: list):
    """Fire all configured alert channels concurrently."""
    await asyncio.gather(
        send_email_alert(snapshot_path, count, confidences),
        send_telegram_alert(snapshot_path, count, confidences),
        return_exceptions=True,
    )
