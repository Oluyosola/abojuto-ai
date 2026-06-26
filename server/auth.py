"""
Vigirix Server — JWT Auth
Single-admin model. Credentials live in .env.
"""
import os
import hmac
import hashlib
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ADMIN_USER   = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS   = os.getenv("ADMIN_PASSWORD", "vigirix2026")
JWT_SECRET   = os.getenv("JWT_SECRET", "change-me-in-production")
TOKEN_TTL    = int(os.getenv("TOKEN_TTL_SECONDS", "28800"))  # 8 hours

import logging as _logging
if ADMIN_PASS == "vigirix2026" or JWT_SECRET == "change-me-in-production":
    _logging.getLogger("vigirix.auth").warning(
        "INSECURE DEFAULTS ACTIVE — set ADMIN_PASSWORD and JWT_SECRET in .env before deploying"
    )


# ── Minimal JWT (no external lib needed) ─────────────────────────────────────
# Uses HMAC-SHA256. Header.Payload.Signature — all base64url encoded.

import base64, json

def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64u_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def create_token(username: str) -> str:
    header  = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64u(json.dumps({
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL,
    }).encode())
    sig = _b64u(hmac.new(
        JWT_SECRET.encode(),
        f"{header}.{payload}".encode(),
        hashlib.sha256,
    ).digest())
    return f"{header}.{payload}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    try:
        header, payload, sig = token.split(".")
        expected = _b64u(hmac.new(
            JWT_SECRET.encode(),
            f"{header}.{payload}".encode(),
            hashlib.sha256,
        ).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_b64u_decode(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def check_credentials(username: str, password: str) -> bool:
    u_ok = hmac.compare_digest(username.encode(), ADMIN_USER.encode())
    p_ok = hmac.compare_digest(password.encode(), ADMIN_PASS.encode())
    return u_ok and p_ok
