"""Security and privacy utilities for VERS v3.0.

Implements privacy-aware design principles:
  - Frame anonymization (face blurring before any persistence)
  - No raw frame storage — frames exist only in RAM
  - API key middleware for FastAPI endpoint protection
  - Privacy audit logging
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from typing import Optional

import cv2
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("vers.security")

# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

# In production, load from a secrets vault. For local dev, auto-generate
# a key and print it to the console on first startup.
_API_KEY: Optional[str] = os.environ.get("VERS_API_KEY")


def get_or_create_api_key() -> str:
    """Return the configured API key, generating one if none is set."""
    global _API_KEY
    if not _API_KEY:
        _API_KEY = secrets.token_urlsafe(32)
        logger.warning(
            "No VERS_API_KEY environment variable set. "
            "Auto-generated key for this session: %s",
            _API_KEY,
        )
    return _API_KEY


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid X-API-Key header.

    Exempt paths: /docs, /openapi.json, /api/v1/health (public health check).
    """

    EXEMPT_PATHS = {"/docs", "/openapi.json", "/redoc", "/api/v1/health"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        api_key = get_or_create_api_key()
        provided = request.headers.get("X-API-Key", "")

        if not secrets.compare_digest(provided, api_key):
            logger.warning("Rejected request to %s — invalid API key.", request.url.path)
            return JSONResponse(
                {"detail": "Invalid or missing API key."},
                status_code=403,
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Frame Anonymization
# ---------------------------------------------------------------------------

def anonymize_frame(frame: np.ndarray, face_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Blur detected face regions in-place for privacy-safe logging.

    Parameters
    ----------
    frame : np.ndarray
        BGR or RGB frame (H×W×3).
    face_regions : list of (x, y, w, h)
        Bounding boxes of detected faces.

    Returns
    -------
    np.ndarray
        Frame with faces blurred.
    """
    anon = frame.copy()
    for (x, y, w, h) in face_regions:
        roi = anon[y:y+h, x:x+w]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (99, 99), 30)
            anon[y:y+h, x:x+w] = blurred
    return anon


# ---------------------------------------------------------------------------
# Privacy Audit
# ---------------------------------------------------------------------------

def log_privacy_event(event_type: str, detail: str = "") -> None:
    """Record a privacy-relevant event for compliance auditing."""
    logger.info("PRIVACY_AUDIT | event=%s | detail=%s", event_type, detail)
