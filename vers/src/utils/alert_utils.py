"""Shared alert payload construction and logging for the VERS pipeline.

References
----------
- Phase-1 Architecture: Alert Generation -> Communication
- Methodology #6-#7: Message generation, alert transmission
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .data_utils import ALERT_LOG_PATH, ERROR_LOG_PATH

ALERT_MAP: dict[str, dict[str, str]] = {
    "ACCIDENT": {"message": "Accident detected (Fist)", "severity": "High"},
    "EMERGENCY": {"message": "Medical/General emergency (Full Palm)", "severity": "Critical"},
    "SOS": {"message": "SOS request (Peace)", "severity": "High"},
    "SAFETY": {"message": "Safety confirmed (Thumbs up)", "severity": "Low"},
    "NONE": {"message": "No mapped gesture", "severity": "Low"},
}

_alert_logger: logging.Logger | None = None
_error_logger: logging.Logger | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_alert_payload(
    label: str,
    confidence: float,
    distress_score: float,
    distress_flag: bool,
) -> dict[str, Any]:
    """Build the canonical Phase-1 alert payload."""
    info = ALERT_MAP.get(label, ALERT_MAP["NONE"])
    timestamp = _utc_now()
    return {
        "AlertID": timestamp.strftime("GEN_%Y%m%dT%H%M%S%fZ"),
        "Timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "Location": "Simulated Booth A12",
        "Severity": info["severity"],
        "MainGesture": label,
        "GestureConfidence": round(float(confidence), 3),
        "DistressScore": round(float(distress_score), 3),
        "DistressFlag": bool(distress_flag),
        "Message": info["message"],
        "SecondaryEmotion": "High Distress" if distress_flag else "Normal",
    }


def _get_alert_logger() -> logging.Logger:
    global _alert_logger
    if _alert_logger is None:
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _alert_logger = logging.getLogger("vers.alerts")
        _alert_logger.setLevel(logging.INFO)
        _alert_logger.propagate = False
        if not _alert_logger.handlers:
            handler = logging.FileHandler(ALERT_LOG_PATH, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            _alert_logger.addHandler(handler)
    return _alert_logger


def _get_error_logger() -> logging.Logger:
    global _error_logger
    if _error_logger is None:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _error_logger = logging.getLogger("vers.errors")
        _error_logger.setLevel(logging.ERROR)
        _error_logger.propagate = False
        if not _error_logger.handlers:
            handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            _error_logger.addHandler(handler)
    return _error_logger


def log_alert(payload: dict[str, Any]) -> None:
    """Append a JSON serialised alert to ``logs/alerts.log``."""
    _get_alert_logger().info(json.dumps(payload))


def log_error(message: str) -> None:
    """Append a timestamped error to ``logs/errors.log``."""
    _get_error_logger().error(message)
