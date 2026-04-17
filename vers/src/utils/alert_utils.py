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
    # Phase-1 demo labels (Custom Mapping)
    "SOS": {"message": "General SOS request (Open Hand)", "severity": "High"},
    "EMERGENCY": {"message": "Urgent Emergency! (2 Fingers)", "severity": "Critical"},
    "ACCIDENT": {"message": "Accident reported (Fist)", "severity": "High"},
    "MEDICAL": {"message": "Medical assistance needed (4 Fingers)", "severity": "High"},
    "SAFE": {"message": "Status: Safe (Thumbs Up)", "severity": "Low"},
    "NONE": {"message": "No mapped gesture", "severity": "Low"},
}
SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
SEVERITY_BY_ORDER = {value: key for key, value in SEVERITY_ORDER.items()}

_alert_logger: logging.Logger | None = None
_error_logger: logging.Logger | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_label(label: str) -> str:
    """Return a canonical uppercase label for alert mapping."""
    normalized = str(label or "").strip().upper()
    return normalized if normalized else "NONE"


def calculate_fused_severity(
    confidence: float,
    distress_score: float,
    base_severity: str,
) -> tuple[str, float]:
    """Fuse gesture confidence + distress score into a final severity class.

    We keep this simple and demo-friendly:
    - 60% weight: gesture confidence
    - 40% weight: normalized distress score
    - final severity never drops below the gesture's base severity
    """
    conf = max(0.0, min(1.0, float(confidence)))
    normalized_distress = max(0.0, min(1.0, float(distress_score) / 0.12))
    fusion_score = 0.6 * conf + 0.4 * normalized_distress

    if fusion_score >= 0.85:
        derived = "Critical"
    elif fusion_score >= 0.65:
        derived = "High"
    elif fusion_score >= 0.45:
        derived = "Medium"
    else:
        derived = "Low"

    base_rank = SEVERITY_ORDER.get(base_severity, SEVERITY_ORDER["Low"])
    derived_rank = SEVERITY_ORDER[derived]
    final_rank = max(base_rank, derived_rank)
    return SEVERITY_BY_ORDER[final_rank], float(fusion_score)


def make_alert_payload(
    label: str,
    confidence: float,
    distress_score: float,
    distress_flag: bool,
) -> dict[str, Any]:
    """Build the canonical Phase-1 alert payload."""
    normalized_label = normalize_label(label)
    info = ALERT_MAP.get(normalized_label, ALERT_MAP["NONE"])
    fused_severity, fusion_score = calculate_fused_severity(
        confidence,
        distress_score,
        info["severity"],
    )
    timestamp = _utc_now()
    return {
        "AlertID": timestamp.strftime("GEN_%Y%m%dT%H%M%S%fZ"),
        "Timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "Location": "Simulated Booth A12",
        "Severity": fused_severity,
        "BaseSeverity": info["severity"],
        "FusionScore": round(float(fusion_score), 3),
        "MainGesture": normalized_label,
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
