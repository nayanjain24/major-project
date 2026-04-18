"""Alert Dispatcher service for VERS v2.0.

Responsible for:
  1. Packaging raw fusion results into structured JSON alert payloads
  2. Simulating GPS location data for dashboard map rendering
  3. Managing alert cooldown logic to prevent duplicate spam
  4. Forwarding payloads to the webhook endpoint and TTS engine
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from typing import Any, Optional

from src.services import voice_tts

logger = logging.getLogger("vers.services.alert_dispatcher")

# ---------------------------------------------------------------------------
# Simulated location data — a set of plausible campus/building coordinates
# that rotate on each alert to demonstrate geo-awareness.
# ---------------------------------------------------------------------------
_SIMULATED_LOCATIONS: list[dict[str, float]] = [
    {"lat": 28.6139, "lon": 77.2090, "label": "New Delhi HQ"},
    {"lat": 28.5355, "lon": 77.3910, "label": "Noida Sector 62"},
    {"lat": 28.4595, "lon": 77.0266, "label": "Gurgaon Campus"},
    {"lat": 28.7041, "lon": 77.1025, "label": "Delhi University"},
    {"lat": 28.6304, "lon": 77.2177, "label": "India Gate Area"},
]


def _simulated_location() -> dict[str, Any]:
    """Pick a random simulated GPS coordinate."""
    loc = random.choice(_SIMULATED_LOCATIONS)
    return {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "label": loc["label"],
        "source": "simulated",
    }


# ---------------------------------------------------------------------------
# Cooldown state
# ---------------------------------------------------------------------------
_last_signature: str = ""
_last_alert_time: float = 0.0
ALERT_COOLDOWN_SECONDS: float = 5.0


def build_alert_payload(
    gesture_label: str,
    gesture_confidence: float,
    dominant_emotion: str,
    emotion_distress: float,
    severity_score: float,
    threat_level: str,
    distress_flag: bool,
) -> dict[str, Any]:
    """Construct a fully structured JSON alert payload."""
    location = _simulated_location()

    severity_map = {
        "CRITICAL": "Critical",
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
        "NONE": "None",
    }

    message_map = {
        "SOS": "SOS distress signal detected — immediate assistance required.",
        "EMERGENCY": "Emergency gesture recognised — dispatching alert.",
        "ACCIDENT": "Accident indicator detected — medical team notified.",
        "MEDICAL": "Medical assistance gesture detected — standby.",
        "SAFE": "Safe gesture acknowledged — situation under control.",
    }

    return {
        "Timestamp": datetime.now().isoformat() + "Z",
        "SystemVersion": "VERS-2.0-Multimodal",
        "MainGesture": gesture_label,
        "GestureConfidence": round(gesture_confidence, 4),
        "DominantEmotion": dominant_emotion,
        "EmotionDistress": round(emotion_distress, 4),
        "SeverityScore": round(severity_score, 4),
        "ThreatLevel": threat_level,
        "Severity": severity_map.get(threat_level, "Low"),
        "DistressFlag": distress_flag,
        "Message": message_map.get(gesture_label, "Unknown gesture detected."),
        "Location": location,
    }


def dispatch(
    gesture_label: str,
    gesture_confidence: float,
    dominant_emotion: str,
    emotion_distress: float,
    severity_score: float,
    threat_level: str,
    distress_flag: bool,
    *,
    enable_tts: bool = True,
) -> Optional[dict[str, Any]]:
    """Build and dispatch an alert if cooldown allows; returns payload or None.

    Voice alerts are triggered asynchronously for CRITICAL and HIGH threats.
    """
    global _last_signature, _last_alert_time

    if gesture_label == "NONE":
        return None

    signature = f"{gesture_label}:{threat_level}:{distress_flag}"
    now = time.time()

    if signature == _last_signature and (now - _last_alert_time) < ALERT_COOLDOWN_SECONDS:
        return None  # Suppressed by cooldown

    payload = build_alert_payload(
        gesture_label=gesture_label,
        gesture_confidence=gesture_confidence,
        dominant_emotion=dominant_emotion,
        emotion_distress=emotion_distress,
        severity_score=severity_score,
        threat_level=threat_level,
        distress_flag=distress_flag,
    )

    _last_signature = signature
    _last_alert_time = now

    logger.info("ALERT dispatched: %s [%s]", gesture_label, threat_level)

    # Trigger voice alert for high-severity events
    if enable_tts and threat_level in ("CRITICAL", "HIGH"):
        voice_tts.speak(
            f"Warning. {gesture_label} detected. Threat level {threat_level}. "
            f"Emotion state: {dominant_emotion}."
        )

    return payload
