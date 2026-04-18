"""Pydantic v2 schemas for the VERS production API.

Defines the strict JSON contract for all API request/response payloads,
ensuring type-safe validation and automatic OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ThreatLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class SeverityLevel(str, Enum):
    Critical = "Critical"
    High = "High"
    Medium = "Medium"
    Low = "Low"
    none = "None"


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

class LocationData(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    label: str = "Unknown"
    source: str = "simulated"


# ---------------------------------------------------------------------------
# Alert Payload (inbound from the vision pipeline)
# ---------------------------------------------------------------------------

class AlertPayload(BaseModel):
    """Standardised alert payload emitted by the multimodal fusion engine."""

    Timestamp: str
    SystemVersion: str = "VERS-3.0-Production"
    MainGesture: str
    GestureConfidence: float = Field(..., ge=0.0, le=1.0)
    DominantEmotion: str = "neutral"
    EmotionDistress: float = Field(0.0, ge=0.0, le=1.0)
    SeverityScore: float = Field(..., ge=0.0, le=1.0)
    ThreatLevel: ThreatLevel = ThreatLevel.NONE
    Severity: SeverityLevel = SeverityLevel.Low
    DistressFlag: bool = False
    Message: str = ""
    Location: Optional[LocationData] = None

    class Config:
        json_schema_extra = {
            "example": {
                "Timestamp": "2026-04-18T23:00:00Z",
                "SystemVersion": "VERS-3.0-Production",
                "MainGesture": "SOS",
                "GestureConfidence": 1.0,
                "DominantEmotion": "fear",
                "EmotionDistress": 0.85,
                "SeverityScore": 0.94,
                "ThreatLevel": "CRITICAL",
                "Severity": "Critical",
                "DistressFlag": True,
                "Message": "SOS distress signal detected — immediate assistance required.",
                "Location": {
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                    "label": "New Delhi HQ",
                    "source": "simulated",
                },
            }
        }


# ---------------------------------------------------------------------------
# API Responses
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "VERS-3.0-Production"
    uptime_seconds: float = 0.0
    camera_available: bool = False
    models_loaded: bool = False
    tts_active: bool = False
    deepface_loaded: bool = False


class AlertReceiptResponse(BaseModel):
    status: str = "received"
    alert_id: str = ""
    message: str = "Alert processed successfully."


class StatsResponse(BaseModel):
    total_alerts: int = 0
    alerts_by_threat: dict[str, int] = {}
    average_fps: float = 0.0
    uptime_seconds: float = 0.0
    last_gesture: str = "NONE"
    last_emotion: str = "neutral"


class RecentAlertsResponse(BaseModel):
    count: int = 0
    alerts: list[dict[str, Any]] = []
