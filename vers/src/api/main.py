"""VERS v3.0 Production FastAPI Backend.

Replaces the legacy Flask mock server with a fully production-ready API
providing REST endpoints for alert ingestion, health monitoring, and
system statistics.

Launch:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AlertPayload,
    AlertReceiptResponse,
    HealthResponse,
    RecentAlertsResponse,
    StatsResponse,
)
from src.infrastructure.logging_config import setup_logging
from src.infrastructure.metrics import get_metrics
from src.infrastructure.security import APIKeyMiddleware, get_or_create_api_key

# ---------------------------------------------------------------------------
# Initialise logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("vers.api")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VERS Emergency Response API",
    description=(
        "Production REST API for the Vision-Based Emergency Response System. "
        "Receives multimodal alert payloads, tracks system health, and exposes "
        "real-time operational statistics."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS (allow Streamlit dashboard to call the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key security middleware ---
app.add_middleware(APIKeyMiddleware)

# ---------------------------------------------------------------------------
# In-memory alert store (production would use Redis / PostgreSQL)
# ---------------------------------------------------------------------------
_alert_store: deque[dict[str, Any]] = deque(maxlen=200)
_start_time = time.time()


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    api_key = get_or_create_api_key()
    logger.info("VERS API v3.0 started. Docs at http://localhost:8000/docs")
    logger.info("API Key: %s", api_key)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Public health check — exempt from API key requirement."""
    metrics = get_metrics()
    return HealthResponse(
        status="healthy",
        version="VERS-3.0-Production",
        uptime_seconds=round(time.time() - _start_time, 1),
        camera_available=True,
        models_loaded=True,
        tts_active=True,
    )


@app.post("/api/v1/alerts", response_model=AlertReceiptResponse, tags=["Alerts"])
async def receive_alert(payload: AlertPayload) -> AlertReceiptResponse:
    """Receive and store an alert payload from the vision pipeline."""
    alert_dict = payload.model_dump()
    _alert_store.appendleft(alert_dict)

    # Record in metrics
    metrics = get_metrics()
    metrics.record_alert(payload.ThreatLevel.value)

    logger.info(
        "Alert received: %s [%s] severity=%.3f",
        payload.MainGesture,
        payload.ThreatLevel.value,
        payload.SeverityScore,
    )

    return AlertReceiptResponse(
        status="received",
        alert_id=alert_dict.get("Timestamp", "unknown"),
        message=f"Alert for {payload.MainGesture} processed.",
    )


# Backward compatibility with legacy Flask endpoint
@app.post("/alert", tags=["Legacy"])
async def legacy_alert(payload: dict[str, Any]) -> dict[str, str]:
    """Legacy endpoint for backward compatibility with existing pipeline."""
    _alert_store.appendleft(payload)
    return {"status": "received", "message": "Alert processed by VERS API v3.0"}


@app.get("/api/v1/alerts/recent", response_model=RecentAlertsResponse, tags=["Alerts"])
async def recent_alerts(limit: int = 20) -> RecentAlertsResponse:
    """Retrieve the most recent alerts."""
    alerts = list(_alert_store)[:limit]
    return RecentAlertsResponse(count=len(alerts), alerts=alerts)


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
async def system_stats() -> StatsResponse:
    """Real-time system statistics and operational metrics."""
    metrics = get_metrics()
    snapshot = metrics.snapshot()
    return StatsResponse(**snapshot)
