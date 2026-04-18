"""Production logging configuration for VERS v3.0.

Provides:
  - Rotating file handlers (alerts.log, errors.log, system.log)
  - JSON-structured log format for machine parsing
  - Console + file dual output
  - Log levels configurable via VERS_LOG_LEVEL environment variable
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.environ.get("VERS_LOG_LEVEL", "INFO").upper()


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _rotating_handler(
    filename: str,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    use_json: bool = True,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    return handler


def setup_logging() -> None:
    """Configure the global logging infrastructure for VERS.

    Call once at application startup. Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # --- Console handler (human-readable) ---
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    console.setLevel(logging.INFO)
    root.addHandler(console)

    # --- File handlers (JSON, rotating) ---
    system_handler = _rotating_handler("system.log")
    system_handler.setLevel(logging.DEBUG)
    root.addHandler(system_handler)

    alert_logger = logging.getLogger("vers.alerts")
    alert_logger.addHandler(_rotating_handler("alerts.log"))
    alert_logger.propagate = False

    error_handler = _rotating_handler("errors.log")
    error_handler.setLevel(logging.ERROR)
    root.addHandler(error_handler)

    logging.getLogger("vers").info("VERS logging infrastructure initialised (level=%s)", LOG_LEVEL)
