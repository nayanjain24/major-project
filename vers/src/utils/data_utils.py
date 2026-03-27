"""Shared paths, constants, and data helpers for the VERS pipeline.

References
----------
- Phase-1 System Design: Input -> Preprocessing -> Feature Extraction -> AI Module
- Methodology #1-#3: Camera input, hand detection, landmark extraction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_PATH = DATA_DIR / "landmarks.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_classifier.pkl"

LOG_DIR = PROJECT_ROOT / "logs"
ALERT_LOG_PATH = LOG_DIR / "alerts.log"
ERROR_LOG_PATH = LOG_DIR / "errors.log"
DISTRESS_HISTORY_PATH = LOG_DIR / "distress_history.csv"

DOCS_DIR = PROJECT_ROOT / "docs"
EXPECTED_OUTPUT_DIR = DOCS_DIR / "expected_output"

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63


def ensure_project_dirs() -> None:
    """Create the canonical project directories if they do not already exist."""
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        LOG_DIR,
        DOCS_DIR,
        EXPECTED_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def csv_header() -> list[str]:
    """Return the canonical CSV header for gesture landmark data."""
    return ["label"] + [f"f_{index}" for index in range(FEATURE_DIM)]


def extract_hand_vector(results: object) -> np.ndarray | None:
    """Flatten MediaPipe hand landmarks into a ``(1, 63)`` feature array."""
    if not getattr(results, "multi_hand_landmarks", None):
        return None

    vector: list[float] = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        vector.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
    return np.asarray(vector, dtype=np.float32).reshape(1, -1)
