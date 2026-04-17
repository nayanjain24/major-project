"""Shared paths, constants, and data helpers for the VERS pipeline.

References
----------
- Phase-1 System Design: Input -> Preprocessing -> Feature Extraction -> AI Module
- Methodology #1-#3: Camera input, hand detection, landmark extraction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def extract_hand_vector(results: object) -> Any:
    """Flatten MediaPipe hand landmarks into a scale-invariant normalized (1, 63) DataFrame.

    Returns a pandas DataFrame with column names ``f_0`` through ``f_62``,
    or ``None`` if no hand was detected.

    The lazy import of pandas ensures scripts that don't call this function
    (e.g. the alert server) don't pay the import cost.
    """
    if not getattr(results, "multi_hand_landmarks", None):
        return None

    import pandas as pd

    hand_lms = results.multi_hand_landmarks[0].landmark
    aspect_ratio = 1280.0 / 720.0
    wrist = np.array([hand_lms[0].x * aspect_ratio, hand_lms[0].y, hand_lms[0].z])
    mid_base = np.array([hand_lms[9].x * aspect_ratio, hand_lms[9].y, hand_lms[9].z])

    # Calculate scale factor (distance from wrist to middle finger base)
    # We use a small epsilon to avoid division by zero
    scale = np.linalg.norm(mid_base - wrist) + 1e-6

    vector: list[float] = []
    for lm in hand_lms:
        # Subtract wrist coordinates and normalize by scale
        dx = (lm.x * aspect_ratio - wrist[0]) / scale
        dy = (lm.y - wrist[1]) / scale
        dz = (lm.z - wrist[2]) / scale
        vector.extend([dx, dy, dz])

    # Return as a DataFrame with feature names to match training data
    cols = [f"f_{i}" for i in range(FEATURE_DIM)]
    return pd.DataFrame([vector], columns=cols)


def open_camera_capture(
    *,
    max_index: int = 3,
    warmup_reads: int = 20,
    warmup_sleep_sec: float = 0.03,
):
    """Open the first responsive camera capture with macOS-friendly fallbacks.

    Returns
    -------
    tuple[cv2.VideoCapture | None, str]
        ``(capture, backend_info)`` where capture is ``None`` if no usable camera
        could be opened.
    """
    import time
    import cv2

    attempts: list[tuple[str, Any]] = [("DEFAULT", cv2.CAP_ANY)]
    cap_avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
    if cap_avfoundation is not None:
        attempts.insert(0, ("AVFOUNDATION", cap_avfoundation))

    for backend_name, backend_flag in attempts:
        for cam_idx in range(max_index):
            try:
                cap = cv2.VideoCapture(cam_idx, backend_flag)
            except TypeError:
                cap = cv2.VideoCapture(cam_idx)
            if not cap or not cap.isOpened():
                continue

            got_frame = False
            for _ in range(max(1, warmup_reads)):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    got_frame = True
                    break
                time.sleep(max(0.0, warmup_sleep_sec))

            if got_frame:
                return cap, f"{backend_name}:{cam_idx}"

            cap.release()

    return None, "none"
