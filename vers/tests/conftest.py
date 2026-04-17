"""Shared test fixtures for the VERS test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "gesture_classifier.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "landmarks.csv"


@pytest.fixture
def model_bundle():
    """Load the trained model bundle, skip if not present."""
    if not MODEL_PATH.exists():
        pytest.skip("Model file not present; train the classifier first.")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture
def zero_vector():
    """A (1, 63) zero vector for smoke testing predictions."""
    return np.zeros((1, 63), dtype=np.float32)


@pytest.fixture
def random_vector():
    """A (1, 63) random vector for testing predictions don't crash."""
    rng = np.random.default_rng(42)
    return rng.random((1, 63)).astype(np.float32)


EXPECTED_LABELS = {"SOS", "EMERGENCY", "ACCIDENT", "MEDICAL", "SAFE"}
