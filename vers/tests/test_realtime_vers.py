"""Tests for core functions in src/realtime_vers.py."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime_vers import (
    calc_distress,
    predict_gesture,
    smooth_prediction,
)


class TestCalcDistress:
    def test_returns_zero_for_none(self):
        assert calc_distress(None, 640, 480) == 0.0

    def test_returns_float(self):
        result = calc_distress(None, 1280, 720)
        assert isinstance(result, float)


class TestSmoothPrediction:
    def test_empty_history_returns_none(self):
        label, conf = smooth_prediction(deque())
        assert label == "NONE"
        assert conf == 0.0

    def test_single_prediction(self):
        history: deque[tuple[str, float]] = deque([("SOS", 0.9)])
        label, conf = smooth_prediction(history)
        assert label == "SOS"
        assert abs(conf - 0.9) < 1e-6

    def test_majority_wins(self):
        history: deque[tuple[str, float]] = deque([
            ("SOS", 0.8),
            ("SOS", 0.85),
            ("EMERGENCY", 0.6),
        ])
        label, _ = smooth_prediction(history)
        assert label == "SOS"

    def test_all_none_returns_none(self):
        history: deque[tuple[str, float]] = deque([
            ("NONE", 0.0),
            ("NONE", 0.0),
        ])
        label, conf = smooth_prediction(history)
        assert label == "NONE"
        assert conf == 0.0


class TestPredictGesture:
    def test_heuristic_sos_all_extended(self):
        """Test SOS gesture (all fingers extended according to heuristic distances)."""
        v = np.zeros((21, 3))
        # Extend fingers by making tips further than PIPS
        for tip in [8, 12, 16, 20]:
            v[tip] = [0, 10, 0]
        # Extend thumb (dist from thumb tip 4 to pinky base 17 must be > thumb ip 3 to pinky base 17)
        v[4] = [10, 0, 0]
        v[17] = [0, 10, 0]
        
        label, conf = predict_gesture({}, v.flatten())
        assert label == "SOS"
        assert conf == 1.0

    def test_heuristic_accident_all_folded(self, zero_vector):
        """Test ACCIDENT gesture (zero vector means all dists are 0 -> all fingers folded)."""
        label, conf = predict_gesture({}, zero_vector)
        assert label == "ACCIDENT"
        assert conf == 1.0

    def test_confidence_range(self, zero_vector):
        """Confidence should always be in [0, 1]."""
        _, conf = predict_gesture({}, zero_vector)
        assert 0.0 <= conf <= 1.0
