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
    def test_centroid_bundle(self, zero_vector):
        """Test prediction with a manually constructed centroid bundle."""
        bundle = {
            "model_type": "centroid",
            "labels": ["SOS", "SAFE"],
            "centroids": {
                "SOS": np.zeros(63).tolist(),
                "SAFE": np.ones(63).tolist(),
            },
            "scales": {
                "SOS": np.ones(63).tolist(),
                "SAFE": np.ones(63).tolist(),
            },
        }
        label, conf = predict_gesture(bundle, zero_vector)
        assert label == "SOS"  # Zero vector closest to SOS centroid (also zeros)
        assert conf > 0.5

    def test_pipeline_bundle(self, model_bundle, random_vector):
        """Test prediction with the real trained model."""
        label, conf = predict_gesture(model_bundle, random_vector)
        assert isinstance(label, str)
        assert len(label) > 0
        assert 0.0 <= conf <= 1.0

    def test_confidence_range(self, model_bundle, zero_vector):
        """Confidence should always be in [0, 1]."""
        _, conf = predict_gesture(model_bundle, zero_vector)
        assert 0.0 <= conf <= 1.0
