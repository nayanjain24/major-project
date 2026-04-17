"""Smoke test for the trained gesture classifier.

Loads models/gesture_classifier.pkl and verifies the model can produce
predictions on a zero vector without exceptions. Handles both sklearn
pipeline bundles and centroid-based bundles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import EXPECTED_LABELS


def test_prediction_zero_vector(model_bundle, zero_vector):
    """Prediction on a zero vector should return without error."""
    if "pipeline" in model_bundle:
        # Sklearn pipeline bundle
        pipeline = model_bundle["pipeline"]
        probs = pipeline.predict_proba(zero_vector)
        assert probs.shape[0] == 1
        assert probs.shape[1] >= 2
        assert np.isclose(probs.sum(), 1.0, atol=1e-3)
    elif model_bundle.get("model_type") == "centroid":
        # Centroid-based bundle
        labels = list(model_bundle["labels"])
        assert len(labels) >= 2
        vector = zero_vector.reshape(-1).astype(np.float32)
        for label in labels:
            centroid = np.asarray(model_bundle["centroids"][label], dtype=np.float32)
            scales = np.asarray(model_bundle["scales"][label], dtype=np.float32)
            scales = np.where(np.abs(scales) < 1e-6, 1.0, scales)
            distance = np.linalg.norm((vector - centroid) / scales)
            confidence = 1.0 / (1.0 + float(distance))
            assert 0.0 <= confidence <= 1.0
    else:
        pytest.fail(f"Unsupported model type: {model_bundle.get('model_type')}")


def test_labels_present(model_bundle):
    """Model bundle should contain a non-empty labels list."""
    labels = model_bundle.get("labels", [])
    assert len(labels) >= 2, "Expected at least 2 gesture labels."


def test_expected_label_set(model_bundle):
    """Model should recognize exactly the 5 expected gesture classes."""
    labels = set(model_bundle.get("labels", []))
    assert labels == EXPECTED_LABELS, (
        f"Expected labels {EXPECTED_LABELS}, got {labels}"
    )


def test_training_metadata_present(model_bundle):
    """Model bundle should contain training metadata for traceability."""
    if model_bundle.get("model_type") == "centroid":
        pytest.skip("Centroid fallback bundle does not have training metadata.")
    metadata = model_bundle.get("training_metadata")
    assert metadata is not None, "Missing training_metadata in model bundle"
    assert "trained_at" in metadata
    assert "test_accuracy" in metadata
    assert metadata["test_accuracy"] > 0.8, "Test accuracy should be above 80%"


def test_prediction_deterministic(model_bundle, zero_vector):
    """Two consecutive predictions on the same input should match."""
    if "pipeline" not in model_bundle:
        pytest.skip("Only applicable to sklearn pipeline bundles.")
    pipeline = model_bundle["pipeline"]
    pred1 = pipeline.predict(zero_vector)[0]
    pred2 = pipeline.predict(zero_vector)[0]
    assert pred1 == pred2
