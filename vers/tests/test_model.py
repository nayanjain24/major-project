"""Smoke test for the trained gesture classifier.

Loads models/gesture_classifier.pkl and verifies the model can produce
predictions on a zero vector without exceptions. Handles both sklearn
pipeline bundles and centroid-based bundles.
"""

from pathlib import Path

import numpy as np
import pytest


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_classifier.pkl"


@pytest.fixture
def model_bundle():
    if not MODEL_PATH.exists():
        pytest.skip("Model file not present; train the classifier first.")
    import joblib
    return joblib.load(MODEL_PATH)


def test_prediction_zero_vector(model_bundle):
    """Prediction on a zero vector should return without error."""
    zero_vector = np.zeros((1, 63), dtype=np.float32)

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
