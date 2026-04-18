"""ONNX export utility for VERS gesture classifier.

Provides a path to convert trained sklearn pipelines or custom models
into ONNX format for edge deployment (Jetson Nano, CPU-only inference).

Usage:
    python -m src.infrastructure.onnx_export

This is an optional optimisation layer — the system runs perfectly
without ONNX using the deterministic physics heuristic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vers.infrastructure.onnx")

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "gesture_classifier.pkl"
ONNX_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "gesture_classifier.onnx"


def export_sklearn_to_onnx(
    model_path: Path = MODEL_PATH,
    output_path: Path = ONNX_PATH,
    n_features: int = 63,
) -> Optional[Path]:
    """Convert a trained sklearn pipeline to ONNX format.

    Requires ``skl2onnx`` to be installed:
        pip install skl2onnx

    Returns the path to the exported .onnx file, or None on failure.
    """
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.error(
            "skl2onnx is required for ONNX export. Install with: pip install skl2onnx"
        )
        return None

    if not model_path.exists():
        logger.error("Model file not found at %s", model_path)
        return None

    bundle = joblib.load(model_path)
    pipeline = bundle.get("pipeline")
    if pipeline is None:
        logger.error("Model bundle does not contain a 'pipeline' key.")
        return None

    initial_type = [("input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info("ONNX model exported to %s", output_path)
        return output_path

    except Exception as exc:
        logger.error("ONNX export failed: %s", exc)
        return None


def load_onnx_session(onnx_path: Path = ONNX_PATH) -> Any:
    """Load an ONNX Runtime inference session with CPU-only fallback.

    Returns the session object, or None if ONNX Runtime is unavailable.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX Runtime session loaded from %s", onnx_path)
        return session

    except ImportError:
        logger.warning("onnxruntime is not installed — using default inference path.")
        return None
    except Exception as exc:
        logger.warning("Failed to load ONNX session: %s", exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = export_sklearn_to_onnx()
    if result:
        print(f"✅ Exported to {result}")
    else:
        print("❌ Export failed. See logs for details.")
