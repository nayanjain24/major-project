"""Facial Emotion Recognition (FER) module for VERS v2.0.

Uses the DeepFace library to perform real-time facial emotion analysis on
cropped face regions extracted from MediaPipe Face Mesh landmarks.

Design decisions:
  - We call DeepFace.analyze with enforce_detection=False so that frames
    where the face detector fails do not crash the pipeline.
  - Emotion analysis is intentionally run on every Nth frame (controlled by
    the caller) to preserve >15 FPS on laptop hardware.
  - Results are cached in a thread-safe dict so intermediate frames
    can read the last known emotion without recomputing.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("vers.vision.emotion")

# ---------------------------------------------------------------------------
# Lazy-loaded DeepFace — import is deferred to avoid paying the 2-3 s
# start-up cost until the first actual emotion analysis call.
# ---------------------------------------------------------------------------
_deepface_module: Any = None
_deepface_lock = threading.Lock()


def _get_deepface() -> Any:
    global _deepface_module
    if _deepface_module is None:
        with _deepface_lock:
            if _deepface_module is None:  # double-check
                try:
                    from deepface import DeepFace
                    _deepface_module = DeepFace
                    logger.info("DeepFace loaded successfully.")
                except ImportError:
                    logger.warning(
                        "deepface is not installed — emotion recognition "
                        "will fall back to the heuristic distress scorer."
                    )
                    _deepface_module = False  # sentinel: unavailable
    return _deepface_module


# Emotion → distress weight mapping (higher = more distressing)
EMOTION_DISTRESS_WEIGHTS: dict[str, float] = {
    "angry": 0.85,
    "fear": 1.0,
    "sad": 0.70,
    "surprise": 0.50,
    "disgust": 0.60,
    "happy": 0.05,
    "neutral": 0.10,
}


def analyze_emotion(
    frame_rgb: np.ndarray,
    *,
    detector_backend: str = "skip",
) -> dict[str, Any]:
    """Run facial emotion recognition on *frame_rgb* (H×W×3 RGB uint8).

    Returns a dict with keys:
      - ``dominant_emotion`` (str): e.g. "fear", "angry", "neutral"
      - ``emotion_scores`` (dict[str, float]): per-emotion percentage 0-100
      - ``distress_contribution`` (float): 0.0 – 1.0 scalar representing
        the weighted distress implied by the detected emotion distribution.

    If DeepFace is unavailable or analysis fails, returns safe defaults.
    """
    deepface = _get_deepface()
    default: dict[str, Any] = {
        "dominant_emotion": "neutral",
        "emotion_scores": {},
        "distress_contribution": 0.0,
    }

    if deepface is False or deepface is None:
        return default

    try:
        # DeepFace expects BGR by default but we pass RGB and skip its own
        # face detector (MediaPipe already found the face for us).
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = deepface.analyze(
            bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=detector_backend,
            silent=True,
        )

        # DeepFace may return a list of dicts (one per face) in newer versions
        if isinstance(results, list):
            if not results:
                return default
            result = results[0]
        else:
            result = results

        emotion_scores: dict[str, float] = result.get("emotion", {})
        dominant: str = result.get("dominant_emotion", "neutral")

        # Compute weighted distress contribution from the full distribution
        total_weight = 0.0
        for emotion_name, pct in emotion_scores.items():
            w = EMOTION_DISTRESS_WEIGHTS.get(emotion_name.lower(), 0.1)
            total_weight += w * (pct / 100.0)

        return {
            "dominant_emotion": dominant.lower(),
            "emotion_scores": emotion_scores,
            "distress_contribution": min(total_weight, 1.0),
        }

    except Exception as exc:
        logger.debug("Emotion analysis failed: %s", exc)
        return default
