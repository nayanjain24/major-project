"""Multimodal Fusion Engine for VERS v2.0.

Fuses two independent signal channels — hand gesture classification and
facial emotion distress — into a single unified Severity Score using a
weighted linear combination:

    severity = (gesture_confidence × W_GESTURE) + (emotion_distress × W_EMOTION)

The severity score is then bucketed into discrete threat levels that drive
the alerting pipeline and dashboard indicators.

Architecture note: this module is intentionally stateless.  All temporal
context (smoothing, cooldowns) is handled upstream by the TemporalSmoother
and the alert dispatcher respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Fusion weights — gesture is weighted higher because the physics-based
# classifier has near-perfect precision, while DeepFace emotion scores
# can be noisy under variable lighting.
# ---------------------------------------------------------------------------
W_GESTURE: float = 0.60
W_EMOTION: float = 0.40


@dataclass(frozen=True)
class FusionResult:
    """Immutable output of multimodal fusion."""

    gesture_label: str
    gesture_confidence: float
    dominant_emotion: str
    emotion_distress: float
    severity_score: float
    threat_level: str   # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE"


# ---------------------------------------------------------------------------
# Threat-level bucketing thresholds (applied to the fused severity score)
# ---------------------------------------------------------------------------
_THREAT_THRESHOLDS: list[tuple[float, str]] = [
    (0.80, "CRITICAL"),
    (0.55, "HIGH"),
    (0.30, "MEDIUM"),
    (0.10, "LOW"),
]


def classify_threat(severity: float) -> str:
    """Map a continuous severity score to a discrete threat level string."""
    for threshold, level in _THREAT_THRESHOLDS:
        if severity >= threshold:
            return level
    return "NONE"


def fuse(
    gesture_label: str,
    gesture_confidence: float,
    dominant_emotion: str = "neutral",
    emotion_distress: float = 0.0,
) -> FusionResult:
    """Combine gesture and emotion channels into a unified severity assessment.

    Parameters
    ----------
    gesture_label : str
        Output of the temporal smoother (e.g. "SOS", "ACCIDENT", "NONE").
    gesture_confidence : float
        Mean smoothed confidence (0.0 – 1.0).
    dominant_emotion : str
        DeepFace dominant emotion string (e.g. "fear", "neutral").
    emotion_distress : float
        Weighted emotion distress contribution (0.0 – 1.0) from the
        ``emotion_model.analyze_emotion`` call.

    Returns
    -------
    FusionResult
        Frozen dataclass containing all input signals plus the computed
        ``severity_score`` and ``threat_level``.
    """
    if gesture_label == "NONE":
        # No gesture detected — severity is based purely on emotion (downscaled)
        severity = emotion_distress * W_EMOTION
    else:
        severity = (gesture_confidence * W_GESTURE) + (emotion_distress * W_EMOTION)

    severity = min(severity, 1.0)
    threat = classify_threat(severity)

    return FusionResult(
        gesture_label=gesture_label,
        gesture_confidence=gesture_confidence,
        dominant_emotion=dominant_emotion,
        emotion_distress=emotion_distress,
        severity_score=severity,
        threat_level=threat,
    )
