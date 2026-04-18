"""Gesture tracking module using MediaPipe hand landmarks and 3D physics heuristics.

Extracted from the monolithic realtime_vers.py into a standalone vision service
to support domain-driven architecture.  Provides a deterministic, geometry-based
gesture classifier that replaces the legacy Random Forest model — yielding 100 %
reproducible predictions at zero inference cost.
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Deterministic 3D finger-extension heuristic
# ---------------------------------------------------------------------------

def predict_gesture(hand_vector: Any) -> tuple[str, float]:
    """Classify a hand gesture from a 21×3 landmark vector using Euclidean geometry.

    Returns ``(label, confidence)`` where *confidence* is always 1.0 for a
    strong match (physics-based) and 0.0 if no pattern matches.
    """
    if hasattr(hand_vector, "to_numpy"):
        vec = hand_vector.to_numpy()
    else:
        vec = np.asarray(hand_vector)

    v = vec.reshape(21, 3)

    def dist(i: int, j: int = 0) -> float:
        return float(np.linalg.norm(v[i] - v[j]))

    idx_ext = dist(8) > dist(6)
    mid_ext = dist(12) > dist(10)
    rng_ext = dist(16) > dist(14)
    pnk_ext = dist(20) > dist(18)
    thumb_ext = dist(4, 17) > dist(3, 17)

    if thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return "SOS", 1.0
    if not thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return "MEDICAL", 1.0
    if not thumb_ext and idx_ext and mid_ext and not rng_ext and not pnk_ext:
        return "EMERGENCY", 1.0
    if thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return "SAFE", 1.0
    if not thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return "ACCIDENT", 1.0

    return "NONE", 0.0
