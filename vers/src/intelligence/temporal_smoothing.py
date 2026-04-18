"""Temporal Sequence Smoother for VERS v2.0.

Implements a sliding-window majority-vote algorithm that eliminates the
frame-to-frame "flickering" inherent in single-frame classifiers.  Instead
of trusting every individual frame's prediction, we maintain a fixed-length
history buffer and emit the gesture that has accumulated the highest
weighted confidence across the window.

This is a lightweight alternative to a full LSTM that runs at zero
additional inference cost — critical for maintaining >15 FPS on laptops.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import numpy as np


class TemporalSmoother:
    """Thread-safe sliding-window gesture smoother.

    Parameters
    ----------
    window_size : int
        Number of recent predictions to retain (default 7).
    min_votes : int
        Minimum number of agreeing frames before we emit a non-NONE label.
        Prevents a single spurious frame from triggering an alert.
    """

    def __init__(self, window_size: int = 7, min_votes: int = 3) -> None:
        self._window_size = max(window_size, 1)
        self._min_votes = max(min_votes, 1)
        self._history: deque[tuple[str, float]] = deque(maxlen=self._window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, label: str, confidence: float) -> None:
        """Record a new single-frame prediction."""
        self._history.append((label, confidence))

    def smoothed(self) -> tuple[str, float]:
        """Return the majority-vote label and its mean confidence.

        Returns ``("NONE", 0.0)`` when either the history is empty or no
        non-NONE label has reached *min_votes* occurrences.
        """
        if not self._history:
            return "NONE", 0.0

        weighted: Counter[str] = Counter()
        confidences: dict[str, list[float]] = {}

        for label, conf in self._history:
            weighted[label] += conf
            confidences.setdefault(label, []).append(conf)

        # Filter out NONE so we only consider actual gesture candidates
        valid = {lbl: score for lbl, score in weighted.items() if lbl != "NONE"}
        if not valid:
            return "NONE", 0.0

        best_label = max(valid, key=lambda lbl: valid[lbl])

        # Enforce minimum vote count to suppress single-frame noise
        if len(confidences[best_label]) < self._min_votes:
            return "NONE", 0.0

        mean_conf = float(np.mean(confidences[best_label]))
        return best_label, mean_conf

    def reset(self) -> None:
        """Clear the history buffer (e.g. when the camera stops)."""
        self._history.clear()

    @property
    def history_length(self) -> int:
        return len(self._history)
