"""Sliding-window sequence buffer for sign language recognition.

Accumulates MediaPipe hand landmark frames into fixed-length sequences
suitable for LSTM inference.  Each frame is normalised to be wrist-centered
and scale-invariant so the model generalises across hand sizes and camera
distances.

Usage::

    buf = SequenceBuffer(window_size=30)
    buf.push(landmarks_63)           # call every frame
    if buf.ready:
        tensor = buf.get_tensor()    # shape (1, 30, 63) — ready for LSTM
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class SequenceBuffer:
    """Fixed-length sliding window over landmark frames.

    Parameters
    ----------
    window_size : int
        Number of frames per sequence (default 30 ≈ 1 s at 30 FPS).
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = max(window_size, 1)
        self._buffer: deque[np.ndarray] = deque(maxlen=self._window_size)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(landmarks: np.ndarray) -> np.ndarray:
        """Wrist-centre and scale-normalise a 21×3 landmark vector.

        1. Subtract wrist position (landmark 0) from all points.
        2. Divide by the max absolute value to fit within [-1, 1].
        """
        v = landmarks.reshape(21, 3).copy()
        wrist = v[0].copy()
        v -= wrist  # wrist-centred
        scale = np.max(np.abs(v))
        if scale > 1e-6:
            v /= scale
        return v.flatten()  # back to (63,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, landmarks: np.ndarray) -> None:
        """Add a single frame's 63-dim landmark vector to the buffer."""
        vec = np.asarray(landmarks, dtype=np.float32).flatten()
        if vec.shape[0] != 63:
            return  # Silently skip malformed frames
        normalised = self._normalise(vec)
        self._buffer.append(normalised)

    @property
    def ready(self) -> bool:
        """True when the buffer contains a full window of frames."""
        return len(self._buffer) == self._window_size

    def get_sequence(self) -> np.ndarray:
        """Return the current buffer as a (window_size, 63) NumPy array."""
        return np.array(list(self._buffer), dtype=np.float32)

    def get_tensor(self) -> np.ndarray:
        """Return a batch-ready (1, window_size, 63) array for model input."""
        return self.get_sequence()[np.newaxis, ...]

    def reset(self) -> None:
        """Clear the buffer (e.g. when switching modes)."""
        self._buffer.clear()

    @property
    def fill_level(self) -> float:
        """Fraction of the buffer that is filled (0.0 – 1.0)."""
        return len(self._buffer) / self._window_size
