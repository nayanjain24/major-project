"""Feature preprocessing utilities.

This module converts raw MediaPipe landmarks into model-ready tensors.
Typical responsibilities include normalization, padding, and batching.
"""

from __future__ import annotations

from typing import Any

import torch


def landmarks_to_tensor(landmark_payload: Any) -> torch.Tensor:
    """Convert extracted landmarks to a placeholder feature tensor.

    Replace this stub with actual feature engineering logic.
    """
    # Baseline stub: return a zero vector matching `input_dim` in the model.
    return torch.zeros(258, dtype=torch.float32)
