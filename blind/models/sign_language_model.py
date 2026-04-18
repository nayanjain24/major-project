"""PyTorch model definitions for sign language translation.

This module is responsible for:
1. Defining neural network architectures.
2. Encapsulating forward passes for sequence/frame features.
3. Serving as the core ML block used by training and inference code.
"""

from __future__ import annotations

import torch
from torch import nn


class SignLanguageClassifier(nn.Module):
    """Simple baseline classifier for extracted landmark features.

    Replace/extend this with sequence models (LSTM/Transformer/TCN)
    once your feature pipeline is stable.
    """

    def __init__(self, input_dim: int = 258, hidden_dim: int = 256, num_classes: int = 20) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Tensor shaped `[batch, input_dim]` for this baseline model.

        Returns:
            Class logits shaped `[batch, num_classes]`.
        """
        return self.network(x)
