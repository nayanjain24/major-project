"""LSTM-based gesture recognition model.

This module defines a sequence model for gesture recognition from keypoints.
Expected input per frame is a flattened vector of body/hand landmarks.
"""

from __future__ import annotations

import torch
from torch import nn


class GestureLSTMClassifier(nn.Module):
    """Classify gesture sequences using an LSTM encoder.

    Input:
        x with shape [batch_size, sequence_length, input_dim]
        where `input_dim` contains concatenated body + hand keypoints.

    Output:
        Class probabilities with shape [batch_size, num_classes].
    """

    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Batch-normalize keypoint features before sequence modeling.
        # We normalize across feature channels by flattening [B, T, C] -> [B*T, C].
        self.input_bn = nn.BatchNorm1d(input_dim)

        # LSTM models temporal dependencies between keypoint frames.
        # LSTM dropout is only active when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Classification head: normalize + regularize + project to class logits.
        self.classifier_bn = nn.BatchNorm1d(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gesture class probabilities from keypoint sequences.

        Args:
            x: Tensor of shape [batch_size, sequence_length, input_dim].

        Returns:
            Tensor of shape [batch_size, num_classes] with probabilities.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [batch, seq_len, input_dim], got {tuple(x.shape)}"
            )

        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {feat_dim}")

        # Apply input batch norm on feature channels for each time step.
        x = x.reshape(batch_size * seq_len, feat_dim)
        x = self.input_bn(x)
        x = x.reshape(batch_size, seq_len, feat_dim)

        # Run the temporal encoder.
        _, (hidden_n, _) = self.lstm(x)

        # Select the final hidden state from the last layer.
        # For bidirectional LSTM, concatenate forward/backward last-layer states.
        if self.bidirectional:
            forward_last = hidden_n[-2]
            backward_last = hidden_n[-1]
            sequence_embedding = torch.cat([forward_last, backward_last], dim=1)
        else:
            sequence_embedding = hidden_n[-1]

        # Classification head with normalization and dropout.
        sequence_embedding = self.classifier_bn(sequence_embedding)
        sequence_embedding = self.dropout(sequence_embedding)
        logits = self.fc(sequence_embedding)

        # Return probabilities to match dataset/inference usage requirements.
        probabilities = self.softmax(logits)
        return probabilities

