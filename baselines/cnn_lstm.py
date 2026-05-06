"""
CNN+LSTM baseline.

Reference: Zhu et al., "File Fragment Type Identification Based on CNN and
LSTM" (ICDSP 2023, https://doi.org/10.1145/3585542.3585545).

Architecture:
    Embedding(256, 64)
    -> 2x [Conv1D(128, k=27) -> BatchNorm -> LeakyReLU(0.3) -> MaxPool(s=2)]
    -> LSTM(hidden=256)  (last hidden state)
    -> Linear(512) -> ReLU -> Linear(num_classes)
"""

from typing import Optional

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM file fragment classifier.

    Args:
        num_classes: Output classes.
        emb_dim: Byte embedding width.
        conv_channels: Channel count for both conv layers.
        kernel_size: Conv1D kernel size.
        lstm_hidden: LSTM hidden state width.
        fc_hidden: Hidden width of the classifier head.

    Forward shape: ``[B, L]`` byte tensor -> ``[B, num_classes]`` logits.
    """

    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 64,
        conv_channels: int = 128,
        kernel_size: int = 27,
        lstm_hidden: int = 256,
        fc_hidden: int = 512,
    ):
        super().__init__()

        self.embedding = nn.Embedding(256, emb_dim)

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n.squeeze(0))


def create_cnn_lstm_model(num_classes: int, **kwargs) -> CNNLSTM:
    """Factory: CNN-LSTM with the published hyperparameters."""
    return CNNLSTM(num_classes=num_classes, **kwargs)
