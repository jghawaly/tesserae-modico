"""
FiFTy: a faithful PyTorch port of the original Keras CNN.

Reference: Mittal et al., "FiFTy: Large-scale File Fragment Type
Identification using Neural Networks" (arXiv:1908.06148).

Architecture (Scenario 1, 4 KB input):
    Embedding(256, 32)
    -> Conv1D(128, k=19) -> LeakyReLU(0.3) -> MaxPool1D(6)
    -> Conv1D(128, k=19) -> LeakyReLU(0.3) -> MaxPool1D(6)
    -> GlobalAveragePooling1D -> Dropout(0.1)
    -> Dense(256) -> LeakyReLU(0.3) -> Dense(num_classes)

Hyperparameters and weight initializations match the published Keras
implementation exactly: Glorot uniform for Dense / Conv, uniform(-0.05, 0.05)
for Embedding, zeros for biases.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FIFTY_CONFIGS = {
    "4096_scenario1": {
        "embed_dim": 32,
        "num_layers": 2,
        "num_filters": 128,
        "kernel_size": 19,
        "pool_size": 6,
        "dense_dim": 256,
    },
    "512_scenario1": {
        "embed_dim": 64,
        "num_layers": 1,
        "num_filters": 128,
        "kernel_size": 27,
        "pool_size": 4,
        "dense_dim": 256,
    },
}


class FiFTy(nn.Module):
    """FiFTy file fragment classifier.

    Args:
        num_classes: Output classes.
        block_size: Input fragment size (512 or 4096).
        embed_dim, num_layers, num_filters, kernel_size, pool_size, dense_dim:
            Architectural hyperparameters; defaults are Scenario-1 values.
        dropout: Dropout before the final dense layers.

    Forward shape: ``[B, L]`` byte tensor -> ``[B, num_classes]`` logits.
    """

    def __init__(
        self,
        num_classes: int = 75,
        block_size: int = 4096,
        embed_dim: int = 32,
        num_layers: int = 2,
        num_filters: int = 128,
        kernel_size: int = 19,
        pool_size: int = 6,
        dense_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.block_size = block_size

        self.embedding = nn.Embedding(256, embed_dim)

        self.conv_blocks = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, num_filters, kernel_size, padding=0),
                    nn.LeakyReLU(0.3),
                    nn.MaxPool1d(pool_size),
                )
            )
            in_channels = num_filters

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, dense_dim)
        self.fc2 = nn.Linear(dense_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        for block in self.conv_blocks:
            conv = block[0]
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), 0.3)
        return self.fc2(x)


def create_fifty_model(
    num_classes: int = 75,
    block_size: int = 4096,
    **kwargs,
) -> FiFTy:
    """Factory: FiFTy with the published Scenario-1 hyperparameters."""
    config_key = f"{block_size}_scenario1"
    if config_key not in FIFTY_CONFIGS:
        raise ValueError(
            f"No published config for block_size={block_size}. "
            f"Available: {list(FIFTY_CONFIGS)}"
        )
    params = FIFTY_CONFIGS[config_key].copy()
    params.update(kwargs)
    return FiFTy(num_classes=num_classes, block_size=block_size, **params)
