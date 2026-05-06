"""
ByteRCNN: a faithful PyTorch port of Skracic et al. (IEEE Access, 2023).

Original architecture:
    Embedding(256, 16) -> Dropout(0.1)
    -> 2-layer bidirectional GRU(64)
    -> Concatenate(embedding, GRU output)
    -> For each kernel in [9, 27, 40, 65]:
         Conv1D(64) -> LeakyReLU(0.3)
         MaxPool1D(4)  (applied to the concatenated features directly)
         Conv1D(64) -> LeakyReLU(0.3)
    -> GlobalAveragePooling1D + GlobalMaxPool1D on each branch
    -> Concatenate -> Dense(1024) -> Dropout(0.1) -> Dense(512) -> Dense(num_classes)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BYTERCNN_CONFIGS = {
    "512": {
        "embed_dim": 16,
        "rnn_size": 64,
        "cnn_size": 64,
        "kernels": [9, 27, 40, 65],
    },
    "4096": {
        "embed_dim": 16,
        "rnn_size": 64,
        "cnn_size": 64,
        "kernels": [9, 27, 40, 65],
    },
}


class ByteRCNN(nn.Module):
    """ByteRCNN: GRU + multi-kernel CNN baseline.

    Args:
        num_classes: Number of output classes.
        block_size: Input block size in bytes (512 or 4096).
        embed_dim: Byte embedding width.
        rnn_size: Per-direction GRU hidden size; bidirectional output is 2x.
        cnn_size: Convolutional channel count per branch.
        kernels: Kernel sizes for the parallel conv branches.
        dropout: Dropout used after embedding and between FC layers.

    Forward shape: ``[B, L]`` byte tensor -> ``[B, num_classes]`` logits.
    """

    def __init__(
        self,
        num_classes: int = 75,
        block_size: int = 512,
        embed_dim: int = 16,
        rnn_size: int = 64,
        cnn_size: int = 64,
        kernels: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kernels is None:
            kernels = [9, 27, 40, 65]

        self.num_classes = num_classes
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.rnn_size = rnn_size
        self.cnn_size = cnn_size
        self.kernels = kernels

        self.embedding = nn.Embedding(256, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Fused two-layer bidirectional GRU. cuDNN handles this much better
        # than two separate stacked GRU modules at this sequence length.
        self.gru = nn.GRU(
            embed_dim,
            rnn_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        concat_dim = embed_dim + rnn_size * 2

        # Each kernel gets two conv branches; we also keep a maxpool branch
        # of the concatenated input (no conv) for the third pooling source.
        self.conv_branches = nn.ModuleList()
        for k in kernels:
            self.conv_branches.append(nn.Conv1d(concat_dim, cnn_size, k))
            self.conv_branches.append(nn.Conv1d(concat_dim, cnn_size, k))

        self.maxpool = nn.MaxPool1d(4)

        n_conv_branches = len(kernels) * 2
        n_pool_branches = len(kernels)
        pool_features = (n_conv_branches * cnn_size + n_pool_branches * concat_dim) * 2

        self.fc1 = nn.Linear(pool_features, 1024)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for conv in self.conv_branches:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.embedding(x)
        emb_drop = self.emb_dropout(emb)

        gru_out, _ = self.gru(emb_drop)
        concat = torch.cat([emb, gru_out], dim=2)
        concat_t = concat.transpose(1, 2)

        # Match the TF original branch order: [Conv_A, MaxPool, Conv_B] per kernel.
        branches = []
        conv_idx = 0
        for _ in self.kernels:
            c1 = F.leaky_relu(self.conv_branches[conv_idx](concat_t), 0.3)
            conv_idx += 1
            mp = self.maxpool(concat_t)
            c2 = F.leaky_relu(self.conv_branches[conv_idx](concat_t), 0.3)
            conv_idx += 1
            branches.extend([c1, mp, c2])

        # Average pool first, then max pool; concatenate in that order to
        # match the published checkpoint layout.
        avg_pooled = [b.mean(dim=2) for b in branches]
        max_pooled = [b.max(dim=2)[0] for b in branches]
        x = torch.cat(avg_pooled + max_pooled, dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_bytercnn_model(
    num_classes: int = 75,
    block_size: int = 512,
    **kwargs,
) -> ByteRCNN:
    """Factory: ByteRCNN with the published Scenario-1 hyperparameters."""
    config_key = str(block_size) if str(block_size) in BYTERCNN_CONFIGS else "512"
    params = BYTERCNN_CONFIGS[config_key].copy()
    params.update(kwargs)
    return ByteRCNN(num_classes=num_classes, block_size=block_size, **params)
