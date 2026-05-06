"""
Attentive fusion of the three encoders.

Each encoder emits a ``[B, d_model]`` summary. We treat them as a tiny
3-token sequence, run cross-attention so each encoder can rewrite its own
view in light of the other two, then mix them with input-dependent weights.
"""

from typing import List

import torch
import torch.nn as nn


class AttentiveFusion(nn.Module):
    """Cross-attention plus dynamic-weight fusion of encoder outputs.

    The dynamic-weight head looks at the concatenated post-attention features
    and emits a softmax over encoders, so the model can rely more on the
    motif encoder for headers, the distribution encoder for entropy-pure
    fragments, and so on.

    Args:
        num_branches: Number of encoder branches to fuse (3 in MoDiCo).
        d_model: Encoder output dimension. Same as ``d_model`` everywhere.
        dropout: Dropout in the refinement MLP.

    Forward shapes:
        Input ``branches``: list of ``[B, d_model]`` tensors.
        Output:             ``[B, d_model]``.
    """

    def __init__(self, num_branches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_branches = num_branches
        self.d_model = d_model

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(d_model * num_branches, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_branches),
            nn.Softmax(dim=-1),
        )

        self.refiner = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        B = branches[0].shape[0]

        stacked = torch.stack(branches, dim=1)  # [B, num_branches, d_model]
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # reshape (not view) because attention may return a non-contiguous tensor.
        weights = self.weight_generator(attended.reshape(B, -1))
        weighted = (attended * weights.unsqueeze(-1)).sum(dim=1)
        return self.refiner(weighted)
