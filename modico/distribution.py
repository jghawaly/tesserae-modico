"""
Distribution encoder: byte histogram + global entropy + windowed-entropy CDF.

The CDF representation is the key piece: instead of a few hand-picked entropy
statistics, we report, for each of K entropy thresholds, what fraction of the
fragment's windows fall below that threshold. This is normalized by definition,
so it stays comparable across fragment sizes, and it cleanly separates
high-entropy compressed/encrypted data from structured low-entropy regions.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionEncoder(nn.Module):
    """Statistical encoder over byte distributions.

    Concatenates three views of the fragment and projects them to ``d_model``:
      * a 256-dim normalized byte histogram,
      * a single scalar global entropy,
      * an entropy-CDF vector of length ``entropy_cdf_points``.

    Args:
        d_model: Output feature dimension.
        entropy_window_size: Window size used for the local entropy profile.
        entropy_cdf_points: Number of entropy thresholds in the CDF.

    Forward shapes:
        Input  ``x``: ``[B, L]`` byte tensor.
        Output:      ``[B, d_model]``.
    """

    def __init__(
        self,
        d_model: int = 256,
        entropy_window_size: int = 64,
        entropy_cdf_points: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.entropy_window_size = entropy_window_size
        self.entropy_cdf_points = entropy_cdf_points

        self.histogram_dim = 256
        self.global_stats_dim = 1
        self.windowed_entropy_dim = entropy_cdf_points
        self.total_stats_dim = (
            self.histogram_dim + self.global_stats_dim + self.windowed_entropy_dim
        )

        # Sample the entropy axis evenly from 0 to 8 bits (the max for bytes).
        self.register_buffer("entropy_levels", torch.linspace(0, 8, entropy_cdf_points))

        self.projector = nn.Sequential(
            nn.Linear(self.total_stats_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

        self.eps = 1e-10

    def compute_entropy(self, counts: torch.Tensor) -> torch.Tensor:
        """Shannon entropy in bits over the last axis of a count vector."""
        total = counts.sum(dim=-1, keepdim=True)
        probs = counts / (total + self.eps)
        log_probs = torch.log2(probs + self.eps)
        return -(probs * log_probs).sum(dim=-1)

    def compute_windowed_entropy(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Entropy in sliding windows of ``entropy_window_size`` bytes.

        Returns a ``[B, num_windows]`` tensor. If the input is shorter than
        one window we fall back to the global entropy of the whole fragment.
        """
        B, L = x.shape

        if L < self.entropy_window_size:
            x_long = x.long().clamp(0, 255)
            one_hot = F.one_hot(x_long, num_classes=256).float()
            if pad_mask is not None:
                one_hot = one_hot * (~pad_mask).unsqueeze(-1).float()
            hist = one_hot.sum(dim=1)
            return self.compute_entropy(hist).unsqueeze(1)

        stride = self.entropy_window_size // 2
        x_long = x.long().clamp(0, 255)
        windows = x_long.unfold(dimension=1, size=self.entropy_window_size, step=stride)
        one_hot = F.one_hot(windows, num_classes=256).float()

        if pad_mask is not None:
            mask_unfolded = pad_mask.unfold(
                dimension=1, size=self.entropy_window_size, step=stride
            )
            one_hot = one_hot * (~mask_unfolded).unsqueeze(-1).float()

        histograms = one_hot.sum(dim=2)
        total = histograms.sum(dim=-1, keepdim=True) + self.eps
        probs = histograms / total
        log_probs = torch.log2(probs + self.eps)
        return -(probs * log_probs).sum(dim=-1)

    def compute_entropy_cdf(self, window_entropies: torch.Tensor) -> torch.Tensor:
        """Empirical CDF of the per-window entropies, sampled at ``entropy_levels``.

        For each level h, we report what fraction of windows have entropy <= h.
        Output shape is ``[B, entropy_cdf_points]``.
        """
        levels = self.entropy_levels.view(1, 1, -1)
        expanded = window_entropies.unsqueeze(-1)
        below = (expanded <= levels).float()
        return below.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = x.shape

        # Normalized byte histogram.
        x_long = x.long().clamp(0, 255)
        one_hot = F.one_hot(x_long, num_classes=256).float()
        if pad_mask is not None:
            one_hot = one_hot * (~pad_mask).unsqueeze(-1).float()
            valid_lengths = (~pad_mask).sum(dim=1).float()
        else:
            valid_lengths = torch.full((B,), L, device=x.device, dtype=torch.float)

        byte_hist = one_hot.sum(dim=1)
        normalized_hist = byte_hist / (valid_lengths.unsqueeze(1) + self.eps)

        # Global entropy is computed from the un-normalized counts (any
        # consistent scaling works; compute_entropy normalizes internally).
        global_entropy = self.compute_entropy(byte_hist).unsqueeze(1)

        # Entropy CDF over windows.
        window_entropies = self.compute_windowed_entropy(x, pad_mask)
        entropy_cdf = self.compute_entropy_cdf(window_entropies)

        features = torch.cat([normalized_hist, global_entropy, entropy_cdf], dim=1)
        return self.projector(features)

    @torch.no_grad()
    def get_entropy_profile(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the raw windowed entropy values for visualization."""
        return self.compute_windowed_entropy(x, pad_mask)

    @torch.no_grad()
    def get_entropy_cdf(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the entropy CDF for visualization."""
        return self.compute_entropy_cdf(self.compute_windowed_entropy(x, pad_mask))
