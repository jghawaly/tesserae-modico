"""
Motif encoder: a CNN that scans the fragment in overlapping 512-byte windows
and aggregates per-window features by content-aware importance weighting.

The window-then-aggregate structure decouples capacity from fragment length:
the same encoder runs on a 512 B fragment (one window) or a 16 KB fragment
(many windows) without architectural changes.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifEncoder(nn.Module):
    """Local-pattern encoder over byte windows.

    Runs a multi-scale CNN on each overlapping window, then weights the
    resulting per-window features by a learned importance score and sums.
    Header-like windows tend to dominate; high-entropy / compressed regions
    contribute less.

    Args:
        d_model: Output feature dimension. Must match the other encoders so
            the fusion layer can compare them.
        window_size: Number of bytes per window.
        window_stride: Stride between consecutive windows. Defaults to
            ``window_size // 2`` for 50 percent overlap.
        cnn_channels: Channel widths for the three CNN stages.

    Forward shapes:
        Input  ``x``: ``[B, L]`` of byte values in ``[0, 255]``.
        Output:      ``[B, d_model]``.
    """

    def __init__(
        self,
        d_model: int = 256,
        window_size: int = 512,
        window_stride: int = 256,
        cnn_channels: tuple = (64, 128, 256),
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.window_stride = window_stride

        self.embed_dim = 32
        self.byte_embed = nn.Embedding(256, self.embed_dim)

        c1, c2, c3 = cnn_channels

        # Three CNN stages with progressively wider receptive fields, tuned
        # to detect 1-4 byte, 4-16 byte, and 16-64 byte motifs respectively.
        self.window_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c1),

            nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c2),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c2),

            nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c3),
            nn.Conv1d(c3, c3, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c3),

            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(c3, d_model),
        )

        # Per-window importance head. Outputs raw scores; we softmax across
        # windows to get a probability distribution over the fragment.
        self.importance_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
        )

        self.feature_refiner = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Start with near-uniform importance so early training treats every
        # window the same; the network learns specialization on its own.
        def init_importance(m: nn.Module) -> None:
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

        self.importance_network.apply(init_importance)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of byte sequences into ``[B, d_model]`` features.

        Args:
            x: ``[B, L]`` byte tensor.
            pad_mask: Optional ``[B, L]`` bool tensor, True at padded positions.
        """
        B, L = x.shape

        # Single-window fast path: pad short inputs up to one full window.
        if L <= self.window_size:
            embedded = self.byte_embed(x.long()).transpose(1, 2)
            if L < self.window_size:
                pad_length = self.window_size - L
                embedded = F.pad(embedded, (0, pad_length), mode="constant", value=0)
                if pad_mask is not None:
                    pad_mask = F.pad(pad_mask, (0, pad_length), mode="constant", value=True)
            features = self.window_cnn(embedded)
            return self.feature_refiner(features)

        # Multi-window path. We process every window in parallel by folding
        # the window axis into the batch axis, then run a single CNN forward.
        embedded = self.byte_embed(x.long()).transpose(1, 2)  # [B, E, L]
        windows = embedded.unfold(
            dimension=2, size=self.window_size, step=self.window_stride
        )  # [B, E, num_windows, W]
        num_windows = windows.size(2)

        windows = windows.permute(0, 2, 1, 3).contiguous()
        windows = windows.view(B * num_windows, self.embed_dim, self.window_size)
        window_features = self.window_cnn(windows).view(B, num_windows, self.d_model)

        importance_logits = self.importance_network(window_features).squeeze(-1)

        if pad_mask is not None:
            # A window is "valid" if at least 90 percent of its positions are
            # real bytes. Invalid windows get -inf logits so softmax ignores them.
            pad_mask_unfolded = pad_mask.unfold(
                dimension=1, size=self.window_size, step=self.window_stride
            )
            validity_mask = (~pad_mask_unfolded).float().mean(dim=2) > 0.9
            importance_logits = importance_logits.masked_fill(~validity_mask, -1e9)

        importance_weights = F.softmax(importance_logits, dim=1)
        refined = self.feature_refiner(window_features)
        return (refined * importance_weights.unsqueeze(-1)).sum(dim=1)

    @torch.no_grad()
    def get_window_importance(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the learned importance distribution over windows.

        Useful for visualizing which regions of a fragment the model treats
        as informative. Returns ``[B, num_windows]``.
        """
        B, L = x.shape

        if L <= self.window_size:
            return torch.ones(B, 1, device=x.device)

        embedded = self.byte_embed(x.long())

        # We replay the same window grid the forward path uses, including a
        # final tail window if the stride doesn't reach the end of the input.
        window_starts = list(range(0, L - self.window_size + 1, self.window_stride))
        if not window_starts or window_starts[-1] + self.window_size < L:
            window_starts.append(L - self.window_size)

        feats = []
        for start_idx in window_starts:
            window = embedded[:, start_idx:start_idx + self.window_size, :].transpose(1, 2)
            feats.append(self.window_cnn(window))
        window_features = torch.stack(feats, dim=1)

        importance_logits = torch.cat(
            [self.importance_network(window_features[:, i, :]) for i in range(window_features.size(1))],
            dim=1,
        )
        return F.softmax(importance_logits, dim=1)
