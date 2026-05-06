"""
Context encoder: a hierarchical byte-level transformer.

Stage 1 runs shifted-window attention at the byte level (cheap, local).
Optional stages 2 and 3 progressively downsample with depthwise convolutions
and run attention on the compressed sequence, which gives global mixing
without paying full quadratic attention on raw bytes.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ShiftedWindowAttention(nn.Module):
    """Swin-style local attention with cyclic shift between layers.

    Operating on small windows keeps attention linear in sequence length;
    the cyclic shift lets information flow across window boundaries on
    successive applications.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 64,
        shift_size: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        pad_len = (self.window_size - L % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=True)

        L_padded = x.size(1)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
            if mask is not None:
                mask = torch.roll(mask, shifts=-self.shift_size, dims=1)

        num_windows = L_padded // self.window_size
        x = x.view(B, num_windows, self.window_size, D)
        x_flat = x.view(B * num_windows, self.window_size, D)

        if mask is not None:
            mask_flat = mask.view(B, num_windows, self.window_size).view(
                B * num_windows, self.window_size
            )
        else:
            mask_flat = None

        attended, _ = self.attention(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        x = attended.contiguous().view(B, num_windows, self.window_size, D).view(B, L_padded, D)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        if pad_len > 0:
            x = x[:, :L, :]

        return self.dropout(x) + x


class _ByteEmbedding(nn.Module):
    """Byte value embedding plus a learnable positional code."""

    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()

        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.byte_embed.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        byte_emb = self.byte_embed(x.long())

        # If the input is longer than max_len, interpolate the positional
        # codes so we never crash on slightly oversized fragments.
        if L <= self.pos_embed.size(1):
            pos_emb = self.pos_embed[:, :L, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=L,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return self.dropout(self.norm(byte_emb + pos_emb))


class _ConvDownsample(nn.Module):
    """Depthwise convolutional downsample that doubles the channel dim."""

    def __init__(self, dim: int, downsample_factor: int = 4):
        super().__init__()

        self.conv = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim * 2)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.activation(self.norm(self.conv(x.transpose(1, 2)).transpose(1, 2)))

        if mask is not None:
            # Any padding inside a kernel makes the whole pooled slot padded.
            mask = F.max_pool1d(
                mask.float().unsqueeze(1),
                kernel_size=self.conv.kernel_size[0],
                stride=self.conv.stride[0],
            ).squeeze(1).bool()

        return x, mask


class ContextEncoder(nn.Module):
    """Sequential / contextual byte encoder.

    A stack of one to three hierarchical stages. Stage 1 always runs at the
    byte level; later stages downsample by 4 each, so a 4 KB fragment shrinks
    to 256 then 64 tokens before global attention.

    Args:
        d_model: Output feature dimension.
        max_len: Largest fragment length we expect. Sets positional code size.
        embed_dim: Internal channel width at the first (byte-level) stage.
        num_layers: 1, 2, or 3. Higher means more global mixing.

    Forward shapes:
        Input  ``x``: ``[B, L]`` byte tensor.
        Output:      ``[B, d_model]``.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 16384,
        embed_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.byte_embedding = _ByteEmbedding(embed_dim, max_len)

        self.stage1_attention = _ShiftedWindowAttention(
            dim=embed_dim,
            window_size=64,
            shift_size=32,
            num_heads=max(1, embed_dim // 32),
        )
        self.stage1_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.stage1_norm = nn.LayerNorm(embed_dim)

        current_dim = embed_dim

        if num_layers >= 2:
            self.downsample1 = _ConvDownsample(embed_dim, downsample_factor=4)
            current_dim = embed_dim * 2
            self.stage2_attention = _ShiftedWindowAttention(
                dim=current_dim,
                window_size=64,
                shift_size=32,
                num_heads=max(1, current_dim // 32),
            )
            self.stage2_ffn = nn.Sequential(
                nn.Linear(current_dim, current_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(current_dim * 4, current_dim),
            )
            self.stage2_norm = nn.LayerNorm(current_dim)

        if num_layers >= 3:
            self.downsample2 = _ConvDownsample(current_dim, downsample_factor=4)
            current_dim = current_dim * 2
            self.stage3_attention = nn.MultiheadAttention(
                embed_dim=current_dim,
                num_heads=max(1, current_dim // 32),
                dropout=0.1,
                batch_first=True,
            )
            self.stage3_ffn = nn.Sequential(
                nn.Linear(current_dim, current_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(current_dim * 4, current_dim),
            )
            self.stage3_norm = nn.LayerNorm(current_dim)

        self.final_dim = current_dim

        # Attention pooling with a learned query distills the variable-length
        # sequence into a single feature vector.
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=current_dim,
            num_heads=max(1, current_dim // 32),
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, current_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(current_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = x.shape

        h = self.byte_embedding(x)
        current_mask = pad_mask

        h = self.stage1_attention(h, mask=current_mask)
        h = h + self.stage1_ffn(self.stage1_norm(h))

        if self.num_layers >= 2:
            h, current_mask = self.downsample1(h, current_mask)
            h = self.stage2_attention(h, mask=current_mask)
            h = h + self.stage2_ffn(self.stage2_norm(h))

        if self.num_layers >= 3:
            h, current_mask = self.downsample2(h, current_mask)
            h_attn, _ = self.stage3_attention(h, h, h, key_padding_mask=current_mask)
            h = h_attn + self.stage3_ffn(self.stage3_norm(h_attn))

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attention(query, h, h, key_padding_mask=current_mask)
        return self.output_proj(pooled.squeeze(1))
