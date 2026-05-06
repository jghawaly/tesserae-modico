"""
ByteFormer / ByteResNet baselines (the two variants share most code).

Reference: Liu et al., "ByteNet: Rethinking Multimedia File Fragment
Classification through Visual Perspectives" (IEEE TMM, 2024).

Both variants are dual-branch:

  * Byte branch (BBFE):  Linear(block_size -> 128) + ReLU
  * Image branch (IBFE): a Byte2Image transform turns raw bytes into a small
    grayscale image, which is then fed through PoolFormer-S36 (ByteFormer)
    or ResNet-18 (ByteResNet) for a 512-dim representation.
  * Fusion:              Concat(128 + 512) -> Linear(640 -> num_classes)

The Byte2Image transform maps a 512 B block to a (128, 496) image; for a
4 KB block it produces an 8-channel (8, 128, 496) tensor by tiling the 8
component 512 B blocks across channels.

We expose a unified ``create_byteformer_model(model_type='byteformer'|'byteresnet')``
factory; both variants are exported via the same wrapper that applies
Byte2Image internally so callers can pass raw bytes.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from timm.models.layers import DropPath, trunc_normal_
    HAS_TIMM = True
except ImportError:  # pragma: no cover
    HAS_TIMM = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


NGRAM = 16


# Byte2Image transforms: the CPU paths are used by DataLoader workers; the
# torch path is used at inference time so we can run on the GPU.

if HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _byte2image_numba(block, ngram):
        block_size = block.shape[0]
        out_w = block_size - ngram
        result = np.empty((ngram * 8, out_w), dtype=np.uint8)

        # Bit-shift the bytes to produce 8 sub-byte views with 1-bit stride.
        shifts = np.empty((8, block_size), dtype=np.uint8)
        for j in range(block_size):
            blk = np.uint16(block[j])
            nxt = np.uint16(block[j + 1]) if j + 1 < block_size else np.uint16(0)
            for s in range(8):
                shifts[s, j] = np.uint8(((blk << s) & 0xFF) + ((nxt >> (8 - s)) & 0xFF))

        # Stack ngram consecutive shift vectors so each output column sees an
        # n-gram-long context window.
        for col in range(out_w):
            for s in range(8):
                result[s, col] = shifts[s, col]
            for i in range(1, ngram):
                src_col = (col + i) % block_size
                for s in range(8):
                    result[i * 8 + s, col] = shifts[s, src_col]
        return result


def _byte2image_numpy(block: np.ndarray, ngram: int = NGRAM) -> np.ndarray:
    block_size = len(block)
    blk = block.astype(np.uint16)
    nxt = np.empty_like(blk)
    nxt[:-1] = blk[1:]
    nxt[-1] = 0

    shifts = np.empty((8, block_size), dtype=np.uint8)
    for s in range(8):
        shifts[s] = (((blk << s) & 0xFF) + ((nxt >> (8 - s)) & 0xFF)).astype(np.uint8)

    result = np.empty((ngram * 8, block_size), dtype=np.uint8)
    result[:8] = shifts
    for i in range(1, ngram):
        result[i * 8 : (i + 1) * 8] = np.roll(shifts, -i)

    return result[:, :-ngram]


def byte2image(block: np.ndarray, ngram: int = NGRAM) -> np.ndarray:
    """Convert a raw byte block to a 2D uint8 image.

    Returns ``(ngram * 8, block_size - ngram)``. Uses Numba when available.
    """
    if HAS_NUMBA:
        return _byte2image_numba(block.astype(np.uint8), ngram)
    return _byte2image_numpy(block, ngram)


def byte2image_4k(block: np.ndarray, ngram: int = NGRAM) -> np.ndarray:
    """Same as ``byte2image`` but for 4 KB blocks: 8-channel output."""
    return np.stack(
        [byte2image(block[i * 512 : (i + 1) * 512], ngram) for i in range(8)],
        axis=0,
    )


@torch.no_grad()
def byte2image_torch(block: torch.Tensor, ngram: int = NGRAM) -> torch.Tensor:
    """GPU implementation for batched 512 B blocks.

    Returns ``(B, 1, ngram*8, block_size - ngram)`` normalized to [-1, 1].
    """
    B, L = block.shape
    blk = block.to(torch.int32)
    nxt = torch.zeros_like(blk)
    nxt[:, :-1] = blk[:, 1:]

    shift_list = [((blk << s) & 0xFF) + ((nxt >> (8 - s)) & 0xFF) for s in range(8)]
    shifts = torch.stack(shift_list, dim=1)  # (B, 8, L)

    flat = shifts.reshape(B, -1)
    parts = [flat] + [torch.roll(flat, -i, dims=1) for i in range(1, ngram)]
    stacked = torch.cat(parts, dim=1).reshape(B, ngram * 8, L)

    result = stacked[:, :, :-ngram].float()
    result = (result / 255.0 - 0.5) / 0.5
    return result.unsqueeze(1)


@torch.no_grad()
def byte2image_4k_torch(block: torch.Tensor, ngram: int = NGRAM) -> torch.Tensor:
    """GPU implementation for batched 4 KB blocks: 8-channel output."""
    return torch.cat(
        [byte2image_torch(block[:, i * 512 : (i + 1) * 512], ngram) for i in range(8)],
        dim=1,
    )


# PoolFormer building blocks (only used by the ByteFormer variant).


class _GroupNorm1(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class _Pooling(nn.Module):
    """Average-pool token mixer minus identity (the PoolFormer recipe)."""

    def __init__(self, pool_size: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class _PoolFormerMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class _PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU,
                 norm_layer=_GroupNorm1, drop=0.0, drop_path=0.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = _Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        self.mlp = _PoolFormerMlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


def _build_stages(layers, embed_dims, mlp_ratios, downsamples,
                  pool_size=3, norm_layer=_GroupNorm1, act_layer=nn.GELU,
                  drop_rate=0.0, drop_path_rate=0.0,
                  use_layer_scale=True, layer_scale_init_value=1e-5):
    network = []
    total_blocks = sum(layers)
    block_idx = 0
    for i in range(len(layers)):
        blocks = []
        for _ in range(layers[i]):
            dpr = drop_path_rate * block_idx / max(total_blocks - 1, 1)
            blocks.append(_PoolFormerBlock(
                embed_dims[i], pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            block_idx += 1
        network.append(nn.Sequential(*blocks))

        if i < len(layers) - 1:
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(_PatchEmbed(
                    patch_size=3, stride=2, padding=1,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                ))
    return nn.ModuleList(network)


class _PoolFormerBackbone(nn.Module):
    """PoolFormer-S36 feature extractor: ``layers=[6,6,18,6]``, dims to 512."""

    def __init__(self, in_channel=1, emb=64,
                 layers=(6, 6, 18, 6),
                 embed_dims=None,
                 mlp_ratios=(4, 4, 4, 4),
                 downsamples=(True, True, True, True),
                 pool_size=3,
                 in_patch_size=8, in_stride=8, in_pad=0,
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6):
        super().__init__()
        if embed_dims is None:
            embed_dims = [emb, 128, 320, 512]

        self.patch_embed = _PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=in_channel, embed_dim=embed_dims[0],
        )
        self.network = _build_stages(
            list(layers), embed_dims, list(mlp_ratios), list(downsamples),
            pool_size=pool_size, drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.norm = _GroupNorm1(embed_dims[-1])
        self.out_dim = embed_dims[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.network:
            x = block(x)
        x = self.norm(x)
        return x.mean([-2, -1])


class ByteFormer(nn.Module):
    """ByteFormer: dual-branch network with PoolFormer-S36 image branch."""

    def __init__(self, num_classes: int, block_size: int = 512):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required for ByteFormer / ByteResNet baselines.")

        self.num_classes = num_classes
        self.block_size = block_size

        self.byte_fc = nn.Linear(block_size, 128)
        self.byte_act = nn.ReLU(inplace=True)

        in_channel, emb = (1, 64) if block_size <= 512 else (8, 96)
        self.poolformer = _PoolFormerBackbone(
            in_channel=in_channel, emb=emb,
            layers=(6, 6, 18, 6),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            layer_scale_init_value=1e-6,
        )

        self.classifier = nn.Linear(128 + self.poolformer.out_dim, num_classes)

    def forward(self, raw_1d: torch.Tensor, img_2d: torch.Tensor) -> torch.Tensor:
        byte_feat = self.byte_act(self.byte_fc(raw_1d))
        img_feat = self.poolformer(img_2d)
        return self.classifier(torch.cat([byte_feat, img_feat], dim=1))


class ByteResNet(nn.Module):
    """ByteResNet variant: same byte branch, ResNet-18 image branch."""

    def __init__(self, num_classes: int, block_size: int = 512):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required for ByteFormer / ByteResNet baselines.")
        import timm

        self.num_classes = num_classes
        self.block_size = block_size

        self.byte_fc = nn.Linear(block_size, 128)
        self.byte_act = nn.ReLU(inplace=True)

        in_chans = 1 if block_size <= 512 else 8
        self.resnet = timm.create_model(
            "resnet18", pretrained=False, num_classes=0, in_chans=in_chans
        )
        self.resnet_out_dim = self.resnet.num_features

        self.classifier = nn.Linear(128 + self.resnet_out_dim, num_classes)

    def forward(self, raw_1d: torch.Tensor, img_2d: torch.Tensor) -> torch.Tensor:
        byte_feat = self.byte_act(self.byte_fc(raw_1d))
        img_feat = self.resnet(img_2d)
        return self.classifier(torch.cat([byte_feat, img_feat], dim=1))


class ByteFormerWrapper(nn.Module):
    """Wraps a ByteFormer or ByteResNet so it accepts raw bytes directly.

    Internally applies Byte2Image on the input, so the user-facing API is
    just ``forward(x)`` with ``x`` shaped ``[B, block_size]`` (long).
    """

    def __init__(self, model, block_size: int, ngram: int = NGRAM):
        super().__init__()
        self.model = model
        self.block_size = block_size
        self.ngram = ngram

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        raw_1d = x.float()
        if self.block_size <= 512:
            img_2d = byte2image_torch(x, self.ngram)
        else:
            img_2d = byte2image_4k_torch(x, self.ngram)
        return self.model(raw_1d, img_2d)


def create_byteformer_model(
    num_classes: int,
    block_size: int = 512,
    model_type: str = "byteformer",
) -> ByteFormerWrapper:
    """Factory: returns a wrapper that takes raw bytes.

    Args:
        num_classes: Output classes.
        block_size: 512 or 4096.
        model_type: ``'byteformer'`` (PoolFormer image branch) or
            ``'byteresnet'`` (ResNet-18 image branch).
    """
    if model_type == "byteformer":
        model = ByteFormer(num_classes=num_classes, block_size=block_size)
    elif model_type == "byteresnet":
        model = ByteResNet(num_classes=num_classes, block_size=block_size)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")
    return ByteFormerWrapper(model, block_size)
