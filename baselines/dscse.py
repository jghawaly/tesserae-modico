"""
DSC-SE: depthwise separable convolutions with Squeeze-and-Excitation.

Faithful PyTorch port of Ghaleb et al., "File Fragment Classification using
Light-Weight Convolutional Neural Networks" (arXiv:2305.00656, 2023).

Architecture (Figure 2b in the paper):
    Embedding(256, 32)
    -> Conv1D(32, k=19, s=2)
    -> Inception(32 -> 64) + SE(reduction=1)
    -> Inception(64 -> 64) + SE(reduction=2)   (no spatial reduction)
    -> Inception(64 -> 128) + SE(reduction=8)
    -> AdaptiveAvgPool -> Conv1D(128 -> num_classes)

Each Inception block has four parallel branches (1x1 + DSC k=11/19/27); the
DSC branches are summed and pooled, then added back to the 1x1 / skip branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise then pointwise 1D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SqueezeExcitation(nn.Module):
    """Channel-wise SE block (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(2)


class InceptionBlock(nn.Module):
    """4-branch Inception with DSC mixers and a 1x1 / skip path.

    The three DSC branches (k=11/19/27) are summed, pooled by ``pool_stride``,
    and added to the (possibly downsampled / projected) skip branch.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_stride: int = 4):
        super().__init__()
        self.pool_stride = pool_stride
        self.change_channels = in_channels != out_channels

        if self.change_channels:
            self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.bn_pw = nn.BatchNorm1d(out_channels)
        if pool_stride > 1:
            self.skip_pool = nn.AvgPool1d(pool_stride, stride=pool_stride)

        self.dsc11 = DepthwiseSeparableConv1d(in_channels, out_channels, 11)
        self.bn11 = nn.BatchNorm1d(out_channels)
        self.dsc19 = DepthwiseSeparableConv1d(in_channels, out_channels, 19)
        self.bn19 = nn.BatchNorm1d(out_channels)
        self.dsc27 = DepthwiseSeparableConv1d(in_channels, out_channels, 27)
        self.bn27 = nn.BatchNorm1d(out_channels)

        if pool_stride > 1:
            self.maxpool = nn.MaxPool1d(pool_stride, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d11 = self.bn11(self.dsc11(x))
        d19 = self.bn19(self.dsc19(x))
        d27 = self.bn27(self.dsc27(x))
        dsc_sum = d11 + d19 + d27
        if self.pool_stride > 1:
            dsc_sum = self.maxpool(dsc_sum)

        skip = self.bn_pw(self.pointwise(x)) if self.change_channels else x
        if self.pool_stride > 1:
            skip = self.skip_pool(skip)

        return dsc_sum + skip


class DSCSE(nn.Module):
    """DSC-SE file fragment classifier (around 105K params).

    Args:
        num_classes: Output classes.
        block_size: Input fragment size (512 or 4096).

    Forward shape: ``[B, L]`` byte tensor -> ``[B, num_classes]`` logits.
    """

    def __init__(self, num_classes: int = 75, block_size: int = 4096):
        super().__init__()
        self.num_classes = num_classes
        self.block_size = block_size

        self.embedding = nn.Embedding(256, 32)
        self.conv1 = nn.Conv1d(32, 32, kernel_size=19, stride=2, padding=9, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        self.inc1 = InceptionBlock(32, 64, pool_stride=4)
        self.se1 = SqueezeExcitation(64, reduction=1)

        self.inc2 = InceptionBlock(64, 64, pool_stride=1)
        self.se2 = SqueezeExcitation(64, reduction=2)

        self.inc3 = InceptionBlock(64, 128, pool_stride=4)
        self.se3 = SqueezeExcitation(128, reduction=8)

        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        h = self.embedding(x).transpose(1, 2)
        h = F.hardswish(self.bn1(self.conv1(h)))
        h = F.hardswish(self.se1(self.inc1(h)))
        h = F.hardswish(self.se2(self.inc2(h)))
        h = F.hardswish(self.se3(self.inc3(h)))
        h = h.mean(dim=2, keepdim=True)
        return self.classifier(h).squeeze(2)


def create_dscse_model(num_classes: int = 75, block_size: int = 4096) -> DSCSE:
    """Factory for DSC-SE."""
    return DSCSE(num_classes=num_classes, block_size=block_size)
