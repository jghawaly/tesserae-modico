"""
MoDiCo: the full multi-branch file-fragment classifier.

Three encoders look at the same byte sequence from different angles
(motifs, distribution, context), an attentive-fusion module combines them,
and a small head produces class logits. A scalar size embedding is appended
so the head knows whether it's looking at a 512 B or a 4 KB fragment.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .context import ContextEncoder
from .distribution import DistributionEncoder
from .fusion import AttentiveFusion
from .motif import MotifEncoder


class MoDiCoClassifier(nn.Module):
    """Multi-branch byte-fragment classifier.

    Args:
        num_classes: Number of output classes.
        d_model: Shared feature width across all encoders and the fusion.
        max_len: Maximum supported fragment length (drives positional codes).
        hypothesis_dropout: Dropout in the fusion module.
        classifier_dropout: Dropout in the classification head.
        local_window_size: Window size for the motif encoder.
        local_window_stride: Stride between motif windows.
        entropy_window_size: Window size for the distribution encoder.
        entropy_cdf_points: Number of CDF samples in the entropy summary.
        seq_embed_dim: Internal channel width of the context encoder's
            byte-level stage.
        seq_num_layers: Number of hierarchical context-encoder stages.

    Forward shapes:
        Input  ``x``: ``[B, L]`` byte tensor; ``L`` may vary across batches.
        Output ``logits``: ``[B, num_classes]``.
        With ``return_aux=True`` also returns a list of three per-encoder
        auxiliary logits, used during training.
    """

    def __init__(
        self,
        num_classes: int = 619,
        d_model: int = 512,
        max_len: int = 16384,
        hypothesis_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        local_window_size: int = 512,
        local_window_stride: int = 256,
        entropy_window_size: int = 64,
        entropy_cdf_points: int = 64,
        seq_embed_dim: int = 128,
        seq_num_layers: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_len = max_len

        # Tiny MLP that turns a length-ratio scalar into a feature vector.
        self.size_embedder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, d_model),
        )

        self.motif_encoder = MotifEncoder(
            d_model=d_model,
            window_size=local_window_size,
            window_stride=local_window_stride,
        )
        self.distribution_encoder = DistributionEncoder(
            d_model=d_model,
            entropy_window_size=entropy_window_size,
            entropy_cdf_points=entropy_cdf_points,
        )
        self.context_encoder = ContextEncoder(
            d_model=d_model,
            max_len=max_len,
            embed_dim=seq_embed_dim,
            num_layers=seq_num_layers,
        )

        self.encoders = nn.ModuleList(
            [self.motif_encoder, self.distribution_encoder, self.context_encoder]
        )

        self.fusion = AttentiveFusion(
            num_branches=3,
            d_model=d_model,
            dropout=hypothesis_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, num_classes),
        )

        # Per-encoder auxiliary classifiers used during training. We always
        # build them so checkpoints have predictable keys; the main forward
        # path simply ignores them when return_aux is False.
        self.aux_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B, L = x.shape

        # Length-ratio scalar so the head sees fragment size explicitly.
        size_ratio = torch.tensor([L / self.max_len], device=x.device, dtype=torch.float)
        size_embedding = self.size_embedder(size_ratio.unsqueeze(-1)).expand(B, -1)

        motif_features = self.motif_encoder(x, pad_mask)
        distribution_features = self.distribution_encoder(x, pad_mask)
        context_features = self.context_encoder(x, pad_mask)

        branches = [motif_features, distribution_features, context_features]
        fused = self.fusion(branches)

        combined = torch.cat([fused, size_embedding], dim=-1)
        logits = self.classifier(combined)

        if return_aux:
            aux_logits = [head(feat) for head, feat in zip(self.aux_classifiers, branches)]
            return logits, aux_logits
        return logits


def collate_fragments(batch):
    """Collate function for variable-length fragment batches.

    Accepts both ``(seq, label, orig_len)`` triples and ``(seq, orig_len)``
    pairs (test-time without labels). Returns padded ``[B, L]`` byte tensors,
    a label tensor (when present), and a ``[B, L]`` boolean pad mask where
    True marks padding.

    There's a fast path: when every sequence in the batch is the same length
    (the common case for npy datasets) we skip per-sample padding entirely.
    """
    if len(batch[0]) == 3:
        seqs, labels, sizes = zip(*batch)
    else:
        seqs, sizes = zip(*batch)
        labels = None

    seq_lens = [len(seq) for seq in seqs]
    max_len = max(seq_lens)
    min_len = min(seq_lens)

    if min_len == max_len:
        padded_batch = torch.stack(seqs).long()
        if labels is not None:
            if isinstance(labels[0], torch.Tensor):
                labels_tensor = torch.stack(
                    [l if l.dim() > 0 else l.unsqueeze(0) for l in labels]
                ).squeeze().long()
            else:
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            return padded_batch, labels_tensor, None
        return padded_batch, None

    padded_seqs = []
    pad_masks = []
    for seq in seqs:
        seq_len = len(seq)
        if seq_len < max_len:
            padded = torch.cat([seq, torch.zeros(max_len - seq_len, dtype=seq.dtype)])
            mask = torch.cat(
                [
                    torch.zeros(seq_len, dtype=torch.bool),
                    torch.ones(max_len - seq_len, dtype=torch.bool),
                ]
            )
        else:
            padded = seq
            mask = torch.zeros(max_len, dtype=torch.bool)
        padded_seqs.append(padded)
        pad_masks.append(mask)

    seqs_tensor = torch.stack(padded_seqs)
    pad_masks_tensor = torch.stack(pad_masks)

    if labels is not None:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return seqs_tensor, labels_tensor, pad_masks_tensor
    return seqs_tensor, pad_masks_tensor
