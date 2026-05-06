"""
Samplers for long-tailed file-fragment recognition.

Based on Kang et al. (ICLR 2020). The class-balanced sampler is what makes
Stage 2 of MoDiCo work: it draws an equal number of instances per class so
the classifier head sees uniform supervision across the long tail.
"""

from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """Equal-per-class instance sampling.

    For each class, draw ``samples_per_class`` indices (with replacement if
    the class has fewer instances). Useful for the cRT step in decoupled
    training of long-tailed classifiers.

    Args:
        labels: Per-sample class labels for the underlying dataset.
        samples_per_class: How many instances to draw per class per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        labels: np.ndarray,
        samples_per_class: int = 100,
        seed: Optional[int] = None,
    ):
        self.labels = np.asarray(labels)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build the per-class index lists in one pass: argsort once, then
        # slice. This is much faster than a Python loop on tens of millions
        # of samples.
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        self.classes = unique_classes.tolist()
        self.num_classes = len(self.classes)

        sorted_indices = np.argsort(self.labels)
        boundaries = np.concatenate([[0], np.cumsum(counts)])

        self.class_indices = {
            cls: sorted_indices[boundaries[i] : boundaries[i + 1]]
            for i, cls in enumerate(self.classes)
        }
        self.class_sizes = dict(zip(self.classes, counts.tolist()))
        self.min_class_size = int(counts.min())
        self.max_class_size = int(counts.max())

    def __iter__(self) -> Iterator[int]:
        indices = []
        for cls in self.classes:
            cls_indices = self.class_indices[cls]
            replace = len(cls_indices) < self.samples_per_class
            sampled = self.rng.choice(
                cls_indices, size=self.samples_per_class, replace=replace
            )
            indices.extend(sampled.tolist())
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_classes * self.samples_per_class

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(
            self.seed + epoch if self.seed is not None else epoch
        )


class SquareRootSampler(Sampler):
    """Square-root balanced sampling: ``p_j proportional to sqrt(n_j)``.

    A softer version of class-balanced sampling that keeps some preference
    for frequent classes while still upweighting the tail.
    """

    def __init__(
        self,
        labels: np.ndarray,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.labels = np.asarray(labels)
        self.num_samples = num_samples or len(labels)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        unique_classes, counts = np.unique(self.labels, return_counts=True)
        sqrt_counts = np.sqrt(counts.astype(np.float64))
        class_probs = sqrt_counts / sqrt_counts.sum()

        sample_weights = np.zeros(len(labels), dtype=np.float64)
        for cls, prob in zip(unique_classes, class_probs):
            mask = self.labels == cls
            sample_weights[mask] = prob / mask.sum()
        self.sample_weights = sample_weights / sample_weights.sum()

    def __iter__(self) -> Iterator[int]:
        indices = self.rng.choice(
            len(self.labels), size=self.num_samples, replace=True, p=self.sample_weights
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(
            self.seed + epoch if self.seed is not None else epoch
        )


class ProgressivelyBalancedSampler(Sampler):
    """Interpolates from instance-balanced (epoch 0) to class-balanced.

    ``p_j(t) = (1 - t/T) * p_j_IB + (t/T) * p_j_CB``, where ``t`` is the
    current epoch and ``T`` is the total epoch count. Provides a smooth
    schedule from natural-frequency to uniform sampling over training.
    """

    def __init__(
        self,
        labels: np.ndarray,
        total_epochs: int,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.labels = np.asarray(labels)
        self.total_epochs = total_epochs
        self.num_samples = num_samples or len(labels)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0

        unique_classes, counts = np.unique(self.labels, return_counts=True)
        self.classes = unique_classes
        self.counts = counts
        self.num_classes = len(unique_classes)

        self.p_ib = counts.astype(np.float64) / counts.sum()
        self.p_cb = np.ones(self.num_classes, dtype=np.float64) / self.num_classes
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def _get_current_probs(self) -> np.ndarray:
        t = self.current_epoch / max(self.total_epochs - 1, 1)
        return (1 - t) * self.p_ib + t * self.p_cb

    def __iter__(self) -> Iterator[int]:
        class_probs = self._get_current_probs()
        sample_weights = np.zeros(len(self.labels), dtype=np.float64)
        for i, cls in enumerate(self.classes):
            indices = self.class_indices[cls]
            sample_weights[indices] = class_probs[i] / len(indices)
        sample_weights /= sample_weights.sum()

        indices = self.rng.choice(
            len(self.labels), size=self.num_samples, replace=True, p=sample_weights
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.rng = np.random.default_rng(
            self.seed + epoch if self.seed is not None else epoch
        )


def get_class_balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Per-sample weights for ``WeightedRandomSampler`` that equalize classes.

    Each class ends up with the same total weight, so larger classes get
    proportionally smaller per-sample weights.
    """
    labels = np.asarray(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts.astype(np.float64)

    sample_weights = np.zeros(len(labels), dtype=np.float64)
    for cls, weight in zip(unique_classes, class_weights):
        sample_weights[labels == cls] = weight
    return sample_weights
