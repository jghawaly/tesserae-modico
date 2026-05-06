"""
Dataset loaders for the Tesserae file-fragment dataset.

The on-disk format is two numpy arrays:

  * ``block.npy``       shape ``(N, 512)``, ``uint8``  -- raw 512 B blocks
  * ``filetype_id.npy`` shape ``(N,)``,    ``int64``   -- class label per block

Larger fragments (4 KB, 8 KB, 16 KB) are stored as group files that hold,
for each fragment, the indices of the consecutive 512 B blocks that make
it up. So a 4 KB sample is ``np.concatenate(blocks[group_indices])``.

Splits live in a separate directory and are also numpy arrays:
``train_indices.npy``, ``val_indices.npy``, ``test_indices.npy`` for 512 B,
``{train,val,test}_4k_groups.npy`` for 4 KB, and so on for 8 KB / 16 KB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[Path, str]


class _BaseFragmentDataset(Dataset):
    """Common loading + label-remapping logic, shared by all sizes."""

    def __init__(
        self,
        npy_dir: PathLike,
        max_len: Optional[int] = None,
        class_weights_path: Optional[PathLike] = None,
        label_map_path: Optional[PathLike] = None,
        in_memory: bool = False,
        shared_blocks: Optional[np.ndarray] = None,
        shared_filetype_ids: Optional[np.ndarray] = None,
    ):
        self.npy_dir = Path(npy_dir)
        self.max_len = max_len
        self.in_memory = in_memory

        # Datasets that share the same underlying arrays save a lot of RAM:
        # we can load block.npy and filetype_id.npy once and pass them to the
        # train/val/test datasets instead of mmap'ing them three times.
        if shared_blocks is not None and shared_filetype_ids is not None:
            self.blocks = shared_blocks
            self.filetype_ids = shared_filetype_ids
        else:
            mmap_mode = None if in_memory else "r"
            self.blocks = np.load(self.npy_dir / "block.npy", mmap_mode=mmap_mode)
            self.filetype_ids = np.load(
                self.npy_dir / "filetype_id.npy", mmap_mode=mmap_mode
            )

        self.class_weights = (
            np.load(class_weights_path) if class_weights_path is not None else None
        )

        # Optional remapping from raw class IDs to a contiguous range. Useful
        # when a subset of classes was filtered out before training.
        self.label_map: Optional[dict] = None
        self._num_classes: Optional[int] = None
        if label_map_path is not None:
            label_map_path = Path(label_map_path)
            if label_map_path.exists():
                self.label_map = np.load(label_map_path, allow_pickle=True).item()
                self._num_classes = len(self.label_map)

    def _remap_label(self, label: int) -> int:
        if self.label_map is None:
            return label
        remapped = self.label_map.get(label)
        if remapped is None:
            raise ValueError(
                f"Label {label} not found in label_map; data and split are inconsistent."
            )
        return remapped

    def _vectorized_remap(self, raw_labels: np.ndarray) -> np.ndarray:
        if self.label_map is None:
            return raw_labels
        max_old = max(self.label_map)
        lookup = np.full(max_old + 1, -1, dtype=np.int64)
        for old_id, new_id in self.label_map.items():
            lookup[old_id] = new_id
        remapped = lookup[raw_labels]
        if (remapped == -1).any():
            n = (remapped == -1).sum()
            raise ValueError(f"{n} labels not found in label_map.")
        return remapped

    def _pad_or_trim(self, block_data: np.ndarray, orig_len: int) -> Tuple[torch.Tensor, int]:
        if self.max_len is None or self.max_len == orig_len:
            return torch.from_numpy(block_data).to(torch.uint8), orig_len
        if orig_len < self.max_len:
            out = torch.zeros(self.max_len, dtype=torch.uint8)
            out[:orig_len] = torch.from_numpy(block_data)
            return out, orig_len
        return torch.from_numpy(block_data[: self.max_len]).to(torch.uint8), self.max_len

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        return int(self.filetype_ids.max()) + 1


class TesseraeBlocks512(_BaseFragmentDataset):
    """Tesserae dataset of individual 512 B blocks.

    Returns triples ``(seq, label, orig_len)`` where ``seq`` is a uint8 tensor
    of shape ``(512,)`` (or ``(max_len,)`` if padded).
    """

    def __init__(
        self,
        indices_path: PathLike,
        npy_dir: PathLike,
        max_len: Optional[int] = None,
        class_weights_path: Optional[PathLike] = None,
        label_map_path: Optional[PathLike] = None,
        in_memory: bool = False,
        shared_blocks: Optional[np.ndarray] = None,
        shared_filetype_ids: Optional[np.ndarray] = None,
    ):
        super().__init__(
            npy_dir=npy_dir,
            max_len=max_len,
            class_weights_path=class_weights_path,
            label_map_path=label_map_path,
            in_memory=in_memory,
            shared_blocks=shared_blocks,
            shared_filetype_ids=shared_filetype_ids,
        )
        self.indices = np.load(indices_path)
        self.orig_len = int(self.blocks.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        real_idx = int(self.indices[idx])
        block_data = np.array(self.blocks[real_idx])
        label = self._remap_label(int(self.filetype_ids[real_idx]))
        seq, orig_len = self._pad_or_trim(block_data, self.orig_len)
        return seq, label, orig_len

    @property
    def labels(self) -> np.ndarray:
        return self._vectorized_remap(self.filetype_ids[self.indices])


class TesseraeBlocksGrouped(_BaseFragmentDataset):
    """Tesserae dataset of grouped blocks (4 KB / 8 KB / 16 KB).

    Each item is the concatenation of ``num_consecutive`` adjacent 512 B
    blocks. The label is read from the first block of the group; all blocks
    in a group share a class by construction.

    Returns triples ``(seq, label, orig_len)`` where ``seq`` is a uint8 tensor
    of shape ``(num_consecutive * 512,)``.
    """

    def __init__(
        self,
        groups_path: PathLike,
        num_consecutive: int,
        npy_dir: PathLike,
        max_len: Optional[int] = None,
        class_weights_path: Optional[PathLike] = None,
        label_map_path: Optional[PathLike] = None,
        in_memory: bool = False,
        shared_blocks: Optional[np.ndarray] = None,
        shared_filetype_ids: Optional[np.ndarray] = None,
    ):
        super().__init__(
            npy_dir=npy_dir,
            max_len=max_len,
            class_weights_path=class_weights_path,
            label_map_path=label_map_path,
            in_memory=in_memory,
            shared_blocks=shared_blocks,
            shared_filetype_ids=shared_filetype_ids,
        )
        self.groups = np.load(groups_path)
        self.num_consecutive = num_consecutive

        if self.groups.ndim == 2 and self.groups.shape[1] != num_consecutive:
            raise ValueError(
                f"Expected groups with {num_consecutive} indices per row, "
                f"got shape {self.groups.shape}"
            )

        self.base_block_size = int(self.blocks.shape[1])
        self.orig_len = self.base_block_size * num_consecutive

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        block_indices = self.groups[idx]
        chunks = [np.array(self.blocks[int(i)]) for i in block_indices]
        block_data = np.concatenate(chunks)
        label = self._remap_label(int(self.filetype_ids[int(block_indices[0])]))
        seq, orig_len = self._pad_or_trim(block_data, self.orig_len)
        return seq, label, orig_len

    @property
    def labels(self) -> np.ndarray:
        first_indices = self.groups[:, 0].astype(np.int64)
        return self._vectorized_remap(self.filetype_ids[first_indices])


def TesseraeBlocks4k(*args, **kwargs) -> TesseraeBlocksGrouped:
    """4 KB fragments (8 consecutive 512 B blocks)."""
    return TesseraeBlocksGrouped(*args, num_consecutive=8, **kwargs)


def TesseraeBlocks8k(*args, **kwargs) -> TesseraeBlocksGrouped:
    """8 KB fragments (16 consecutive 512 B blocks)."""
    return TesseraeBlocksGrouped(*args, num_consecutive=16, **kwargs)


def TesseraeBlocks16k(*args, **kwargs) -> TesseraeBlocksGrouped:
    """16 KB fragments (32 consecutive 512 B blocks)."""
    return TesseraeBlocksGrouped(*args, num_consecutive=32, **kwargs)


_BLOCK_SIZE_CONFIG = {
    512: {
        "uses_groups": False,
        "files": ("train_indices.npy", "val_indices.npy", "test_indices.npy"),
        "num_consecutive": 1,
    },
    4096: {
        "uses_groups": True,
        "files": ("train_4k_groups.npy", "val_4k_groups.npy", "test_4k_groups.npy"),
        "num_consecutive": 8,
    },
    8192: {
        "uses_groups": True,
        "files": ("train_8k_groups.npy", "val_8k_groups.npy", "test_8k_groups.npy"),
        "num_consecutive": 16,
    },
    16384: {
        "uses_groups": True,
        "files": ("train_16k_groups.npy", "val_16k_groups.npy", "test_16k_groups.npy"),
        "num_consecutive": 32,
    },
}


def load_tesserae_datasets(
    splits_dir: PathLike,
    npy_dir: PathLike,
    block_size: int = 512,
    max_len: Optional[int] = None,
    load_class_weights: bool = True,
    in_memory: bool = False,
    shared_blocks: Optional[np.ndarray] = None,
    shared_filetype_ids: Optional[np.ndarray] = None,
    skip_train: bool = False,
) -> Tuple[Optional[Dataset], Dataset, Dataset, Optional[np.ndarray]]:
    """Convenience helper that loads train / val / test for one block size.

    Returns ``(train_ds, val_ds, test_ds, class_weights)``. ``train_ds`` is
    ``None`` when ``skip_train`` is set; ``class_weights`` is ``None`` if
    no ``class_weights.npy`` file is present in ``splits_dir``.
    """
    if block_size not in _BLOCK_SIZE_CONFIG:
        raise ValueError(
            f"Unsupported block_size {block_size}; choose from {list(_BLOCK_SIZE_CONFIG)}"
        )

    splits_dir = Path(splits_dir)
    npy_dir = Path(npy_dir)
    cfg = _BLOCK_SIZE_CONFIG[block_size]
    train_file, val_file, test_file = cfg["files"]

    # Load shared arrays once if the caller asked for in-memory mode but
    # didn't pre-load them. This avoids loading them three times.
    if in_memory and shared_blocks is None:
        shared_blocks = np.load(npy_dir / "block.npy")
        shared_filetype_ids = np.load(npy_dir / "filetype_id.npy")

    class_weights = None
    class_weights_path = None
    if load_class_weights:
        cw_path = splits_dir / "class_weights.npy"
        if cw_path.exists():
            class_weights = np.load(cw_path)
            class_weights_path = cw_path

    label_map_path = splits_dir / "old_to_new_class.npy"
    if not label_map_path.exists():
        label_map_path = None

    common = dict(
        npy_dir=npy_dir,
        max_len=max_len,
        label_map_path=label_map_path,
        in_memory=in_memory,
        shared_blocks=shared_blocks,
        shared_filetype_ids=shared_filetype_ids,
    )

    if not cfg["uses_groups"]:
        train_ds = (
            None
            if skip_train
            else TesseraeBlocks512(
                indices_path=splits_dir / train_file,
                class_weights_path=class_weights_path,
                **common,
            )
        )
        val_ds = TesseraeBlocks512(indices_path=splits_dir / val_file, **common)
        test_ds = TesseraeBlocks512(indices_path=splits_dir / test_file, **common)
    else:
        n = cfg["num_consecutive"]
        train_ds = (
            None
            if skip_train
            else TesseraeBlocksGrouped(
                groups_path=splits_dir / train_file,
                num_consecutive=n,
                class_weights_path=class_weights_path,
                **common,
            )
        )
        val_ds = TesseraeBlocksGrouped(
            groups_path=splits_dir / val_file, num_consecutive=n, **common
        )
        test_ds = TesseraeBlocksGrouped(
            groups_path=splits_dir / test_file, num_consecutive=n, **common
        )

    return train_ds, val_ds, test_ds, class_weights


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Optional per-class weight tensor.
        gamma: Focusing parameter; higher means more emphasis on hard examples.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        probs = torch.softmax(logits, dim=-1)
        pt = probs[torch.arange(len(targets), device=targets.device), targets]
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
