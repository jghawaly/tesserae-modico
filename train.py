#!/usr/bin/env python3
"""
Train MoDiCo with the decoupled two-stage recipe of Kang et al. (ICLR 2020).

Stage 1 learns representations under instance-balanced sampling. Stage 2
freezes the encoders and re-trains the classifier head with class-balanced
sampling, which is what gets us most of the way out of the long tail.

Stages can be run sequentially (the default) or independently via ``--stage``.
A single run trains both the 512 B and 4 KB views by interleaving batches
from each at every step; the model receives the same encoder for both sizes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Set spawn before importing anything that touches CUDA, so DataLoader
# workers don't inherit a forked CUDA state.
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from data import (
    ClassBalancedSampler,
    FocalLoss,
    load_tesserae_datasets,
)
from modico import MoDiCoClassifier, collate_fragments

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MoDiCo with decoupled two-stage learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--npy_dir", type=Path, required=True,
                        help="Directory with block.npy and filetype_id.npy.")
    parser.add_argument("--splits_dir", type=Path, required=True,
                        help="Directory with train/val/test split files.")
    parser.add_argument("--block_size", type=int, default=512, choices=[512, 4096],
                        help="Primary block size for validation. Training uses both 512 and 4 KB jointly.")
    parser.add_argument("--joint_train", action="store_true", default=True,
                        help="Interleave 512 B and 4 KB samples during training.")
    parser.add_argument("--single_size_train", dest="joint_train", action="store_false",
                        help="Train on a single block size only (the value of --block_size).")

    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"],
                        help="Run stage 1, stage 2, or both sequentially.")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume", type=Path, default=None,
                        help="Optional checkpoint to load before training (e.g. stage1_best.pt).")

    parser.add_argument("--num_classes", type=int, default=619)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--hypothesis_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)

    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--local_window_stride", type=int, default=None,
                        help="Default: local_window_size // 2.")
    parser.add_argument("--entropy_window_size", type=int, default=64)
    parser.add_argument("--entropy_cdf_points", type=int, default=64)
    parser.add_argument("--seq_embed_dim", type=int, default=128)
    parser.add_argument("--seq_num_layers", type=int, default=2)

    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--lr", "--stage1_lr", type=float, default=1e-4, dest="lr")
    parser.add_argument("--stage2_lr", type=float, default=None,
                        help="Default: same as --lr.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--samples_per_class", type=int, default=100,
                        help="Per-class sample count for stage 2's class-balanced sampler.")
    parser.add_argument("--aux_loss_weight", type=float, default=0.1,
                        help="Weight on per-encoder auxiliary losses during stage 1.")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use focal loss instead of cross-entropy in stage 1.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training set size for quick experiments.")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Cap validation set size.")

    parser.add_argument("--wandb_project", type=str, default=None,
                        help="If set, log to this Weights & Biases project.")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize torch.distributed if launched via torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def print_main(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(msg, flush=True)


def build_model(args: argparse.Namespace) -> MoDiCoClassifier:
    local_window_stride = args.local_window_stride or (args.local_window_size // 2)
    return MoDiCoClassifier(
        num_classes=args.num_classes,
        d_model=args.d_model,
        max_len=args.max_len,
        hypothesis_dropout=args.hypothesis_dropout,
        classifier_dropout=args.classifier_dropout,
        local_window_size=args.local_window_size,
        local_window_stride=local_window_stride,
        entropy_window_size=args.entropy_window_size,
        entropy_cdf_points=args.entropy_cdf_points,
        seq_embed_dim=args.seq_embed_dim,
        seq_num_layers=args.seq_num_layers,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    world_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        collate_fn=collate_fragments,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def maybe_subset(
    dataset, max_samples: Optional[int], seed: int, label: str, rank: int
):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=max_samples, replace=False)
    print_main(f"[Data] Subsampled {label} to {max_samples:,} of {len(dataset):,}", rank)
    return Subset(dataset, idx.tolist())


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloaders,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    rank: int = 0,
    desc: str = "Val",
) -> Tuple[float, float]:
    """Run validation across one or more dataloaders and return (loss, acc)."""
    model.eval()

    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]

    total_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0

    for dl in dataloaders:
        if dl is None:
            continue
        for batch in tqdm(dl, desc=desc, leave=False, disable=(rank != 0)):
            seqs, labels, pad_mask = batch
            seqs = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True) if pad_mask is not None else None

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(seqs, pad_mask=pad_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logits, labels)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    model.train()
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float("nan")
    return avg_loss, correct / max(total, 1)


def train_stage1(
    model: nn.Module,
    train_dls,
    val_dls,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    rank: int,
    world_size: int,
    use_amp: bool,
    aux_loss_weight: float,
    ckpt_dir: Path,
    use_wandb: bool,
) -> Tuple[nn.Module, float]:
    """Stage 1: representation learning with instance-balanced sampling."""
    print_main("=" * 60, rank)
    print_main("Stage 1: representation learning", rank)
    print_main("=" * 60, rank)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_acc = 0.0

    train_dls = [dl for dl in train_dls if dl is not None]

    for epoch in range(epochs):
        model.train()
        if isinstance(train_dls[0].sampler, DistributedSampler):
            for dl in train_dls:
                dl.sampler.set_epoch(epoch)

        # Interleave one batch from each loader. When loaders have different
        # lengths, the longer ones simply contribute extra batches at the end.
        iters = [iter(dl) for dl in train_dls]
        steps_per_dl = [len(dl) for dl in train_dls]
        total_steps = sum(steps_per_dl)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        pbar = tqdm(
            total=total_steps,
            desc=f"Stage1 Epoch {epoch+1}/{epochs}",
            disable=(rank != 0),
        )

        loader_idx = 0
        steps_taken = [0] * len(iters)
        while sum(steps_taken) < total_steps:
            # Round-robin across loaders, skipping any that are exhausted.
            for _ in range(len(iters)):
                if steps_taken[loader_idx] < steps_per_dl[loader_idx]:
                    break
                loader_idx = (loader_idx + 1) % len(iters)

            try:
                batch = next(iters[loader_idx])
            except StopIteration:
                steps_taken[loader_idx] = steps_per_dl[loader_idx]
                loader_idx = (loader_idx + 1) % len(iters)
                continue

            steps_taken[loader_idx] += 1
            loader_idx = (loader_idx + 1) % len(iters)

            seqs, labels, pad_mask = batch
            seqs = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True) if pad_mask is not None else None

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux_logits = model(seqs, pad_mask=pad_mask, return_aux=True)
                loss = criterion(logits, labels)
                for aux in aux_logits:
                    loss = loss + aux_loss_weight * criterion(aux, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                # Skip the bad batch and keep going. Repeated failures
                # usually point at an LR or scaler-scale issue.
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.size(0)
                epoch_loss += loss.item()

            pbar.update(1)
            if pbar.n % 50 == 0 and epoch_total > 0:
                pbar.set_postfix(
                    loss=f"{epoch_loss / pbar.n:.4f}",
                    acc=f"{epoch_correct / epoch_total:.4f}",
                )
        pbar.close()

        train_loss = epoch_loss / max(total_steps, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_loss, val_acc = validate(model, val_dls, criterion, device, use_amp, rank)

        print_main(
            f"[Stage 1] Epoch {epoch+1}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}",
            rank,
        )

        if rank == 0:
            if use_wandb:
                wandb.log({
                    "stage1/epoch": epoch + 1,
                    "stage1/train_loss": train_loss,
                    "stage1/train_acc": train_acc,
                    "stage1/val_loss": val_loss,
                    "stage1/val_acc": val_acc,
                })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state_dict": _strip_ddp(model).state_dict(), "val_acc": val_acc},
                    ckpt_dir / "stage1_best.pt",
                )
                print_main(f"  saved stage1_best.pt (val_acc={val_acc:.4f})", rank)

    return model, best_val_acc


def train_stage2(
    model: nn.Module,
    train_ds,
    val_dls,
    criterion,
    epochs: int,
    lr: float,
    samples_per_class: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    num_workers: int,
    use_amp: bool,
    ckpt_dir: Path,
    use_wandb: bool,
) -> Tuple[nn.Module, float]:
    """Stage 2: freeze encoders, retrain the classifier head class-balanced."""
    print_main("=" * 60, rank)
    print_main("Stage 2: class-balanced classifier retraining", rank)
    print_main("=" * 60, rank)

    inner = _strip_ddp(model)
    for name, module in inner.named_children():
        if name not in ("classifier", "aux_classifiers"):
            for p in module.parameters():
                p.requires_grad = False

    if isinstance(inner.classifier, nn.Sequential):
        for layer in inner.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    labels = train_ds.labels
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    sampler = ClassBalancedSampler(labels, samples_per_class=samples_per_class, seed=42)

    # Stage 2 keeps workers at 0: the underlying dataset is mmap-backed and
    # forking many workers blows up the kernel page cache.
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fragments,
        num_workers=0,
        pin_memory=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Stage2 Epoch {epoch+1}/{epochs}",
            disable=(rank != 0),
        )

        for batch in pbar:
            seqs, labels_batch, pad_mask = batch
            seqs = seqs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True) if pad_mask is not None else None

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(seqs, pad_mask=pad_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logits, labels_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == labels_batch).sum().item()
                epoch_total += labels_batch.size(0)
                epoch_loss += loss.item()

        train_loss = epoch_loss / len(loader)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_loss, val_acc = validate(model, val_dls, criterion, device, use_amp, rank)

        print_main(
            f"[Stage 2] Epoch {epoch+1}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}",
            rank,
        )

        if rank == 0:
            if use_wandb:
                wandb.log({
                    "stage2/epoch": epoch + 1,
                    "stage2/train_loss": train_loss,
                    "stage2/train_acc": train_acc,
                    "stage2/val_loss": val_loss,
                    "stage2/val_acc": val_acc,
                })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state_dict": _strip_ddp(model).state_dict(), "val_acc": val_acc},
                    ckpt_dir / "stage2_best.pt",
                )
                print_main(f"  saved stage2_best.pt (val_acc={val_acc:.4f})", rank)

    return model, best_val_acc


def _strip_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = HAS_WANDB and args.wandb_project is not None and rank == 0
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print_main("Loading datasets...", rank)
    _, val_ds_512, _, _ = load_tesserae_datasets(
        splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=512
    )
    train_ds_512, _, _, class_weights = load_tesserae_datasets(
        splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=512,
        load_class_weights=True,
    )

    train_ds_4k = val_ds_4k = None
    if args.joint_train or args.block_size == 4096:
        train_ds_4k, val_ds_4k, _, _ = load_tesserae_datasets(
            splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=4096,
        )

    train_ds_512 = maybe_subset(train_ds_512, args.max_train_samples, args.seed, "train 512", rank)
    val_ds_512 = maybe_subset(val_ds_512, args.max_val_samples, args.seed + 1, "val 512", rank)
    if train_ds_4k is not None:
        train_ds_4k = maybe_subset(train_ds_4k, args.max_train_samples, args.seed + 2, "train 4k", rank)
        val_ds_4k = maybe_subset(val_ds_4k, args.max_val_samples, args.seed + 3, "val 4k", rank)

    train_dls = []
    if args.joint_train or args.block_size == 512:
        train_dls.append(build_dataloader(train_ds_512, args.batch_size, world_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          drop_last=True))
    if (args.joint_train or args.block_size == 4096) and train_ds_4k is not None:
        train_dls.append(build_dataloader(train_ds_4k, args.batch_size, world_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          drop_last=True))

    val_dls = []
    if args.joint_train or args.block_size == 512:
        val_dls.append(build_dataloader(val_ds_512, args.val_batch_size, world_size,
                                        shuffle=False, num_workers=args.num_workers))
    if (args.joint_train or args.block_size == 4096) and val_ds_4k is not None:
        val_dls.append(build_dataloader(val_ds_4k, max(args.val_batch_size // 8, 1), world_size,
                                        shuffle=False, num_workers=args.num_workers))

    print_main("Building model...", rank)
    model = build_model(args).to(device)
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip DDP prefix if the checkpoint was saved under DDP and we
        # haven't wrapped yet.
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print_main(f"Loaded weights from {args.resume}", rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    n_params = sum(p.numel() for p in model.parameters())
    print_main(f"Parameters: {n_params:,}", rank)

    if args.use_focal_loss:
        weights = (
            torch.from_numpy(class_weights).float().to(device)
            if class_weights is not None
            else None
        )
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = sum(len(dl) for dl in train_dls)
    total_steps = max(1, args.stage1_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    stage1_val_acc = 0.0
    if args.stage in ("1", "both"):
        model, stage1_val_acc = train_stage1(
            model=model,
            train_dls=train_dls,
            val_dls=val_dls,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.stage1_epochs,
            device=device,
            rank=rank,
            world_size=world_size,
            use_amp=not args.no_amp,
            aux_loss_weight=args.aux_loss_weight,
            ckpt_dir=args.checkpoint_dir,
            use_wandb=use_wandb,
        )

    stage2_val_acc = stage1_val_acc
    if args.stage in ("2", "both"):
        stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
        # Stage 2 uses cross-entropy on a class-balanced sampler; focal loss
        # would double-count rebalancing.
        stage2_criterion = nn.CrossEntropyLoss()
        model, stage2_val_acc = train_stage2(
            model=model,
            train_ds=train_ds_512,
            val_dls=val_dls,
            criterion=stage2_criterion,
            epochs=args.stage2_epochs,
            lr=stage2_lr,
            samples_per_class=args.samples_per_class,
            batch_size=args.batch_size,
            device=device,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            ckpt_dir=args.checkpoint_dir,
            use_wandb=use_wandb,
        )

    if rank == 0:
        torch.save(
            {"model_state_dict": _strip_ddp(model).state_dict()},
            args.checkpoint_dir / "final_model.pt",
        )
        with open(args.checkpoint_dir / "results.json", "w") as f:
            json.dump(
                {
                    "stage1_val_acc": stage1_val_acc,
                    "stage2_val_acc": stage2_val_acc,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                f,
                indent=2,
            )

    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
