#!/usr/bin/env python3
"""
Train a single baseline model on the Tesserae dataset.

Multiplexes over the available baselines via ``--model``:
    bytercnn, fifty, cnn_lstm, dscse, byteformer, byteresnet, sift

The neural baselines reuse the MoDiCo data loaders. SIFT is a non-neural
TF-IDF + Random Forest pipeline; for SIFT we ignore the torch-specific
arguments and pass through to ``baselines.sift.train_sift_model``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from baselines import (
    create_bytercnn_model,
    create_cnn_lstm_model,
    create_dscse_model,
    create_fifty_model,
)
from baselines.byteformer import create_byteformer_model
from baselines.sift import train_sift_model
from data import load_tesserae_datasets
from modico import collate_fragments


NEURAL_BASELINES = {"bytercnn", "fifty", "cnn_lstm", "dscse", "byteformer", "byteresnet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline model on the Tesserae dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        choices=["bytercnn", "fifty", "cnn_lstm", "dscse",
                                 "byteformer", "byteresnet", "sift"])
    parser.add_argument("--npy_dir", type=Path, required=True)
    parser.add_argument("--splits_dir", type=Path, required=True)
    parser.add_argument("--block_size", type=int, default=512, choices=[512, 4096, 8192, 16384])
    parser.add_argument("--num_classes", type=int, default=619)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints"))

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap on training samples (also caps SIFT training set).")

    # SIFT-specific knobs.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=30)
    parser.add_argument("--min_samples_leaf", type=int, default=100)
    parser.add_argument("--rf_max_samples", type=float, default=None)
    parser.add_argument("--rf_n_jobs", type=int, default=-1)

    return parser.parse_args()


def build_neural_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "bytercnn":
        return create_bytercnn_model(num_classes=args.num_classes, block_size=args.block_size)
    if args.model == "fifty":
        # FiFTy ships a Scenario-1 config for 512 and 4096 only.
        bs = args.block_size if args.block_size in (512, 4096) else 4096
        return create_fifty_model(num_classes=args.num_classes, block_size=bs)
    if args.model == "cnn_lstm":
        return create_cnn_lstm_model(num_classes=args.num_classes)
    if args.model == "dscse":
        return create_dscse_model(num_classes=args.num_classes, block_size=args.block_size)
    if args.model in ("byteformer", "byteresnet"):
        return create_byteformer_model(
            num_classes=args.num_classes,
            block_size=args.block_size,
            model_type=args.model,
        )
    raise ValueError(f"Unknown neural baseline {args.model!r}")


def train_neural(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _, _ = load_tesserae_datasets(
        splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=args.block_size,
    )

    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        from torch.utils.data import Subset
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_ds), size=args.max_train_samples, replace=False)
        train_ds = Subset(train_ds, idx.tolist())

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fragments,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fragments,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_neural_model(args).to(device)
    print(f"Model: {args.model}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    use_amp = not args.no_amp and torch.cuda.is_available()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        pbar = tqdm(train_dl, desc=f"{args.model} Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            seqs, labels, pad_mask = batch
            seqs = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(seqs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.size(0)
                epoch_loss += loss.item()

            pbar.set_postfix(loss=f"{epoch_loss / (pbar.n+1):.4f}",
                             acc=f"{epoch_correct / max(epoch_total, 1):.4f}")

        val_loss, val_acc = _validate_neural(model, val_dl, criterion, device, use_amp)
        print(f"Epoch {epoch+1}: train_acc={epoch_correct/max(epoch_total,1):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "val_acc": val_acc, "model_type": args.model},
                args.checkpoint_dir / f"{args.model}_best.pt",
            )
            print(f"  saved {args.model}_best.pt (val_acc={val_acc:.4f})")


@torch.no_grad()
def _validate_neural(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Val", leave=False):
        seqs, labels, _ = batch
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(seqs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def train_sift(args: argparse.Namespace) -> None:
    """Train SIFT (random forest on byte histograms) end-to-end.

    Loads training samples into RAM (just the bytes, not labels are huge),
    computes histograms, fits TF-IDF, trains the forest, pickles to disk.
    """
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _, _, _ = load_tesserae_datasets(
        splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=args.block_size,
    )

    n = len(train_ds)
    cap = args.max_train_samples if args.max_train_samples else n
    cap = min(cap, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=cap, replace=False) if cap < n else np.arange(n)

    print(f"Materializing {cap:,} fragments for SIFT...")
    sample0 = train_ds[int(idx[0])]
    block_size = sample0[0].shape[0]
    blocks = np.empty((cap, block_size), dtype=np.uint8)
    labels = np.empty(cap, dtype=np.int64)
    for i, j in enumerate(tqdm(idx, desc="loading")):
        seq, lab, _ = train_ds[int(j)]
        blocks[i] = seq.numpy()
        labels[i] = lab

    print(f"Training SIFT (RF n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
    bundle = train_sift_model(
        blocks=blocks,
        labels=labels,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=args.rf_max_samples,
        n_jobs=args.rf_n_jobs,
        seed=args.seed,
    )

    out = args.checkpoint_dir / f"sift_{args.block_size}.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved SIFT model to {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.model == "sift":
        train_sift(args)
    elif args.model in NEURAL_BASELINES:
        train_neural(args)
    else:
        raise ValueError(f"Unknown model {args.model!r}")


if __name__ == "__main__":
    main()
