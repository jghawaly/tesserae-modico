#!/usr/bin/env python3
"""
Evaluate MoDiCo or any of the baselines on a Tesserae split.

Computes top-1 / top-5 / balanced top-1 / balanced top-5 accuracy and
saves per-sample predictions to ``predictions.npz`` so you can rerun the
metrics or build confusion matrices later without rerunning the model.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
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
from baselines.sift import predict_sift_model, extract_byte_counts, transform_with_tfidf
from data import load_tesserae_datasets
from modico import MoDiCoClassifier, collate_fragments


NEURAL_MODELS = {"modico", "bytercnn", "fifty", "cnn_lstm", "dscse", "byteformer", "byteresnet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the Tesserae test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        choices=sorted(NEURAL_MODELS | {"sift"}))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained checkpoint (.pt for neural, .pkl for SIFT).")
    parser.add_argument("--npy_dir", type=Path, required=True)
    parser.add_argument("--splits_dir", type=Path, required=True)
    parser.add_argument("--block_size", type=int, default=512, choices=[512, 4096, 8192, 16384])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=619)
    parser.add_argument("--output_dir", type=Path, default=Path("./eval_results"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")

    # Architecture knobs that need to match the trained checkpoint.
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--local_window_stride", type=int, default=None)
    parser.add_argument("--entropy_window_size", type=int, default=64)
    parser.add_argument("--entropy_cdf_points", type=int, default=64)
    parser.add_argument("--seq_embed_dim", type=int, default=128)
    parser.add_argument("--seq_num_layers", type=int, default=2)

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "modico":
        local_window_stride = args.local_window_stride or (args.local_window_size // 2)
        return MoDiCoClassifier(
            num_classes=args.num_classes,
            d_model=args.d_model,
            max_len=args.max_len,
            local_window_size=args.local_window_size,
            local_window_stride=local_window_stride,
            entropy_window_size=args.entropy_window_size,
            entropy_cdf_points=args.entropy_cdf_points,
            seq_embed_dim=args.seq_embed_dim,
            seq_num_layers=args.seq_num_layers,
        )
    if args.model == "bytercnn":
        return create_bytercnn_model(num_classes=args.num_classes, block_size=args.block_size)
    if args.model == "fifty":
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
    raise ValueError(f"Unknown neural model {args.model!r}")


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # strict=False: ByteFormer wrappers wrap an inner module, but the keys
    # land in the same place after the prefix-strip above; a permissive load
    # is robust to small nominal mismatches between training and eval.
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    needs_pad_mask: bool,
) -> Dict[str, np.ndarray]:
    """Run a model over a dataloader; return top-5 preds + true labels."""
    all_preds = []
    all_labels = []
    all_top5 = []

    for batch in tqdm(dataloader, desc="Eval"):
        seqs, labels, pad_mask = batch
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True) if pad_mask is not None else None

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = (
                model(seqs, pad_mask=pad_mask)
                if needs_pad_mask
                else model(seqs)
            )
            if isinstance(logits, tuple):
                logits = logits[0]

        preds = logits.argmax(dim=-1)
        top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_top5.append(top5.cpu().numpy())

    return {
        "predictions": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "top5_predictions": np.concatenate(all_top5),
    }


def evaluate_sift(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """SIFT inference path. SIFT is not torch, so it has its own loop."""
    with open(args.checkpoint, "rb") as f:
        bundle = pickle.load(f)

    train_ds, val_ds, test_ds, _ = load_tesserae_datasets(
        splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=args.block_size,
        skip_train=(args.split != "train"),
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    n = len(ds)
    sample0 = ds[0]
    block_size = sample0[0].shape[0]
    blocks = np.empty((n, block_size), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    for i in tqdm(range(n), desc="Loading SIFT eval set"):
        seq, lab, _ = ds[i]
        blocks[i] = seq.numpy()
        labels[i] = lab

    counts = extract_byte_counts(blocks)
    features = transform_with_tfidf(counts, bundle["tfidf"])

    classes = bundle["classifier"].classes_
    probs = bundle["classifier"].predict_proba(features)
    top5_idx = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    top5 = classes[top5_idx]

    return {
        "predictions": top5[:, 0],
        "labels": labels,
        "top5_predictions": top5,
    }


def compute_metrics(preds: np.ndarray, labels: np.ndarray, top5: np.ndarray) -> Dict[str, float]:
    """Return top-1 / top-5 / balanced versions of each."""
    top1 = (preds == labels).astype(np.float32)

    # Vectorized top-5: check if the true label appears in any of the 5 columns.
    labels_col = labels[:, None]
    top5_correct = np.any(top5.astype(np.int64) == labels_col, axis=1).astype(np.float32)

    unique_classes = np.unique(labels)
    max_class = int(unique_classes.max())
    class_counts = np.bincount(labels, minlength=max_class + 1)
    valid = class_counts > 0

    top1_per_class = np.bincount(labels, weights=top1, minlength=max_class + 1)[valid] / class_counts[valid]
    top5_per_class = np.bincount(labels, weights=top5_correct, minlength=max_class + 1)[valid] / class_counts[valid]

    return {
        "top1_accuracy": float(top1.mean()),
        "top5_accuracy": float(top5_correct.mean()),
        "balanced_top1_accuracy": float(top1_per_class.mean()),
        "balanced_top5_accuracy": float(top5_per_class.mean()),
        "num_samples": int(len(labels)),
        "num_classes_seen": int(valid.sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "sift":
        results = evaluate_sift(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds, val_ds, test_ds, _ = load_tesserae_datasets(
            splits_dir=args.splits_dir, npy_dir=args.npy_dir, block_size=args.block_size,
            skip_train=(args.split != "train"),
        )
        ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fragments,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        model = build_model(args)
        model = load_checkpoint(model, args.checkpoint, device)
        # MoDiCo expects a pad mask; baselines just take the sequence.
        results = run_inference(
            model=model,
            dataloader=dl,
            device=device,
            use_amp=not args.no_amp,
            needs_pad_mask=(args.model == "modico"),
        )

    metrics = compute_metrics(
        results["predictions"], results["labels"], results["top5_predictions"]
    )

    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split} (block_size={args.block_size})")
    print(f"Samples:    {metrics['num_samples']:,}  ({metrics['num_classes_seen']} classes seen)")
    print("-" * 60)
    print(f"  top-1 accuracy:           {metrics['top1_accuracy']:.4f}")
    print(f"  top-5 accuracy:           {metrics['top5_accuracy']:.4f}")
    print(f"  balanced top-1 accuracy:  {metrics['balanced_top1_accuracy']:.4f}")
    print(f"  balanced top-5 accuracy:  {metrics['balanced_top5_accuracy']:.4f}")
    print("=" * 60)

    np.savez(
        args.output_dir / "predictions.npz",
        predictions=results["predictions"],
        labels=results["labels"],
        top5_predictions=results["top5_predictions"],
    )
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved predictions to {args.output_dir / 'predictions.npz'}")
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
