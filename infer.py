#!/usr/bin/env python3
"""
Single-file inference with a trained MoDiCo checkpoint.

Reads bytes from disk (raw binary or a ``.npy`` of uint8) and prints the top-5
predicted file types. The input can be any length up to ``--max_len``; longer
inputs are truncated and shorter inputs are padded.

Example:
    python infer.py --checkpoint stage2_best.pt --input_file mystery.bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from modico import MoDiCoClassifier


def load_input_bytes(path: Path, max_len: int) -> torch.Tensor:
    """Load raw bytes from a file, returning a ``[1, L]`` long tensor."""
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.frombuffer(path.read_bytes(), dtype=np.uint8)

    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    if arr.size > max_len:
        arr = arr[:max_len]
    return torch.from_numpy(arr.copy()).long().unsqueeze(0)


def load_class_names(path: Path | None, num_classes: int) -> List[str]:
    """Return human-readable class names if available, else ``class_<idx>``."""
    if path is not None and path.exists():
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Dict mapping name->idx (or idx->name); normalize to a list.
        if isinstance(list(data.values())[0], int):
            return [name for name, _ in sorted(data.items(), key=lambda kv: kv[1])]
        return [data[str(i)] if str(i) in data else f"class_{i}" for i in range(num_classes)]
    return [f"class_{i}" for i in range(num_classes)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoDiCo on a single file.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained MoDiCo checkpoint (.pt).")
    parser.add_argument("--input_file", type=Path, required=True,
                        help="Path to a raw binary file or a uint8 .npy array.")
    parser.add_argument("--class_names", type=Path, default=None,
                        help="Optional JSON file mapping class indices to names.")
    parser.add_argument("--num_classes", type=int, default=619)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--local_window_stride", type=int, default=None)
    parser.add_argument("--entropy_window_size", type=int, default=64)
    parser.add_argument("--entropy_cdf_points", type=int, default=64)
    parser.add_argument("--seq_embed_dim", type=int, default=128)
    parser.add_argument("--seq_num_layers", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_window_stride = args.local_window_stride or (args.local_window_size // 2)
    model = MoDiCoClassifier(
        num_classes=args.num_classes,
        d_model=args.d_model,
        max_len=args.max_len,
        local_window_size=args.local_window_size,
        local_window_stride=local_window_stride,
        entropy_window_size=args.entropy_window_size,
        entropy_cdf_points=args.entropy_cdf_points,
        seq_embed_dim=args.seq_embed_dim,
        seq_num_layers=args.seq_num_layers,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    x = load_input_bytes(args.input_file, args.max_len).to(device)

    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=min(args.top_k, probs.numel()))

    class_names = load_class_names(args.class_names, args.num_classes)

    print(f"Input: {args.input_file} ({x.shape[1]} bytes after padding/truncation)")
    print(f"Top-{args.top_k} predictions:")
    for rank, (idx, p) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"  {rank}. {name:30s}  p={p:.4f}  (class {idx})")


if __name__ == "__main__":
    main()
