# Tesserae and MoDiCo

This repository accompanies the paper **"Tesserae and MoDiCo: A Billion-Fragment Dataset and Multi-Branch Architecture for File Fragment Classification"**. **Tesserae** is a public, billion-fragment file-fragment dataset spanning hundreds of file types and four block sizes (512 B, 4 KB, 8 KB, 16 KB). **MoDiCo** is a multi-branch architecture that classifies a raw byte fragment by combining three complementary views of it: a CNN over local byte motifs, a statistical encoder over the byte distribution and entropy CDF, and a hierarchical byte-level transformer for sequential context. The three branches are combined by an attentive fusion module and trained with the decoupled two-stage recipe of Kang et al. (ICLR 2020) to handle Tesserae's long-tailed class distribution.

## Installation

```
pip install -r requirements.txt
```

The code is tested with Python 3.10 or newer.

## Quick inference example

```python
import numpy as np
import torch
from modico import MoDiCoClassifier

model = MoDiCoClassifier(num_classes=619, max_len=4096)
state = torch.load("stage2_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model_state_dict"], strict=False)
model.eval()

raw_bytes = np.frombuffer(open("mystery.bin", "rb").read(), dtype=np.uint8)
x = torch.from_numpy(raw_bytes[:4096].copy()).long().unsqueeze(0)

with torch.no_grad():
    logits = model(x)
top5 = torch.topk(torch.softmax(logits, dim=-1).squeeze(0), k=5)
print(list(zip(top5.indices.tolist(), top5.values.tolist())))
```

For a runnable end-to-end example, see `infer.py`.

## Reproducing the paper

1. Download the Tesserae dataset from `https://huggingface.co/datasets/TesseraeAnon/tesserae-dataset`. The expected layout is a directory of npy files (`block.npy`, `filetype_id.npy`) plus a separate splits directory with `train_indices.npy`, `val_indices.npy`, `test_indices.npy`, and matching `_4k_groups.npy` / `_8k_groups.npy` / `_16k_groups.npy` files.

2. Train MoDiCo with the decoupled two-stage recipe:

   ```
   python train.py \
       --npy_dir /path/to/blocks_dataset_npy \
       --splits_dir /path/to/splits \
       --num_classes 619 \
       --stage1_epochs 20 --stage2_epochs 10 \
       --batch_size 64 --lr 1e-4 \
       --checkpoint_dir ./checkpoints/modico
   ```

   Multi-GPU training is supported via `torchrun`:

   ```
   torchrun --nproc_per_node=4 train.py \
       --npy_dir /path/to/blocks_dataset_npy \
       --splits_dir /path/to/splits \
       --batch_size 32
   ```

3. Train any baseline:

   ```
   python train_baseline.py --model bytercnn \
       --npy_dir /path/to/blocks_dataset_npy \
       --splits_dir /path/to/splits \
       --block_size 512 \
       --epochs 20 --batch_size 128 \
       --checkpoint_dir ./checkpoints/bytercnn
   ```

   Available baselines: `bytercnn`, `fifty`, `cnn_lstm`, `dscse`, `byteformer`, `byteresnet`, `sift`.

4. Evaluate on the test split:

   ```
   python evaluate.py --model modico \
       --checkpoint ./checkpoints/modico/stage2_best.pt \
       --npy_dir /path/to/blocks_dataset_npy \
       --splits_dir /path/to/splits \
       --block_size 512 \
       --output_dir ./results/modico_512
   ```

   `evaluate.py` writes `predictions.npz` and `metrics.json` into `--output_dir` and prints top-1, top-5, balanced top-1, and balanced top-5 accuracy.

## Repository structure

```
tesserae-modico/
  modico/                 MoDiCo model code
    motif.py              CNN local-pattern encoder
    distribution.py       byte-distribution + entropy-CDF encoder
    context.py            hierarchical byte-level transformer
    fusion.py             attentive fusion of the three encoders
    classifier.py         the full MoDiCoClassifier and a collate function
  data/                   dataset loaders and samplers
    dataset.py            Tesserae loaders for 512 / 4 K / 8 K / 16 K blocks
    samplers.py           class-balanced and progressively-balanced samplers
  baselines/              competitor architectures
    bytercnn.py           ByteRCNN (Skracic et al., 2023)
    fifty.py              FiFTy (Mittal et al., 2019)
    cnn_lstm.py           CNN-LSTM (Zhu et al., 2023)
    dscse.py              DSC-SE (Ghaleb et al., 2023)
    byteformer.py         ByteFormer / ByteResNet (Liu et al., 2024)
    sift.py               TF-IDF + Random Forest baseline
  train.py                MoDiCo decoupled training entry point
  train_baseline.py       unified baseline training entry point
  evaluate.py             evaluation entry point
  infer.py                single-file inference with a trained checkpoint
  filetype_groups.json    semantic groupings used for grouped-confusion plots
```

## Citation

```bibtex
@misc{tesserae_modico_2026,
  title  = {Tesserae and MoDiCo: A Billion-Fragment Dataset and Multi-Branch
            Architecture for File Fragment Classification},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Anonymous submission}
}
```

## License

This project is released under the Apache License, Version 2.0. See the `LICENSE` file for the full text.
