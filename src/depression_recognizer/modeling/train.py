"""Training loop for the Med3D-ResNet18 binary classifier.

Strategy
--------
1. **Warm-up** (a few epochs): train ``conv1`` + classification head with a
   small learning rate to adapt the first conv to 3-channel inputs.
2. **Main** (many epochs): freeze the backbone and fine-tune the head with
   a larger learning rate.

Run via the Makefile target or directly::

    python -m depression_recognizer.modeling.train
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .. import config
from ..utils.optional_imports import optional_torch
from ..data.dataset import IndicesNifti4DDataset, build_items_from_split
from .model import build_med3d_resnet18_3c, unfreeze_conv1_and_head, unfreeze_head_only


# ----------------------------- Loops ---------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    torch, _ = optional_torch()
    model.train()
    total, correct, running = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running / total, correct / total


def eval_one_epoch(model, loader, criterion, device) -> tuple[float, float]:
    return _eval_one_epoch(model, loader, criterion, device)


@torch.no_grad()  # type: ignore[misc]
def _eval_one_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total, correct, running = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running / total, correct / total


# ----------------------------- CLI -----------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treina Med3D ResNet18 nos índices fMRI.")
    p.add_argument("--data-root", type=str, default=str(config.SPLIT_DIR))
    p.add_argument("--num-classes", type=int, default=2, dest="num_classes")
    p.add_argument("--med3d-ckpt", type=str, default=str(config.MEDICALNET_WEIGHTS), dest="med3d_ckpt")
    p.add_argument("--target-shape", type=int, nargs=3, default=list(config.DEFAULT_TARGET_SHAPE), dest="target_shape")
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE, dest="batch_size")
    p.add_argument("--lr-warmup", type=float, default=config.DEFAULT_LR_WARMUP, dest="lr_warmup")
    p.add_argument("--lr-fc", type=float, default=config.DEFAULT_LR_FC, dest="lr_fc")
    p.add_argument("--warmup-epochs", type=int, default=config.DEFAULT_WARMUP_EPOCHS, dest="warmup_epochs")
    p.add_argument("--main-epochs", type=int, default=config.DEFAULT_MAIN_EPOCHS, dest="main_epochs")
    p.add_argument("--num-workers", type=int, default=2, dest="num_workers")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--output", type=str, default=str(config.CHECKPOINTS_DIR / "med3d_resnet18.pt"))
    return p


def main(argv: list[str] | None = None) -> None:
    torch, _ = optional_torch()
    args = _build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    target_shape = tuple(args.target_shape)

    config.ensure_dirs()
    train_items = build_items_from_split(args.data_root, split="train")
    val_items   = build_items_from_split(args.data_root, split="validation")
    print(f"[INFO] N train={len(train_items)} | N val={len(val_items)}")

    train_ds = IndicesNifti4DDataset(train_items, target_shape=target_shape, zscore=True)
    val_ds   = IndicesNifti4DDataset(val_items,   target_shape=target_shape, zscore=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers > 0,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers > 0,
    )

    model = build_med3d_resnet18_3c(
        num_classes=args.num_classes,
        med3d_weights_path=args.med3d_ckpt,
        sample_shape=target_shape,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # ----- Warm-up -----
    unfreeze_conv1_and_head(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_warmup,
    )
    print("\n== Warm-up (conv1 + head) ==")
    for ep in range(1, args.warmup_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = _eval_one_epoch(model, val_dl, criterion, device)
        print(f"[WU {ep:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    # ----- Main -----
    unfreeze_head_only(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_fc,
    )
    print("\n== Treino principal (apenas head) ==")
    for ep in range(1, args.main_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = _eval_one_epoch(model, val_dl, criterion, device)
        print(f"[FC {ep:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"\nModelo salvo em {out_path}")


if __name__ == "__main__":
    main()
