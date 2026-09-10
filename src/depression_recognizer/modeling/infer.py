"""Evaluate a trained Med3D-ResNet18 on a split.

Saves:

- ``<results-dir>/<split>_predictions.csv`` — per-sample predictions
- ``<results-dir>/<split>_metrics.json``   — overall metrics
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .. import config
from ..utils.optional_imports import optional_torch
from ..data.dataset import IndicesNifti4DDataset, build_items_from_split
from .model import build_med3d_resnet18_3c


# ----------------------------- Evaluation ----------------------------------

def _confusion_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    n = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / n

    prec_pos = tp / max(1, (tp + fp))
    rec_pos = tp / max(1, (tp + fn))
    f1_pos = 2 * prec_pos * rec_pos / max(1e-12, (prec_pos + rec_pos))
    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, (tn + fp))
    f1_neg = 2 * prec_neg * rec_neg / max(1e-12, (prec_neg + rec_neg))
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        "n": tp + tn + fp + fn,
        "acc": acc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision_pos": prec_pos, "recall_pos": rec_pos, "f1_pos": f1_pos,
        "precision_neg": prec_neg, "recall_neg": rec_neg, "f1_neg": f1_neg,
        "macro_f1": macro_f1,
    }


def evaluate_split(model, loader, device, out_csv: Path) -> dict[str, Any]:
    """Run model on a loader and write per-sample predictions + summary metrics."""
    torch, _ = optional_torch()
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_y: list[int] = []
    all_p: list[int] = []
    all_prob1: list[float] = []

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "label", "pred", "prob_control", "prob_depr"])
        with torch.no_grad():
            for x, y, uid, path in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(1)
                for i in range(x.size(0)):
                    pc, pd_ = probs[i, 0].item(), probs[i, 1].item()
                    writer.writerow([uid[i], path[i], int(y[i].item()), int(pred[i].item()), f"{pc:.6f}", f"{pd_:.6f}"])
                    all_y.append(int(y[i].item()))
                    all_p.append(int(pred[i].item()))
                    all_prob1.append(pd_)

    metrics = _confusion_metrics(all_y, all_p)

    roc_auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(all_y, all_prob1))
    except Exception:
        roc_auc = None
    metrics["roc_auc"] = roc_auc
    return metrics


# ----------------------------- CLI -----------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Avalia o modelo salvo em um split.")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--data-root", type=str, default=str(config.SPLIT_DIR), dest="data_root")
    p.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINTS_DIR / "med3d_resnet18.pt"))
    p.add_argument("--med3d-ckpt", type=str, default=str(config.MEDICALNET_WEIGHTS), dest="med3d_ckpt")
    p.add_argument("--target-shape", type=int, nargs=3, default=list(config.DEFAULT_TARGET_SHAPE), dest="target_shape")
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE, dest="batch_size")
    p.add_argument("--num-workers", type=int, default=2, dest="num_workers")
    p.add_argument("--out-dir", type=str, default=str(config.RESULTS_DIR), dest="out_dir")
    return p


def main(argv: list[str] | None = None) -> None:
    torch, _ = optional_torch()
    args = _build_parser().parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = tuple(args.target_shape)

    items = build_items_from_split(args.data_root, split=args.split)
    ds = IndicesNifti4DDataset(items, target_shape=target_shape, zscore=True, return_id=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    model = build_med3d_resnet18_3c(
        num_classes=2, med3d_weights_path=args.med3d_ckpt, sample_shape=target_shape,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.split}_predictions.csv"
    metrics = evaluate_split(model, dl, device, out_csv)

    with (out_dir / f"{args.split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Predições salvas em: {out_csv}")
    print(f"Métricas salvas em:  {out_dir / f'{args.split}_metrics.json'}")
    print("Resumo:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
