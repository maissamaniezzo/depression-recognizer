# infer_med3d_resnet18_from_nifti4d.py
"""
Avalia o modelo salvo em checkpoints/med3d_resnet18_from_nifti4d.pt
no split 'test' (ou 'validation') e exporta métricas e predições.
Mantém exatamente o mesmo preprocess do treino.
"""
import os, sys, json, csv
from pathlib import Path
from typing import List, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================
# Importa utilidades do script de treino (opção A: copiar funções)
# ==============================
# Para evitar dependências relativas, copie as funções essenciais aqui.

LABELS_MAP = {
    "control": 0, "controle": 0, "hc": 0, "healthy": 0,
    "depr": 1, "depression": 1, "depressed": 1, "patient": 1, "patients": 1
}


def _norm_path(s: str) -> Path:
    s = (s or "").strip()
    return Path(s.replace("\\", "/"))


def _is_nifti(p: Path) -> bool:
    suf = p.suffix.lower()
    if suf == ".nii":
        return True
    return p.suffixes[-2:] == [".nii", ".gz"]


def build_items_from_split(root_dir: str, split: str, manifest_name: str = "manifest.csv"):
    root = Path(root_dir)
    assert root.exists(), f"Pasta não encontrada: {root}"
    items: List[Dict] = []

    manifest = root / manifest_name
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("split", "").strip().lower() != split.lower()):
                    continue
                group = (row.get("group", "") or "").strip().lower()
                label = LABELS_MAP.get(group)
                if label is None:
                    continue
                v = row.get("dest_path") or row.get("src_path")
                p = _norm_path(v)
                path = p if p.is_absolute() else (root.parent / p)
                if _is_nifti(path) and path.exists():
                    items.append({"nii": str(path), "label": int(label), "id": row.get("participant_id", Path(path).stem)})
    else:
        for cls in ("control", "depr"):
            label = LABELS_MAP.get(cls)
            base = root / split / cls
            for p in list(base.rglob("*.nii.gz")) + list(base.rglob("*.nii")):
                items.append({"nii": str(p), "label": int(label), "id": Path(p).stem})

    if not items:
        raise RuntimeError(f"Nenhum NIfTI encontrado em {root} para split='{split}'.")
    return items


class IndicesNifti4DDataset(Dataset):
    def __init__(self, items: List[Dict], target_shape=(64, 96, 96), zscore=True):
        self.items = items
        self.target_shape = tuple(target_shape)
        self.zscore = zscore

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path = self.items[i]["nii"]
        y = int(self.items[i]["label"])
        uid = self.items[i].get("id", Path(path).stem)
        vol = nib.load(path).get_fdata(dtype=np.float32)
        if vol.ndim != 4 or vol.shape[-1] != 3:
            raise ValueError(f"{path}: esperado (X,Y,Z,3), obtido {vol.shape}")
        vol = np.moveaxis(vol, -1, 0)
        vol = vol.transpose(0, 3, 2, 1)
        vol = np.nan_to_num(vol, copy=False)
        if self.zscore:
            for c in range(3):
                m = vol[c].mean(); s = vol[c].std() + 1e-6
                vol[c] = (vol[c] - m) / s
        t = torch.from_numpy(vol)[None, ...]
        t = F.interpolate(t, size=self.target_shape, mode="trilinear", align_corners=False)
        x = t[0]
        return x, torch.tensor(y, dtype=torch.long), uid, path


def build_med3d_resnet18_3c(num_classes, med3d_weights_path, sample_shape=(64, 96, 96)):
    MEDICALNET_DIR = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/MedicalNet"
    if MEDICALNET_DIR not in sys.path:
        sys.path.append(MEDICALNET_DIR)
    from models.resnet import resnet18

    D, H, W = sample_shape
    model = resnet18(
        sample_input_D=D, sample_input_H=H, sample_input_W=W,
        num_seg_classes=1, shortcut_type='A', no_cuda=False
    )
    if med3d_weights_path and os.path.isfile(med3d_weights_path):
        sd = torch.load(med3d_weights_path, map_location="cpu")
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            if isinstance(sd, dict) and "state_dict" in sd:
                model.load_state_dict(sd["state_dict"], strict=False)
            else:
                raise
    old = model.conv1
    new = nn.Conv3d(3, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        for c in range(3):
            new.weight[:, c, ...] = old.weight[:, 0, ...] / 3.0
    model.conv1 = new
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_classes)
    )
    return model


@torch.no_grad()
def evaluate_split(model, loader, device, out_csv: Path):
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    all_y, all_p, all_prob1 = [], [], []

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "label", "pred", "prob_control", "prob_depr"])
        for x, y, uid, path in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1)
            for i in range(x.size(0)):
                pc, pd = probs[i,0].item(), probs[i,1].item()
                w.writerow([uid[i], path[i], int(y[i].item()), int(pred[i].item()), f"{pc:.6f}", f"{pd:.6f}"])
                all_y.append(int(y[i].item()))
                all_p.append(int(pred[i].item()))
                all_prob1.append(pd)

    # métricas manuais (binárias)
    tp = sum(1 for y, p in zip(all_y, all_p) if y==1 and p==1)
    tn = sum(1 for y, p in zip(all_y, all_p) if y==0 and p==0)
    fp = sum(1 for y, p in zip(all_y, all_p) if y==0 and p==1)
    fn = sum(1 for y, p in zip(all_y, all_p) if y==1 and p==0)
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec_pos = tp/max(1,(tp+fp))
    rec_pos  = tp/max(1,(tp+fn))
    f1_pos   = 2*prec_pos*rec_pos/max(1e-12,(prec_pos+rec_pos))
    prec_neg = tn/max(1,(tn+fn))
    rec_neg  = tn/max(1,(tn+fp))
    f1_neg   = 2*prec_neg*rec_neg/max(1e-12,(prec_neg+rec_neg))
    macro_f1 = (f1_pos + f1_neg)/2

    # ROC-AUC opcional com scikit-learn (se instalado)
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(all_y, all_prob1))
    except Exception:
        pass

    metrics = {
        "n": len(all_y),
        "acc": acc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision_pos": prec_pos, "recall_pos": rec_pos, "f1_pos": f1_pos,
        "precision_neg": prec_neg, "recall_neg": rec_neg, "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
    }
    return metrics


def main():
    # ====== CONFIG ======
    split = os.environ.get("SPLIT", "validation")  # pode ser "test" ou "validation"
    num_classes   = 2
    med3d_ckpt    = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/model/resnet_18_23dataset.pth"
    target_shape  = (64, 96, 96)
    batch_size    = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/dataset_3d/train-test-validation"
    model_pt  = "model/checkpoints/med3d_resnet18_from_nifti4d.pt"  # gerado no treino

    # ====== Dados ======
    items = build_items_from_split(data_root, split=split, manifest_name="manifest.csv")
    ds    = IndicesNifti4DDataset(items, target_shape=target_shape, zscore=True)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"), persistent_workers=True)

    # ====== Modelo ======
    model = build_med3d_resnet18_3c(num_classes=num_classes, med3d_weights_path=med3d_ckpt, sample_shape=target_shape)
    sd = torch.load(model_pt, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    # ====== Avaliação ======
    out_dir = Path("results-ds002748"); out_dir.mkdir(exist_ok=True, parents=True)
    out_csv = out_dir / f"{split}_predictions.csv"
    metrics = evaluate_split(model, dl, device, out_csv)

    with (out_dir / f"{split}_metrics.json").open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Predições salvas em: {out_csv}")
    print(f"Métricas salvas em:  {out_dir / f'{split}_metrics.json'}")
    print("Resumo:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
