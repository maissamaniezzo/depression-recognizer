# train_med3d_resnet18_from_nifti4d.py
import os, sys
import numpy as np
import torch.nn as nn
import nibabel as nib
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from pathlib import Path
import csv

# mapeamento de rótulos do CSV/pastas -> classe
LABELS_MAP = {
    "control": 0, "controle": 0, "hc": 0, "healthy": 0,
    "depr": 1, "depression": 1, "depressed": 1, "patient": 1, "patients": 1
}

def _norm_path(p: str) -> Path:
    p = (p or "").strip()
    return Path(p.replace("\\", "/"))

def build_items_from_split(root_dir: str, split: str, manifest_name: str = "manifest.csv"):
    """
    Lê 'manifest.csv' (se existir) e filtra por 'split' (train/validation/test).
    - Colunas esperadas: participant_id, group, split, src_path, dest_path
    - Usa 'dest_path' se existir; caso contrário, 'src_path'.
    Fallback: varrer pastas <root>/<split>/{control,depr}/**/*.nii(.gz)
    Retorna: lista de dicts {"nii": caminho, "label": 0/1}
    """
    root = Path(root_dir)
    assert root.exists(), f"Pasta não encontrada: {root}"
    items = []

    manifest = root / manifest_name
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print(row.get("split", ""), split)
                if row.get("split", "") != split:
                    continue
                group = row.get("group", "")
                label = LABELS_MAP.get(group, None)
                # print(group, label)
                if label is None:
                    continue

                p = _norm_path(row.get("dest_path"))
                if not p:
                    continue
                path = p if p.is_absolute() else (root.parent / p)
                # print(path)
                # aceitar .nii e .nii.gz
                if path.suffix.lower() == ".gz":
                    if path.exists():
                        items.append({"nii": str(path), "label": label})
    else:
        # fallback: glob nas pastas
        for cls in ("control", "depr"):
            label = LABELS_MAP.get(cls, None)
            if label is None: 
                continue
            base = root / split / cls
            for p in list(base.rglob("*.nii.gz")) + list(base.rglob("*.nii")):
                items.append({"nii": str(p), "label": label})

    if not items:
        raise RuntimeError(f"Nenhum NIfTI encontrado em {root} para split='{split}'. "
                           f"Verifique manifest.csv e/ou estrutura de pastas.")
    return items

# ----------------------------
# Dataset: NIfTI 4D (X,Y,Z,3) -> tensor [C=3, D, H, W]
# ----------------------------
class IndicesNifti4DDataset(Dataset):
    """
    items: lista de dicts {"nii": caminho_para_arquivo_4d, "label": int}
    Cada NIfTI tem shape (X,Y,Z,3) com canais [ALFF, fALFF, ReHo].
    """
    def __init__(self, items: List[Dict], target_shape=(64, 96, 96), zscore=True):
        self.items = items
        self.target_shape = tuple(target_shape)  # (D,H,W)
        self.zscore = zscore

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path = self.items[i]["nii"]
        y = int(self.items[i]["label"])

        # Carrega NIfTI 4D (X,Y,Z,3)
        vol = nib.load(path).get_fdata(dtype=np.float32)  # (X,Y,Z,3)
        if vol.ndim != 4 or vol.shape[-1] != 3:
            raise ValueError(f"{path}: esperado (X,Y,Z,3), obtido {vol.shape}")

        # Reordena para [C=3, D=Z, H=Y, W=X]
        vol = np.moveaxis(vol, -1, 0)       # (3, X, Y, Z)
        vol = vol.transpose(0, 3, 2, 1)     # (3, Z, Y, X) -> (C,D,H,W)
        vol = np.nan_to_num(vol, copy=False)

        # z-score por canal (por sujeito)
        if self.zscore:
            for c in range(3):
                m = vol[c].mean()
                s = vol[c].std() + 1e-6
                vol[c] = (vol[c] - m) / s

        # Redimensiona para shape padrão (D,H,W)
        t = torch.from_numpy(vol)[None, ...]     # [1, 3, D, H, W]
        t = F.interpolate(t, size=self.target_shape, mode="trilinear", align_corners=False)
        x = t[0]  # [3, D, H, W]
        return x, torch.tensor(y, dtype=torch.long)

# ----------------------------
# Modelo: Med3D-ResNet18 + conv1 (1->3) com pesos replicados/normalizados
# ----------------------------

def build_med3d_resnet18_3c(num_classes, med3d_weights_path, sample_shape=(64,96,96)):
    """
    ResNet-18 3D do MedicalNet (Tencent):
      - instancia com num_seg_classes (API do repo)
      - carrega pesos
      - adapta conv1 (1->3) replicando/÷3
      - substitui 'conv_seg' por cabeça de classificação (GAP3D + Linear)
    """
    MEDICALNET_DIR = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/MedicalNet"  # ajuste
    if MEDICALNET_DIR not in sys.path:
        sys.path.append(MEDICALNET_DIR)
    from models.resnet import resnet18  # MedicalNet

    D, H, W = sample_shape

    # 1) instancia com a assinatura correta
    #    (o valor de num_seg_classes aqui é irrelevante pois trocaremos o head em seguida)
    model = resnet18(
        sample_input_D=D, sample_input_H=H, sample_input_W=W,
        num_seg_classes=1,                 # <- API do MedicalNet
        shortcut_type='A',                 # <- casa com o checkpoint 23-datasets
        no_cuda=False
    )

    # 2) carrega pesos do Med3D
    if med3d_weights_path and os.path.isfile(med3d_weights_path):
        sd = torch.load(med3d_weights_path, map_location="cpu")
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            if isinstance(sd, dict) and "state_dict" in sd:
                model.load_state_dict(sd["state_dict"], strict=False)
            else:
                raise RuntimeError(f"Checkpoint não reconhecido: chaves = {list(sd.keys())[:6]}")

    # 3) adapta conv1 para 3 canais (replica/normaliza)
    old = model.conv1  # Conv3d(1,64,7,7,7,...)
    new = nn.Conv3d(
        in_channels=3,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        new.weight[:, 0, ...] = old.weight[:, 0, ...] / 3.0
        new.weight[:, 1, ...] = old.weight[:, 0, ...] / 3.0
        new.weight[:, 2, ...] = old.weight[:, 0, ...] / 3.0
    model.conv1 = new

    # 4) troca o "cabeçote de segmentação" por um de classificação
    #    Saída de layer4 na ResNet-18 (BasicBlock exp=1) tem 512 canais → usa 512 aqui.
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model

# ----------------------------
# Congelamentos utilitários
# ----------------------------
def unfreeze_conv1_and_fc(model):
    # agora 'fc' é o 'conv_seg'
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("conv1") or n.startswith("conv_seg")

def unfreeze_fc_only(model):
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("conv_seg")

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("fc")

# ----------------------------
# Loops de treino/val
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
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

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
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

# ---------------------------------
# MAIN ajustada: usa a pasta dataset_3d/train-test-validation
# ---------------------------------
def main():
    # ====== CONFIG ======
    num_classes   = 2           # depressão vs controle (ajuste conforme seu caso)
    med3d_ckpt    = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/model/resnet_18_23dataset.pth"  # pesos Med3D
    target_shape  = (64, 96, 96)  # (D,H,W) após resize
    batch_size    = 2
    lr_warmup     = 1e-4
    lr_fc         = 5e-4
    warmup_epochs = 3
    main_epochs   = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== RAIZ DO DATASET ======
    # Estrutura:
    # dataset_3d/
    #   train-test-validation/
    #     train/{control,depr}/*.nii.gz
    #     validation/{control,depr}/*.nii.gz
    #     test/{control,depr}/*.nii.gz (opcional)
    #     manifest.csv
    data_root = "/home/maissa/Documents/UNIFEI/TFG/depression-recognizer/dataset_3d/train-test-validation"

    # monta a lista a partir do manifest.csv (se existir) ou das pastas
    train_items = build_items_from_split(data_root, split="train",      manifest_name="manifest.csv")
    val_items   = build_items_from_split(data_root, split="validation", manifest_name="manifest.csv")
    # Se quiser também um conjunto de teste:
    # test_items  = build_items_from_split(data_root, split="test", manifest_name="manifest.csv")

    print(f"[INFO] N train={len(train_items)} | N val={len(val_items)}")

    # ====== DATASETS / DATALOADERS ======
    train_ds = IndicesNifti4DDataset(train_items, target_shape=target_shape, zscore=True)
    val_ds   = IndicesNifti4DDataset(val_items,   target_shape=target_shape, zscore=True)

    # Nota: ajuste num_workers conforme seu ambiente/IO
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=True)

    # ====== MODELO ======
    model = build_med3d_resnet18_3c(
        num_classes=num_classes,
        med3d_weights_path=med3d_ckpt
    ).to(device)

    # ====== WARM-UP: treinar conv1 + fc ======
    unfreeze_conv1_and_fc(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_warmup
    )

    print("\n== Warm-up (conv1 + fc) ==")
    for ep in range(1, warmup_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_dl, criterion, device)
        print(f"[WU {ep:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    # ====== TREINO PRINCIPAL: apenas fc ======
    unfreeze_fc_only(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_fc
    )

    print("\n== Treino principal (apenas fc) ==")
    for ep in range(1, main_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_dl, criterion, device)
        print(f"[FC {ep:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/med3d_resnet18_from_nifti4d.pt")
    print("\nModelo salvo em checkpoints/med3d_resnet18_from_nifti4d.pt")

if __name__ == "__main__":
    main()
