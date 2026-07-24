"""PyTorch ``Dataset`` and split manifest reader for the 3D-index NIfTI files.

Each sample is a NIfTI with shape ``(X, Y, Z, 3)`` where the last axis
holds the ALFF / fALFF / ReHo channels. The dataset applies a per-channel
z-score and resizes the volume to ``target_shape`` via trilinear
interpolation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .. import config
from ..utils.optional_imports import optional_torch
from ..utils.paths import is_nifti, norm_path


def build_items_from_split(
    root_dir: str, split: str, manifest_name: str = "manifest.csv"
) -> List[Dict[str, Any]]:
    """Build a list of items for a split from a manifest or folder layout.

    Each item is a dict with ``"nii"`` (path), ``"label"`` (0/1) and
    ``"id"`` (participant id, optional).
    """
    root = Path(root_dir)
    assert root.exists(), f"Pasta não encontrada: {root}"
    items: List[Dict[str, Any]] = []

    manifest = root / manifest_name
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("split", "").strip().lower() != split.lower()):
                    continue
                group = (row.get("group", "") or "").strip().lower()
                label = config.LABELS_MAP.get(group)
                if label is None:
                    continue
                v = row.get("dest_path") or row.get("src_path")
                p = norm_path(v)
                path = p if p.is_absolute() else (root.parent / p)
                if is_nifti(path) and path.exists():
                    items.append({
                        "nii": str(path),
                        "label": int(label),
                        "id": row.get("participant_id", path.stem),
                    })
    else:
        for cls in ("control", "depr"):
            label = config.LABELS_MAP.get(cls)
            if label is None:
                continue
            base = root / split / cls
            for p in list(base.rglob("*.nii.gz")) + list(base.rglob("*.nii")):
                items.append({
                    "nii": str(p),
                    "label": int(label),
                    "id": p.stem,
                })

    if not items:
        raise RuntimeError(
            f"Nenhum NIfTI encontrado em {root} para split='{split}'. "
            f"Verifique manifest.csv e/ou estrutura de pastas."
        )
    return items


class IndicesNifti4DDataset:
    """PyTorch-compatible dataset over ``(X, Y, Z, 3)`` index volumes.

    Returns ``(x, y)`` by default, or ``(x, y, id, path)`` when
    ``return_id=True``.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        target_shape=(64, 96, 96),
        zscore: bool = True,
        return_id: bool = False,
    ):
        self.items = items
        self.target_shape = tuple(target_shape)
        self.zscore = zscore
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        torch, F = optional_torch()

        item = self.items[i]
        path = item["nii"]
        y = int(item["label"])

        # Load the 4D NIfTI: (X, Y, Z, 3) with [ALFF, fALFF, ReHo]
        import nibabel as nib  # local import: nibabel is a hard dep

        vol = nib.load(path).get_fdata(dtype=np.float32)
        if vol.ndim != 4 or vol.shape[-1] != 3:
            raise ValueError(f"{path}: esperado (X,Y,Z,3), obtido {vol.shape}")

        # Rearrange to (C=3, D=Z, H=Y, W=X)
        vol = np.moveaxis(vol, -1, 0)
        vol = vol.transpose(0, 3, 2, 1)
        vol = np.nan_to_num(vol, copy=False)

        if self.zscore:
            for c in range(3):
                m = vol[c].mean()
                s = vol[c].std() + 1e-6
                vol[c] = (vol[c] - m) / s

        t = torch.from_numpy(vol)[None, ...]                       # [1, 3, D, H, W]
        t = F.interpolate(t, size=self.target_shape, mode="trilinear", align_corners=False)
        x = t[0]                                                   # [3, D, H, W]

        if self.return_id:
            uid = item.get("id", Path(path).stem)
            return x, torch.tensor(y, dtype=torch.long), uid, path
        return x, torch.tensor(y, dtype=torch.long)
