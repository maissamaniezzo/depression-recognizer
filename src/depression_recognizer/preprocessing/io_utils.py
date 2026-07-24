"""BIDS helpers: sidecar JSON, output path conventions, file discovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def sidecar_json_for(nii_path: Path) -> Optional[Path]:
    """Locate the ``*.json`` sidecar corresponding to a NIfTI file (handles ``.nii.gz``)."""
    if nii_path.name.endswith(".nii.gz"):
        j = nii_path.with_suffix("").with_suffix(".json")
        return j if j.exists() else None
    if nii_path.name.endswith(".nii"):
        j = nii_path.with_suffix(".json")
        return j if j.exists() else None
    return None


def load_tr_from_sidecar(nii_path: Path) -> Optional[float]:
    """Read ``RepetitionTime`` (seconds) from a BIDS sidecar JSON."""
    j = sidecar_json_for(nii_path)
    if j is None:
        return None
    try:
        with j.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime")
        if isinstance(tr, (int, float)) and tr > 0:
            return float(tr)
    except Exception:
        return None
    return None


# ---- BIDS layouts ----
#
# Both OpenNeuro datasets used by the project have slightly different
# organizations. We expose them as named layouts so a single script can
# serve them all.
LAYOUTS: dict[str, dict] = {
    "ds002748": {
        "description": "OpenNeuro ds002748 (no session folder)",
        "glob": "sub-*/func/*_bold.nii.gz",
    },
    "ds005917": {
        "description": "OpenNeuro ds005917 (session-b0, rest task)",
        "glob": "sub-*/ses-b0/func/*task-rest_bold.nii.gz",
        "fallback_glob": "sub-*/ses-b0/func/*_bold.nii.gz",
    },
    "ds005817": {
        "description": "Alias for ds005917 (typo/old naming)",
        "glob": "sub-*/ses-b0/func/*task-rest_bold.nii.gz",
        "fallback_glob": "sub-*/ses-b0/func/*_bold.nii.gz",
    },
}


def find_bold_files(in_dir: Path, layout: str) -> List[Path]:
    """Return sorted BOLD NIfTI files for a given layout.

    ``layout`` is one of the keys in :data:`LAYOUTS`. For layouts with a
    ``fallback_glob`` (e.g. ``ds005917``), the fallback is used if the
    primary pattern yields zero results.
    """
    cfg = LAYOUTS.get(layout)
    if cfg is None:
        raise ValueError(
            f"Unknown BIDS layout {layout!r}. "
            f"Available: {sorted(LAYOUTS.keys())}"
        )
    pats = list(in_dir.glob(cfg["glob"]))
    if not pats and "fallback_glob" in cfg:
        pats = list(in_dir.glob(cfg["fallback_glob"]))
    return sorted(p for p in pats if p.is_file() and ".git" not in p.parts)


def infer_dataset_name(in_root: Path, explicit: Optional[str]) -> str:
    return explicit if explicit else in_root.name


def make_out_path(
    in_file: Path,
    in_root: Path,
    out_root: Path,
    dataset_name: str,
    layout: str,
    suffix: str = "_3d",
) -> Path:
    """Build the output path mirroring the subject hierarchy."""
    rel = in_file.relative_to(in_root)
    parts = list(rel.parts)

    if layout in {"ds002748"}:
        # ds002748 has no session folder; some downloads may still include
        # a stray "ses-*" — drop it to keep the tree flat: sub-XX/func/...
        if len(parts) >= 3 and parts[1].startswith("ses-"):
            parts.pop(1)
            rel = Path(*parts)
    # ds005917 / ds005817 keep the ses-b0/... hierarchy as is.

    out_dir = out_root / dataset_name / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = rel.name[:-7] if rel.name.endswith(".nii.gz") else rel.stem
    return out_dir / f"{stem}{suffix}.nii.gz"
