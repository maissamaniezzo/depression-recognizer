"""Centralized configuration and defaults.

Values can be overridden via environment variables so the package can run on
any host without editing the source.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# ---- Project paths ----
DATA_DIR = Path(os.environ.get("DEPREC_DATA_DIR", REPO_ROOT / "data"))
DATASET_3D_DIR = Path(os.environ.get("DEPREC_DATASET_3D_DIR", DATA_DIR / "dataset_3d"))
SPLIT_DIR = Path(os.environ.get("DEPREC_SPLIT_DIR", DATASET_3D_DIR / "train-test-validation"))
MODEL_DIR = Path(os.environ.get("DEPREC_MODEL_DIR", REPO_ROOT / "model"))
CHECKPOINTS_DIR = Path(os.environ.get("DEPREC_CHECKPOINTS_DIR", MODEL_DIR / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("DEPREC_RESULTS_DIR", REPO_ROOT / "model-verification" / "results"))

# ---- MedicalNet / pretrained weights ----
MEDICALNET_DIR = Path(
    os.environ.get(
        "DEPREC_MEDICALNET_DIR",
        str(REPO_ROOT / "MedicalNet"),
    )
)
MEDICALNET_WEIGHTS = Path(
    os.environ.get(
        "DEPREC_MEDICALNET_WEIGHTS",
        str(MODEL_DIR / "resnet_18_23dataset.pth"),
    )
)

# ---- Defaults for preprocessing ----
DEFAULT_TR = float(os.environ.get("DEPREC_TR", "2.5"))
DEFAULT_LOW = float(os.environ.get("DEPREC_ALFF_LOW", "0.01"))
DEFAULT_HIGH = float(os.environ.get("DEPREC_ALFF_HIGH", "0.10"))
DEFAULT_NEIGHBOR = int(os.environ.get("DEPREC_REHO_NEIGHBOR", "3"))
DEFAULT_PATCH = (64, 64, 64)
DEFAULT_CHUNK_T = int(os.environ.get("DEPREC_CHUNK_T", "32"))

# ---- Defaults for training ----
DEFAULT_TARGET_SHAPE = (64, 96, 96)  # (D, H, W)
DEFAULT_BATCH_SIZE = int(os.environ.get("DEPREC_BATCH_SIZE", "2"))
DEFAULT_WARMUP_EPOCHS = int(os.environ.get("DEPREC_WARMUP_EPOCHS", "100"))
DEFAULT_MAIN_EPOCHS = int(os.environ.get("DEPREC_MAIN_EPOCHS", "2000"))
DEFAULT_LR_WARMUP = float(os.environ.get("DEPREC_LR_WARMUP", "1e-4"))
DEFAULT_LR_FC = float(os.environ.get("DEPREC_LR_FC", "5e-4"))
DEFAULT_SEED = int(os.environ.get("DEPREC_SEED", "1337"))

# ---- Labels (case-insensitive) ----
LABELS_MAP: dict[str, int] = {
    "control": 0, "controle": 0, "hc": 0, "healthy": 0,
    "depr": 1, "depression": 1, "depressed": 1, "patient": 1, "patients": 1,
}


def ensure_dirs() -> None:
    """Create the on-disk directories the pipeline writes to."""
    for d in (DATA_DIR, DATASET_3D_DIR, SPLIT_DIR, MODEL_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
