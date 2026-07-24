"""Path and NIfTI-related helpers."""
from __future__ import annotations

from pathlib import Path


def norm_path(s: str | None) -> Path:
    """Normalize a path string. Returns ``Path`` (empty string -> ``Path('.')``)."""
    return Path((s or "").strip().replace("\\", "/"))


def is_nifti(p: Path) -> bool:
    """Return ``True`` for ``.nii`` and ``.nii.gz`` files."""
    if p.suffix.lower() == ".nii":
        return True
    return p.suffixes[-2:] == [".nii", ".gz"]
