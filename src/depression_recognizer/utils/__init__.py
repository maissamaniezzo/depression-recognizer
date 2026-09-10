"""Shared utilities (path handling, I/O helpers, optional torch import)."""
from .paths import is_nifti, norm_path
from .optional_imports import optional_torch

__all__ = ["is_nifti", "norm_path", "optional_torch"]
