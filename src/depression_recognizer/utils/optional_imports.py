"""Optional dependency: PyTorch.

The package is importable without torch for code paths that don't need it
(e.g. ``data.split``). When torch is required, callers should use
:func:`optional_torch` to defer the import and get a clean error message.
"""
from __future__ import annotations

from types import ModuleType
from typing import Tuple


def optional_torch() -> Tuple[ModuleType, ModuleType]:
    """Return ``(torch, torch.nn.functional)`` or raise a helpful ``RuntimeError``."""
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"PyTorch is required for this entry point, but failed to import: {exc}\n"
            "Install with: pip install 'depression-recognizer[cu121]'  (or [cpu])"
        ) from exc
    return torch, F
