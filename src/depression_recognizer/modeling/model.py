"""Med3D ResNet18 (Tencent MedicalNet) adapted for 3-channel fMRI inputs.

The model uses MedicalNet's ``resnet18`` (with ``conv_seg``) as the
backbone, adapts ``conv1`` from 1 to 3 input channels by replicating and
scaling the original weights, and replaces the segmentation head with a
small classification head.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .. import config
from ..utils.optional_imports import optional_torch


def _import_medicalnet_resnet18():
    """Make MedicalNet's ``models.resnet`` importable.

    The location is controlled by the ``DEPREC_MEDICALNET_DIR`` env var
    (defaults to ``<repo>/MedicalNet``). MedicalNet must be cloned
    alongside this repo — see the README.
    """
    medicalnet_dir = config.MEDICALNET_DIR
    if not medicalnet_dir.exists():
        raise FileNotFoundError(
            f"MedicalNet não encontrado em {medicalnet_dir}.\n"
            "Clone o repositório:\n"
            "    git clone https://github.com/Tencent/MedicalNet.git\n"
            "ou defina DEPREC_MEDICALNET_DIR."
        )
    sys.path.insert(0, str(medicalnet_dir))
    from models.resnet import resnet18  # type: ignore  # noqa: WPS433 (MedicalNet layout)
    return resnet18


def build_med3d_resnet18_3c(
    num_classes: int,
    med3d_weights_path: str | os.PathLike | None = None,
    sample_shape: tuple[int, int, int] = (64, 96, 96),
    dropout: float = 0.2,
) -> Any:
    """Build a Med3D ResNet18 with a 3-channel input and a classification head.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    med3d_weights_path:
        Path to ``resnet_18_23dataset.pth``. Falls back to
        ``config.MEDICALNET_WEIGHTS`` when ``None``.
    sample_shape:
        Spatial shape ``(D, H, W)`` used to instantiate the network.
    dropout:
        Dropout probability in the classification head.
    """
    torch, _ = optional_torch()
    nn = torch.nn

    weights_path = Path(med3d_weights_path) if med3d_weights_path else config.MEDICALNET_WEIGHTS
    resnet18 = _import_medicalnet_resnet18()
    D, H, W = sample_shape

    model = resnet18(
        sample_input_D=D,
        sample_input_H=H,
        sample_input_W=W,
        num_seg_classes=1,         # required signature from MedicalNet
        shortcut_type="A",         # matches the 23-datasets checkpoint
        no_cuda=False,
    )

    if weights_path and weights_path.is_file():
        state_dict = torch.load(weights_path, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"], strict=False)
            else:
                raise RuntimeError(
                    f"Checkpoint não reconhecido: chaves = {list(state_dict.keys())[:6]}"
                )

    # Adapt conv1 from 1 -> 3 channels by replicating/scaling the original weights
    old = model.conv1
    new = nn.Conv3d(
        in_channels=3,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        for c in range(3):
            new.weight[:, c, ...] = old.weight[:, 0, ...] / 3.0
    model.conv1 = new

    # Replace the segmentation head with a small classification head
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),
    )
    return model


# ----------------------------- Freeze helpers ------------------------------

def unfreeze_conv1_and_head(model) -> None:
    """Train only ``conv1`` and the classification head (``conv_seg``)."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("conv1") or name.startswith("conv_seg")


def unfreeze_head_only(model) -> None:
    """Train only the classification head (``conv_seg``)."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("conv_seg")
