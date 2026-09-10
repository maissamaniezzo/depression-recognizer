"""Index computations: detrending, ALFF/fALFF, ReHo.

Both NumPy and PyTorch implementations are provided. Torch is preferred
when CUDA is available; otherwise we fall back to NumPy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..utils.optional_imports import optional_torch


# ----------------------------- Detrending -----------------------------------

def detrend_linear_np(block_np: np.ndarray) -> np.ndarray:
    """Linear detrending along the last axis. NumPy implementation."""
    _, _, _, T = block_np.shape
    t = np.arange(T, dtype=np.float32)
    s_t, s_t2 = t.sum(), (t * t).sum()

    s_y = block_np.sum(axis=3)
    s_ty = (block_np * t).sum(axis=3)
    denom = (T * s_t2 - s_t * s_t)
    denom = np.where(denom <= 0, 1e-12, denom)

    b = (T * s_ty - s_t * s_y) / denom
    a = (s_y / T) - b * (s_t / T)
    return block_np - (a[..., None] + b[..., None] * t)


# ----------------------------- ALFF / fALFF --------------------------------

def alff_falff_np(
    block_dt: np.ndarray, TR: float, low: float, high: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ALFF and fALFF maps from a detrended 4D block. NumPy implementation."""
    T = block_dt.shape[-1]
    F = np.fft.rfft(block_dt, axis=-1)
    P = F.real ** 2 + F.imag ** 2
    freqs = np.fft.rfftfreq(T, d=TR)
    band = (freqs >= low) & (freqs <= high)

    band_power = P[..., band].sum(axis=-1)
    total_power = P.sum(axis=-1) + 1e-12

    ALFF = np.sqrt(np.maximum(band_power, 0.0)).astype(np.float32)
    fALFF = (ALFF / np.sqrt(total_power)).astype(np.float32)
    return ALFF, fALFF


def alff_falff_torch(
    block_dt: np.ndarray, TR: float, low: float, high: float, device: "torch.device"
) -> Tuple[np.ndarray, np.ndarray]:
    """ALFF and fALFF using ``torch.fft`` on the given device."""
    torch, _ = optional_torch()
    x = torch.from_numpy(block_dt).to(device, non_blocking=True)
    Freq = torch.fft.rfft(x, dim=-1)
    P = Freq.real ** 2 + Freq.imag ** 2
    freqs = torch.fft.rfftfreq(block_dt.shape[-1], d=TR, device=device)
    band = (freqs >= low) & (freqs <= high)

    band_power = P[..., band].sum(dim=-1)
    total_power = P.sum(dim=-1).clamp_min(1e-12)

    ALFF = torch.sqrt(torch.clamp(band_power, min=0))
    fALFF = ALFF / torch.sqrt(total_power)
    return ALFF.float().cpu().numpy(), fALFF.float().cpu().numpy()


# ----------------------------- ReHo ----------------------------------------

def make_onehot_kernels(neigh: int, device, dtype) -> "torch.Tensor":
    """``(k, 1, neigh, neigh, neigh)`` one-hot kernels for im2col via conv3d."""
    torch, _ = optional_torch()
    k = neigh ** 3
    W = torch.zeros((k, 1, neigh, neigh, neigh), device=device, dtype=dtype)
    idx = 0
    for dz in range(neigh):
        for dy in range(neigh):
            for dx in range(neigh):
                W[idx, 0, dz, dy, dx] = 1.0
                idx += 1
    return W


@torch.no_grad()
def reho_tile_torch(
    x_dt_tile: "torch.Tensor", neigh: int, chunk_t: int
) -> "torch.Tensor":
    """Kendall's W (ReHo) for a 4D block with halo. Returns ``(X', Y', Z')``."""
    torch, F = optional_torch()
    Xp, Yp, Zp, T = x_dt_tile.shape
    device, dtype = x_dt_tile.device, x_dt_tile.dtype
    r = neigh // 2
    k = neigh ** 3

    W = make_onehot_kernels(neigh, device, dtype)
    xT = x_dt_tile.permute(3, 2, 1, 0).unsqueeze(1).contiguous()
    pad = (r, r, r, r, r, r)

    R_sum = torch.zeros((k, Zp, Yp, Xp), device=device, dtype=dtype)
    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        chunk = xT[t0:t1]
        vals = F.conv3d(F.pad(chunk, pad=pad, mode="replicate"), W)
        order = vals.argsort(dim=1)
        ranks = torch.empty_like(order, dtype=dtype)
        base = torch.arange(k, device=device, dtype=dtype).view(1, k, 1, 1, 1).expand_as(order)
        ranks.scatter_(1, order, base)
        ranks = ranks + 1.0
        R_sum += ranks.sum(dim=0)
        del chunk, vals, order, ranks, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    R_bar = R_sum.mean(dim=0, keepdim=True)
    S = ((R_sum - R_bar) ** 2).sum(dim=0)
    denom = (k ** 2) * (T ** 3 - T)
    Wmap = (12.0 * S / denom).clamp(0.0, 1.0)
    return Wmap.permute(2, 1, 0).contiguous()
