# fmri_to_indices.py
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from itertools import product
from tqdm import tqdm
from scipy.signal import detrend
from scipy.stats import rankdata

def alff_falff_3d(data4d: np.ndarray, TR: float, low: float, high: float):
    """
    data4d: (X,Y,Z,T) já detrended/limpo.
    Retorna (ALFF_3d, fALFF_3d) usando amplitude espectral por-bin:
      ALFF  = mean( sqrt(power) ) nos bins [low, high]
      fALFF = sum( sqrt(power) ) nos bins [low, high] / sum( sqrt(power) ) em todas as freq.
    Referências: Zang 2007/2004 (ALFF), Zou 2008 (fALFF). 
    """
    X, Y, Z, T = data4d.shape
    # reshape p/ (Nvox, T)
    N = X * Y * Z
    M = data4d.reshape(N, T)

    # rFFT
    freqs = np.fft.rfftfreq(T, d=TR)         # [F]
    F = np.fft.rfft(M, axis=1)               # (N, F)
    power = (F.real**2 + F.imag**2)          # espectro de potência
    amp = np.sqrt(np.maximum(power, 0.0))    # amplitude espectral por bin

    # Máscaras de banda
    band = (freqs >= low) & (freqs <= high)
    eps = 1e-12

    # ALFF: média das amplitudes na banda (equivale a soma/num_bins, difere por constante)
    nb = max(int(band.sum()), 1)
    alff = amp[:, band].sum(axis=1) / nb

    # fALFF: soma(amp banda) / soma(amp total)
    falff = amp[:, band].sum(axis=1) / (amp.sum(axis=1) + eps)

    return (alff.reshape(X, Y, Z).astype(np.float32),
            falff.reshape(X, Y, Z).astype(np.float32))

def kendalls_w(timeseries_2d: np.ndarray) -> float:
    """
    Kendall's W em (k, T) para ReHo.
    """
    k, T = timeseries_2d.shape
    if k < 2 or T < 2:
        return 0.0
    # ranks por coluna temporal
    ranks = np.vstack([
        rankdata(timeseries_2d[:, t], method='average') for t in range(T)
    ]).T  # (k, T)
    R = ranks.sum(axis=1)
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()
    denom = (k**2) * (T**3 - T)
    if denom <= 0:
        return 0.0
    W = 12.0 * S / denom
    if not np.isfinite(W):
        return 0.0
    return float(np.clip(W, 0.0, 1.0))

def reho_3d(data4d: np.ndarray, neigh: int = 3) -> np.ndarray:
    """
    ReHo voxel-wise (Kendall's W) com vizinhança cúbica neigh×neigh×neigh (padrão 3).
    Implementação direta (pode ser lenta).
    """
    X, Y, Z, T = data4d.shape
    r = neigh // 2
    pad_width = ((r, r), (r, r), (r, r), (0, 0))
    D = np.pad(data4d, pad_width, mode='edge')
    out = np.zeros((X, Y, Z), dtype=np.float32)

    # Obs: aqui iteramos no mesmo ordenamento dos eixos (x,y,z)
    for x, y, z in tqdm(product(range(X), range(Y), range(Z)),
                        total=X*Y*Z, desc="ReHo 3D (Kendall W)"):
        xb, yb, zb = x + r, y + r, z + r
        block = D[xb - r: xb + r + 1,
                  yb - r: yb + r + 1,
                  zb - r: zb + r + 1, :]
        k = neigh * neigh * neigh
        series = block.reshape(k, T)
        out[x, y, z] = kendalls_w(series)
    return out

def main():
    ap = argparse.ArgumentParser(description="fMRI 4D -> índices ALFF, fALFF e ReHo; salva NIfTI 4D (3 canais).")
    ap.add_argument("--nii", required=True, type=str, help="Caminho para .nii/.nii.gz (fMRI 4D: X,Y,Z,T)")
    ap.add_argument("--tr",  required=True, type=float, help="TR em segundos (ex.: 2.5)")
    ap.add_argument("--low", default=0.01, type=float, help="Freq baixa (Hz) para ALFF/fALFF (padrão 0.01)")
    ap.add_argument("--high", default=0.10, type=float, help="Freq alta (Hz) para ALFF/fALFF (padrão 0.10)")
    ap.add_argument("--neighbor", default=3, type=int, help="Tamanho da vizinhança cúbica para ReHo (use 3)")
    ap.add_argument("--out_nii", required=True, type=str, help="Arquivo de saída .nii.gz com 3 canais [ALFF,fALFF,ReHo]")
    args = ap.parse_args()

    nii_path = Path(args.nii)
    out_path = Path(args.out_nii)

    # 1) Ler NIfTI
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Esperava 4D (X,Y,Z,T), recebi {data.shape}")
    X, Y, Z, T = data.shape

    # 2) Limpeza mínima: centragem + detrend temporal voxel-wise
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = data - data.mean(axis=3, keepdims=True)
    data = detrend(data, axis=3, type='linear')

    # 3) ALFF & fALFF
    ALFF_3d, fALFF_3d = alff_falff_3d(data, TR=args.tr, low=args.low, high=args.high)

    # 4) ReHo (Kendall's W) com vizinhança neigh
    reho_map = reho_3d(data, neigh=int(args.neighbor)).astype(np.float32)

    # 5) Empilhar como 3 canais e salvar NIfTI
    idx_4d = np.stack([ALFF_3d, fALFF_3d, reho_map], axis=3)  # (X,Y,Z,3)
    # preservar affine e header básicos
    out_img = nib.Nifti1Image(idx_4d.astype(np.float32), affine=img.affine, header=img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))
    print(f"Salvo: {out_path}  [shape {idx_4d.shape} = (X,Y,Z,3)  ordem: ALFF,fALFF,ReHo]")

if __name__ == "__main__":
    main()
