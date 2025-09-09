# USE: python fmri_to_png.py --nii sub-01_task-rest_bold.nii.gz --tr 2.5 --out sub-01_indices.png
# opções:
#   --low 0.01 --high 0.10  (faixa ALFF/fALFF)
#   --neighbor 3            (tamanho do cubo da vizinhança para ReHo; use 3)

# fmri_to_png.py
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.signal import detrend
from scipy.stats import rankdata
from PIL import Image

def zscore_im2d(im2d: np.ndarray) -> np.ndarray:
    x = im2d.astype(np.float64)
    m, s = np.nanmean(x), np.nanstd(x)
    if s < 1e-8: s = 1.0
    x = (x - m) / s
    # min-max para 0..1 (só para salvar como PNG depois)
    x = x - x.min()
    rng = x.max() - x.min()
    if rng < 1e-12: rng = 1.0
    return x / rng

def alff_falff_3d(data4d: np.ndarray, TR: float, low: float, high: float):
    """
    data4d: (X,Y,Z,T) já detrended/limpo (ou bruto, se preferir calcular direto).
    Retorna (ALFF_3d, fALFF_3d)
    """
    X, Y, Z, T = data4d.shape
    fs = 1.0 / TR
    # reshape para (Nvox, T)
    N = X * Y * Z
    M = data4d.reshape(N, T)

    # FFT real
    freqs = np.fft.rfftfreq(T, d=TR)                    # [F]
    F = np.fft.rfft(M, axis=1)                          # (Nvox, F)
    # potência (magnitude^2). (dividir por T não afeta razões; opcional)
    P = (F.real**2 + F.imag**2)

    band = (freqs >= low) & (freqs <= high)
    # soma de potência na banda e total
    band_power = P[:, band].sum(axis=1)
    total_power = P.sum(axis=1) + 1e-12

    # ALFF ~ sqrt(power in band); fALFF = ALFF / sqrt(power total)
    ALFF = np.sqrt(np.maximum(band_power, 0.0))
    fALFF = ALFF / np.sqrt(total_power)

    return (ALFF.reshape(X, Y, Z), fALFF.reshape(X, Y, Z))

def kendalls_w(timeseries_2d: np.ndarray) -> float:
    """
    Calcula Kendall's W para um conjunto de séries (k, T),
    onde k = número de séries (vizinhos) e T = número de tempos.
    Implementação direta conforme Zang 2004 (ReHo).
    """
    k, T = timeseries_2d.shape
    # ranks por coluna (tempo): para cada t, ranquear k valores
    # rankdata: menor valor recebe rank 1; método 'average' cuida de empates
    ranks = np.vstack([rankdata(timeseries_2d[:, t], method='average') for t in range(T)]).T  # (k, T)
    R = ranks.sum(axis=1)                               # soma de ranks por série
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()
    denom = (k**2) * (T**3 - T)
    if denom <= 0:
        return 0.0
    W = 12.0 * S / denom
    # limitar a [0,1]
    if not np.isfinite(W): W = 0.0
    return float(np.clip(W, 0.0, 1.0))

def reho_3d(data4d: np.ndarray, neigh: int = 3) -> np.ndarray:
    """
    ReHo voxel-wise com vizinhança cúbica (neigh x neigh x neigh).
    data4d: (X,Y,Z,T) — idealmente já realinhado/limpo.
    Retorna mapa 3D (X,Y,Z) de Kendall's W.
    Obs: implementação direta; pode ser lenta em volumes grandes.
    """
    X, Y, Z, T = data4d.shape
    r = neigh // 2
    # padding espacial para lidar com bordas
    pad_width = ((r, r), (r, r), (r, r), (0, 0))
    D = np.pad(data4d, pad_width, mode='edge')  # (X+2r, Y+2r, Z+2r, T)
    out = np.zeros((X, Y, Z), dtype=np.float32)

    for z in range(X):
        for y in range(Y):
            for x in range(Z):
                # bloco vizinho em D: cuidado com eixos (pad fizemos como (x,y,z,t))
                xb, yb, zb = z + r, y + r, x + r
                block = D[xb - r: xb + r + 1,
                          yb - r: yb + r + 1,
                          zb - r: zb + r + 1, :]           # (neigh, neigh, neigh, T)
                k = neigh * neigh * neigh
                series = block.reshape(k, T)
                out[z, y, x] = kendalls_w(series)
    return out

def main():
    ap = argparse.ArgumentParser(description="fMRI NIfTI (4D) -> PNG 2D via ALFF, fALFF e ReHo")
    ap.add_argument("--nii", required=True, type=str, help="Caminho para .nii ou .nii.gz (fMRI 4D)")
    ap.add_argument("--tr",  required=True, type=float, help="TR em segundos (ex.: 2.5)")
    ap.add_argument("--low", default=0.01, type=float, help="Frequência baixa (Hz) para ALFF/fALFF (padrão 0.01)")
    ap.add_argument("--high", default=0.10, type=float, help="Frequência alta (Hz) para ALFF/fALFF (padrão 0.10)")
    ap.add_argument("--neighbor", default=3, type=int, help="Tamanho da vizinhança cúbica para ReHo (use 3)")
    ap.add_argument("--out", required=True, type=str, help="Arquivo PNG de saída")
    args = ap.parse_args()

    nii_path = Path(args.nii)
    out_path = Path(args.out)

    # 1) Leitura NIfTI
    img = nib.load(str(nii_path))
    data = img.get_fdata()  # (X,Y,Z,T) para fMRI
    if data.ndim != 4:
        raise ValueError(f"Esperava 4D (X,Y,Z,T), mas recebi shape {data.shape}")
    X, Y, Z, T = data.shape

    # 2) Limpeza mínima (detrend + centragem). Em pipelines completos, adiciona-se regressão de confounds, etc.
    #    Você pode substituir por nilearn.signal.clean para band-pass explícito.
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # remover média temporal e tendência linear por voxel
    data = data - data.mean(axis=3, keepdims=True)
    data = detrend(data, axis=3, type='linear')

    # 3) ALFF e fALFF (3D cada)
    ALFF_3d, fALFF_3d = alff_falff_3d(data, TR=args.tr, low=args.low, high=args.high)

    # 4) ReHo (3D) — vizinhança 3×3×3 por padrão
    reho_map = reho_3d(data, neigh=int(args.neighbor))

    # 5) Extrair fatia axial central de cada mapa 3D
    zc = Z // 2
    alff2d  = ALFF_3d[:, :, zc]
    falff2d = fALFF_3d[:, :, zc]
    reho2d  = reho_map[:, :, zc]

    # 6) Padronizar cada 2D (z-score + min-max) e fazer média
    A = zscore_im2d(alff2d)
    F = zscore_im2d(falff2d)
    R = zscore_im2d(reho2d)
    combo = (A + F + R) / 3.0  # 2D em [0,1]

    # 7) Salvar PNG (converte para 8 bits)
    im = (np.clip(combo, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(im, mode="L").save(str(out_path))
    print(f"OK! PNG salvo em: {out_path.resolve()}")

if __name__ == "__main__":
    main()
