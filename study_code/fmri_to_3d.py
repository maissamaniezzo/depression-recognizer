#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um NIfTI 4D (X,Y,Z,3) com canais: [ALFF, fALFF, ReHo].
- GPU (CUDA) automática via PyTorch sempre que disponível (--device auto/cuda/cpu).
- Aceita arquivo único (--nii/--out) OU pasta recursiva (--in_dir/--out_dir).
- Progresso com tqdm: global (arquivos) e interno (ReHo por tempo, com chunks).
- ReHo 3D acelerado: conv3d one-hot (im2col), ranking argsort/scatter_, soma por tempo.

Usar:
    python fmri_to_3d.py --nii path/in.nii.gz --tr 2.5 --out path/out_3d.nii.gz
    python fmri_to_3d.py --in_dir /dados/fmri --out_dir /dados/indices --tr 2.5
    # salva como <nome>_3d.nii.gz em /dados/indices/<subpastas>

Como foi gerado dataset_3d
    python depression-recognizer/study_code/fmri_to_3d.py \
        --in_dir depression-recognizer/ds002748 \
        --out_dir depression-recognizer/dataset_3d  \
        --tr 2.5 --neighbor 3   --patch 64 64 64 \
        --chunk_t 16 --max_split_mb 32 --device cuda --no_inner_tqdm
"""

import argparse
from pathlib import Path
import os
from typing import List, Tuple
import traceback

import numpy as np
import nibabel as nib
from tqdm import tqdm

# PyTorch para GPU e FFT
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception as e:
    _HAS_TORCH = False
    TORCH_ERR = e

# (opcional) dicas ao alocador para GPUs pequenas (evita fragmentação)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:32,expandable_segments:True")



# ----------------------------- Utilidades de I/O -----------------------------

def find_niigz_recursive(in_dir: Path) -> List[Path]:
    # sub-*/func/*_bold.nii.gz apenas
    return sorted([p for p in in_dir.glob("**/func/*_bold.nii.gz")
                   if p.is_file() and ".git" not in p.parts])



def make_out_path_for(in_file: Path, in_root: Path, out_root: Path, suffix: str = "_3d") -> Path:
    rel = in_file.relative_to(in_root)
    out_dir = (out_root / rel.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel.name[:-7] if rel.name.endswith(".nii.gz") else rel.stem  # remove .nii.gz
    out_name = f"{stem}{suffix}.nii.gz"
    return out_dir / out_name


def save_indices_nifti(
    alff: np.ndarray, falff: np.ndarray, reho: np.ndarray,
    affine: np.ndarray, ref_header: nib.nifti1.Nifti1Header, out_path: Path
):
    # Empilha no último eixo: (X,Y,Z,3)
    vol = np.stack([alff, falff, reho], axis=-1).astype(np.float32)
    hdr = ref_header.copy()
    # Ajusta dtype e intenção (opcional: "vector")
    hdr.set_data_dtype(np.float32)
    try:
        hdr.set_intent('vector', (), '')
    except Exception:
        pass
    img = nib.Nifti1Image(vol, affine=affine, header=hdr)
    nib.save(img, str(out_path))

def as_tuple3(x):
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return tuple(int(v) for v in x)
    return (int(x), int(x), int(x))

def iter_tiles_xyz(X, Y, Z, patch, halo):
    """Gera fatias 3D com 'halo' (para ReHo). Retorna (core_slices, padded_slices, crop_from_pad)."""
    px, py, pz = patch
    hx, hy, hz = halo
    for x0 in range(0, X, px):
        x1 = min(X, x0 + px)
        xs0 = max(0, x0 - hx); xs1 = min(X, x1 + hx)
        cx0 = x0 - xs0;        cx1 = cx0 + (x1 - x0)  # recorte de volta do bloco "padded"
        for y0 in range(0, Y, py):
            y1 = min(Y, y0 + py)
            ys0 = max(0, y0 - hy); ys1 = min(Y, y1 + hy)
            cy0 = y0 - ys0;        cy1 = cy0 + (y1 - y0)
            for z0 in range(0, Z, pz):
                z1 = min(Z, z0 + pz)
                zs0 = max(0, z0 - hz); zs1 = min(Z, z1 + hz)
                cz0 = z0 - zs0;        cz1 = cz0 + (z1 - z0)
                core = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
                pad  = (slice(xs0, xs1), slice(ys0, ys1), slice(zs0, zs1))
                crop = (slice(cx0, cx1), slice(cy0, cy1), slice(cz0, cz1))
                yield core, pad, crop

@torch.no_grad()
def reho_tile(x_dt_tile: torch.Tensor, neigh: int, chunk_t: int) -> torch.Tensor:
    """
    x_dt_tile: (X',Y',Z',T) **com halo já incluso** (para bordas).
    Retorna ReHo para o núcleo (sem halo) – por isso quem chama recorta depois.
    """
    Xp, Yp, Zp, T = x_dt_tile.shape
    device, dtype = x_dt_tile.device, x_dt_tile.dtype
    r = neigh // 2
    k = neigh ** 3

    # conv3d one-hot
    W = torch.zeros((k, 1, neigh, neigh, neigh), device=device, dtype=dtype)
    idx = 0
    for dz in range(neigh):
        for dy in range(neigh):
            for dx in range(neigh):
                W[idx, 0, dz, dy, dx] = 1.0; idx += 1

    # tratar T como batch
    xT = x_dt_tile.permute(3, 2, 1, 0).unsqueeze(1).contiguous()  # (T,1,Z',Y',X')
    pad = (r, r, r, r, r, r)
    R_sum = torch.zeros((k, Zp, Yp, Xp), device=device, dtype=dtype)

    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        chunk = xT[t0:t1]
        vals = F.conv3d(F.pad(chunk, pad=pad, mode="replicate"), W)
        order = vals.argsort(dim=1)
        ranks = torch.empty_like(order, dtype=dtype)
        base  = torch.arange(k, device=device, dtype=dtype).view(1, k, 1, 1, 1).expand_as(order)
        ranks.scatter_(1, order, base); ranks = ranks + 1.0
        R_sum += ranks.sum(dim=0)
        del chunk, vals, order, ranks, base
        if device.type == "cuda": torch.cuda.empty_cache()

    R_bar = R_sum.mean(dim=0, keepdim=True)
    S = ((R_sum - R_bar) ** 2).sum(dim=0)
    denom = (k**2) * (T**3 - T)
    Wmap = (12.0 * S / denom).clamp(0.0, 1.0)                    # (Z',Y',X')
    return Wmap.permute(2, 1, 0).contiguous()                    # (X',Y',Z')


# ----------------------- Cálculos (GPU/CPU via PyTorch) ----------------------

def get_device(dev_arg: str) -> torch.device:
    if dev_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponível, use --device cpu ou auto.")
        print("GPUs visíveis:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
        print("Device atual:", torch.cuda.current_device())
        return torch.device("cuda")
    if dev_arg == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def detrend_linear_torch(x: torch.Tensor, TR: float) -> torch.Tensor:
    """
    Remove média e tendência linear ao longo do tempo.
    x: (X,Y,Z,T) float32
    """
    T = x.shape[-1]
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    s_t = t.sum()
    s_t2 = (t * t).sum()

    s_y = x.sum(dim=-1)                 # (X,Y,Z)
    s_ty = (x * t).sum(dim=-1)          # (X,Y,Z)

    denom = (T * s_t2 - s_t * s_t).clamp_min(1e-12)
    b = (T * s_ty - s_t * s_y) / denom  # (X,Y,Z)
    a = (s_y / T) - b * (s_t / T)       # (X,Y,Z)

    trend = a[..., None] + b[..., None] * t  # (X,Y,Z,T)
    return x - trend


def alff_falff_torch(x_dt: torch.Tensor, TR: float, low: float, high: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_dt: (X,Y,Z,T) detrend
    Retorna (ALFF, fALFF) em 3D (X,Y,Z), float32
    """
    T = x_dt.shape[-1]
    # FFT real ao longo de T
    Freq = torch.fft.rfft(x_dt, dim=-1)
    P = (Freq.real**2 + Freq.imag**2)
    freqs = torch.fft.rfftfreq(T, d=TR, device=x_dt.device)
    band = (freqs >= low) & (freqs <= high)
    band_power = P[..., band].sum(dim=-1)
    total_power = P.sum(dim=-1).clamp_min(1e-12)
    ALFF = torch.sqrt(band_power.clamp_min(0))
    fALFF = ALFF / torch.sqrt(total_power)
    return ALFF, fALFF


def make_onehot_kernels(neigh: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Conjunto de filtros one-hot (k=neigh^3) para extrair vizinhanças via conv3d.
    Retorna W com shape (k, 1, D, H, W)
    """
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
def reho_kendalls_w_torch(
    x_dt: torch.Tensor, neigh: int = 3, chunk_t: int = 32, pbar: tqdm = None
) -> torch.Tensor:
    """
    ReHo voxel-wise (Kendall's W) no GPU/CPU via conv3d e ranking.
    x_dt: (X,Y,Z,T) float32
    Retorna W: (X,Y,Z) float32 em [0,1].
    """
    assert neigh % 2 == 1, "neighbor deve ser ímpar (3, 5, ...)."
    X, Y, Z, T = x_dt.shape
    device, dtype = x_dt.device, x_dt.dtype
    r = neigh // 2
    k = neigh ** 3

    # Pesos one-hot para "im2col" 3D
    W = make_onehot_kernels(neigh, device, dtype)

    # Vamos tratar T como "batch": (T,1,Z,Y,X)
    xT = x_dt.permute(3, 2, 1, 0)  # (T, Z, Y, X)
    xT = xT.unsqueeze(1).contiguous()  # (T, 1, Z, Y, X)

    # Padding espacial para preservar shape
    pad = (r, r, r, r, r, r)  # (Wl, Wr, Hl, Hr, Dl, Dr)
    # Soma de ranks por série (k) ao longo de T
    R_sum = torch.zeros((k, Z, Y, X), device=device, dtype=dtype)

    # Processa o tempo em blocos
    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        chunk = xT[t0:t1]  # (B,1,Z,Y,X)

        # pad + conv3d => (B, k, Z, Y, X)
        chunk_p = F.pad(chunk, pad=pad, mode='replicate')
        vals = F.conv3d(chunk_p, W, bias=None, stride=1, padding=0)

        # ranks por tempo em k (dim=1), com ordenação inteira (ties raros em float)
        order = vals.argsort(dim=1)  # (B,k,Z,Y,X) índices do menor->maior
        ranks = torch.empty_like(order, dtype=dtype)
        base = torch.arange(k, device=device, dtype=dtype).view(1, k, 1, 1, 1).expand_as(order)
        ranks.scatter_(1, order, base)  # inverte a permutação
        ranks = ranks + 1.0  # 1..k
        # acumula sobre o batch de tempo (B)
        R_sum = R_sum + ranks.sum(dim=0)

        if pbar is not None:
            pbar.update(t1 - t0)

        # libera memória intermediária
        del chunk, chunk_p, vals, order, ranks, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Kendall's W (sem correção de empates; para floats empates são raros)
    R_bar = R_sum.mean(dim=0, keepdim=True)            # (1, Z, Y, X)
    S = ((R_sum - R_bar) ** 2).sum(dim=0)              # (Z, Y, X)
    denom = (k**2) * (T**3 - T)
    Wmap = (12.0 * S / denom).clamp(0.0, 1.0)          # (Z, Y, X)

    # retorna (X,Y,Z)
    return Wmap.permute(2, 1, 0).contiguous()


def process_one_file(
    in_path: Path, out_path: Path, TR: float, low: float, high: float,
    neighbor: int, device: torch.device, chunk_t: int, inner_tqdm: bool,
    patch_xyz=(64,64,64), max_split_mb: int = 32
):
    # leitura preguiçosa
    img  = nib.load(str(in_path), mmap=True)
    if img.ndim != 4:
        raise ValueError(f"{in_path}: esperado 4D (X,Y,Z,T), mas obtive {img.shape}")
    data = img.dataobj  # não carrega 4D inteiro
    if img.ndim != 4:
        raise ValueError(f"{in_path.name}: esperado 4D, recebi {img.shape}")
    X, Y, Z, T = img.shape

    # ajusta alocador (anti-fragmentação)
    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{int(max_split_mb)},expandable_segments:True"
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False  # evita workspaces gigantes

    # Saídas acumuladas (CPU) – vamos montar por tiles
    ALFF_out = np.zeros((X, Y, Z), dtype=np.float32)
    fALFF_out = np.zeros((X, Y, Z), dtype=np.float32)
    ReHo_out = np.zeros((X, Y, Z), dtype=np.float32)

    # halo para ReHo
    r = neighbor // 2
    patch = as_tuple3(patch_xyz)
    halo  = (r, r, r)

    # barra de progresso externa
    outer = tqdm(total=((X + patch[0]-1)//patch[0]) * ((Y + patch[1]-1)//patch[1]) * ((Z + patch[2]-1)//patch[2]),
                 desc=f"Tiles {in_path.name}", unit="tile", leave=False)

    with torch.no_grad():
        # percorre tiles espaciais
        for core, pad, crop in iter_tiles_xyz(X, Y, Z, patch, halo):
            # fatia com halo: (X',Y',Z',T); já vem como memmap -> vira np.ndarray leve
            block_np = np.asarray(data[pad[0], pad[1], pad[2], :], dtype=np.float32)

            # detrend (CPU) para poupar VRAM; é barato
            t = np.arange(T, dtype=np.float32)
            s_t, s_t2 = t.sum(), (t*t).sum()
            s_y = block_np.sum(axis=3);           # (X',Y',Z')
            s_ty = (block_np * t).sum(axis=3);    # (X',Y',Z')
            denom = (T * s_t2 - s_t * s_t); denom = np.where(denom <= 0, 1e-12, denom)
            b = (T * s_ty - s_t * s_y) / denom
            a = (s_y / T) - b * (s_t / T)
            block_dt = block_np - (a[..., None] + b[..., None] * t)  # (X',Y',Z',T)
            del block_np, s_y, s_ty, a, b

            # ---------- ALFF / fALFF (GPU se couber; senão CPU) ----------
            # fazemos FFT no bloco inteiro ao longo do tempo (T), mas com patch pequeno cabe.
            freqs = np.fft.rfftfreq(T, d=TR)
            band  = (freqs >= low) & (freqs <= high)
            try:
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        x = torch.from_numpy(block_dt).to(device, non_blocking=True)
                        Freq = torch.fft.rfft(x, dim=-1)                         # complex64 sob autocast
                        P = (Freq.real**2 + Freq.imag**2)
                        band_power  = P[..., band].sum(dim=-1)
                        total_power = P.sum(dim=-1).clamp_min(1e-12)
                        ALFF_t  = torch.sqrt(torch.clamp(band_power, min=0))
                        fALFF_t = ALFF_t / torch.sqrt(total_power)
                        ALFF_tile  = ALFF_t.float().detach().cpu().numpy()
                        fALFF_tile = fALFF_t.float().detach().cpu().numpy()
                        del x, Freq, P, band_power, total_power, ALFF_t, fALFF_t
                        torch.cuda.empty_cache()
                else:
                    # CPU fallback (mais lento, pouca RAM extra)
                    F = np.fft.rfft(block_dt, axis=-1)
                    P = (F.real**2 + F.imag**2)
                    band_power  = P[..., band].sum(axis=-1)
                    total_power = P.sum(axis=-1) + 1e-12
                    ALFF_tile  = np.sqrt(np.maximum(band_power, 0.0))
                    fALFF_tile = ALFF_tile / np.sqrt(total_power)
                    del F, P
            except RuntimeError:
                # Se mesmo assim faltou memória, cai para CPU
                F = np.fft.rfft(block_dt, axis=-1)
                P = (F.real**2 + F.imag**2)
                band_power  = P[..., band].sum(axis=-1)
                total_power = P.sum(axis=-1) + 1e-12
                ALFF_tile  = np.sqrt(np.maximum(band_power, 0.0))
                fALFF_tile = ALFF_tile / np.sqrt(total_power)
                del F, P

            # ---------- ReHo (GPU/CPU) ----------
            # manda o tile com halo; depois recorta para o núcleo
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    xdt_t = torch.from_numpy(block_dt).to(device, non_blocking=True)
                    reho_full = reho_tile(xdt_t, neigh=neighbor, chunk_t=chunk_t)  # (X',Y',Z')
                    reho_full = reho_full.float().detach().cpu().numpy()
                    del xdt_t
                    torch.cuda.empty_cache()
            else:
                # CPU: reusa mesma função (vai lento), mas evita OOM em 2 GB
                xdt_t = torch.from_numpy(block_dt)
                reho_full = reho_tile(xdt_t, neigh=neighbor, chunk_t=chunk_t).numpy()
                del xdt_t

            # recortes para posição final
            cx, cy, cz = crop
            (x0, x1), (y0, y1), (z0, z1) = (
                (core[0].start, core[0].stop),
                (core[1].start, core[1].stop),
                (core[2].start, core[2].stop),
            )

            # checagem defensiva (detecta desalinhamento cedo)
            exp_shape = (x1 - x0, y1 - y0, z1 - z0)
            alff_crop_shape  = ALFF_tile[cx, cy, cz].shape
            falff_crop_shape = fALFF_tile[cx, cy, cz].shape
            reho_crop_shape  = reho_full[cx, cy, cz].shape
            assert alff_crop_shape == exp_shape and falff_crop_shape == exp_shape and reho_crop_shape == exp_shape, \
                f"Crop != destino: ALFF {alff_crop_shape}, fALFF {falff_crop_shape}, ReHo {reho_crop_shape} vs {exp_shape}"

            # cópia com diagnóstico detalhado se der erro
            try:
                ALFF_out[x0:x1, y0:y1, z0:z1]  = ALFF_tile[cx, cy, cz]
                fALFF_out[x0:x1, y0:y1, z0:z1] = fALFF_tile[cx, cy, cz]
                ReHo_out[x0:x1, y0:y1, z0:z1]  = reho_full[cx, cy, cz]
            except Exception as e:
                print("ALFF_tile total:",  ALFF_tile.shape,
                    "fALFF_tile total:", fALFF_tile.shape,
                    "reho_full total:",  reho_full.shape)
                print("Crop idx:", cx, cy, cz,
                    "Crop shapes:", alff_crop_shape, falff_crop_shape, reho_crop_shape)
                print("Destino shape:", exp_shape,
                    "Core slices:", (x0, x1, y0, y1, z0, z1))
                raise


            del block_dt, ALFF_tile, fALFF_tile, reho_full
            outer.update(1)

    outer.close()
    save_indices_nifti(ALFF_out, fALFF_out, ReHo_out, img.affine, img.header, out_path)


# ---------------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="fMRI 4D (.nii.gz) -> NIfTI 4D (X,Y,Z,3) com [ALFF, fALFF, ReHo]")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--nii", type=str, help="Arquivo único .nii.gz de entrada (4D)")
    m.add_argument("--in_dir", type=str, help="Pasta de entrada (varredura recursiva por *.nii.gz)")
    ap.add_argument("--out", type=str, help="Arquivo .nii.gz de saída (modo arquivo único)")
    ap.add_argument("--out_dir", type=str, help="Pasta de saída (modo pasta)")
    ap.add_argument("--tr", required=True, type=float, help="TR em segundos (ex.: 2.5)")
    ap.add_argument("--low", default=0.01, type=float, help="Freq baixa (Hz) p/ ALFF/fALFF [0.01]")
    ap.add_argument("--high", default=0.10, type=float, help="Freq alta  (Hz) p/ ALFF/fALFF [0.10]")
    ap.add_argument("--neighbor", default=3, type=int, help="Vizinhança cúbica ReHo (use 3)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Acelerador [auto]")
    ap.add_argument("--chunk_t", default=32, type=int, help="Tamanho do lote no eixo tempo para ReHo [32]")
    ap.add_argument("--suffix", default="_3d", type=str, help="Sufixo do arquivo de saída [ _3d ]")
    ap.add_argument("--skip_if_exists", action="store_true", help="Pula se saída já existir")
    ap.add_argument("--no_inner_tqdm", action="store_true", help="Desativa tqdm interna por arquivo")
    ap.add_argument("--patch", nargs=3, type=int, default=[64,64,64],
                help="Tamanho do patch espacial X Y Z (default 64 64 64)")
    ap.add_argument("--max_split_mb", type=int, default=32,
                    help="Hint ao alocador para evitar fragmentação (default 32)")

    args = ap.parse_args()

    if not _HAS_TORCH:
        raise RuntimeError(f"PyTorch não encontrado: {TORCH_ERR}\nInstale com CUDA para aceleração: "
                           f"pip install torch --index-url https://download.pytorch.org/whl/cu121")

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Modo arquivo único
    if args.nii:
        in_path = Path(args.nii)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        if not args.out:
            raise ValueError("Forneça --out para modo arquivo único.")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_if_exists and out_path.exists():
            print(f"[skip] {out_path}")
            return
        process_one_file(
            in_path, out_path, args.tr, args.low, args.high, args.neighbor,
            device, args.chunk_t, inner_tqdm=(not args.no_inner_tqdm), 
            patch_xyz=args.patch, max_split_mb=args.max_split_mb
        )
        print(f"OK: {out_path}")
        return

    # Modo pasta
    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir) if args.out_dir else None
    if out_root is None:
        raise ValueError("Forneça --out_dir no modo pasta.")
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_niigz_recursive(in_root)
    if not files:
        print("Nenhum .nii.gz encontrado.")
        return

    with tqdm(total=len(files), desc="Arquivos", unit="arq") as pbar_files:
        for f in files:
            out_path = make_out_path_for(f, in_root, out_root, suffix=args.suffix)
            if args.skip_if_exists and out_path.exists():
                pbar_files.set_postfix_str(f"skip {f.name}")
                pbar_files.update(1)
                continue
            try:
                process_one_file(
                    f, out_path, args.tr, args.low, args.high, args.neighbor,
                    device, args.chunk_t, inner_tqdm=(not args.no_inner_tqdm),
                    patch_xyz=args.patch, max_split_mb=args.max_split_mb
                )
                pbar_files.set_postfix_str(f"ok {f.name}")
            except Exception:
                # imprime o traceback completo fora da barra (não trunca)
                pbar_files.write(f"[ERRO] {f}")
                pbar_files.write(traceback.format_exc())
                pbar_files.set_postfix_str(f"erro {f.name}")
                # se quiser parar no primeiro erro, descomente:
                # raise
            finally:
                pbar_files.update(1)


if __name__ == "__main__":
    main()