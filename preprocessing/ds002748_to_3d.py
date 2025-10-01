#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converte fMRI 4D (.nii.gz) -> NIfTI 4D (X,Y,Z,3) [ALFF, fALFF, ReHo],
compatível com o pipeline do ds002748, sem qualquer lógica de grupos.

- Saída: OUT_DIR/<dataset_name>/sub-XX/ses-b0/func/<stem>_3d.nii.gz
- Procura apenas: **/ses-b0/func/*task-rest_bold.nii.gz (fallback para *_bold.nii.gz)
- CPU-friendly: tiles 3D + chunk no tempo; GPU opcional (PyTorch).
- TR: use --tr ou detecte do sidecar JSON com --auto_tr (BIDS: chave RepetitionTime). 

Exemplo:
  python ./preprocessing/ds002748_to_3d.py \
      --in_dir ./ds002748 --out_dir ./dataset_3d \
      --dataset_name ds002748 \
      --tr 2.5 --neighbor 3 --patch 24 24 24 --chunk_t 8 --device auto
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm

# PyTorch opcional p/ GPU e FFT
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception as e:
    _HAS_TORCH = False
    TORCH_ERR = e

# alocador anti-fragmentação (útil em GPUs pequenas)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:32,expandable_segments:True")

# ------------------------ Utilidades de I/O e BIDS -------------------------

def sidecar_json_for(nii_path: Path) -> Optional[Path]:
    """Acha o sidecar *.json correspondente ao NIfTI."""
    name = nii_path.name
    if name.endswith(".nii.gz"):
        j = nii_path.with_suffix("")  # remove .gz
        j = j.with_suffix(".json")
        return j if j.exists() else None
    elif name.endswith(".nii"):
        return nii_path.with_suffix(".json") if nii_path.with_suffix(".json").exists() else None
    return None

def load_tr_from_sidecar(nii_path: Path) -> Optional[float]:
    """Lê RepetitionTime (s) do sidecar JSON (BIDS)."""
    j = sidecar_json_for(nii_path)
    if not j:
        return None
    try:
        with j.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime", None)
        if isinstance(tr, (int,float)) and tr > 0:
            return float(tr)
    except Exception:
        pass
    return None

def find_bold_in_func(in_dir: Path) -> List[Path]:
    """
    Procura NIfTI em árvore do ds002748: sub-*/func/*_bold.nii.gz
    (sem pastas de sessão). Mantém resto da lógica idêntica.
    """
    pats = list(in_dir.glob("sub-*/func/*_bold.nii.gz"))
    if not pats:
        # fallback amplo caso a árvore tenha variações
        pats = list(in_dir.glob("**/func/*_bold.nii.gz"))
    pats = sorted([p for p in pats if p.is_file() and ".git" not in p.parts])
    return pats

def infer_dataset_name(in_root: Path, explicit: Optional[str]) -> str:
    return explicit if explicit else in_root.name

def make_out_path(in_file: Path, in_root: Path, out_root: Path,
                  dataset_name: str, suffix: str="_3d") -> Path:
    """
    Mantém a hierarquia relativa a partir do sujeito, sob OUT/<dataset>/...
    Ex.: OUT/ds002748/sub-XX/func/<stem>_3d.nii.gz
    """
    rel = in_file.relative_to(in_root)
    # remove qualquer nível 'ses-*' residual, se existir, para ficar sub-XX/func/
    parts = list(rel.parts)
    if len(parts) >= 3 and parts[1].startswith("ses-"):
        parts.pop(1)  # tira 'ses-*'
        rel = Path(*parts)
    out_dir = out_root / dataset_name / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel.name[:-7] if rel.name.endswith(".nii.gz") else rel.stem
    return out_dir / f"{stem}{suffix}.nii.gz"

# --------------------------- Núcleo de cálculo -----------------------------

def as_tuple3(x):
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return tuple(int(v) for v in x)
    return (int(x), int(x), int(x))

def iter_tiles_xyz(X, Y, Z, patch, halo):
    px, py, pz = patch
    hx, hy, hz = halo
    for x0 in range(0, X, px):
        x1 = min(X, x0 + px)
        xs0 = max(0, x0 - hx); xs1 = min(X, x1 + hx)
        cx0 = x0 - xs0;        cx1 = cx0 + (x1 - x0)
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

def get_device(dev_arg: str) -> "torch.device":
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
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def detrend_linear_np(block_np: np.ndarray) -> np.ndarray:
    Xp, Yp, Zp, T = block_np.shape
    t = np.arange(T, dtype=np.float32)
    s_t, s_t2 = t.sum(), (t*t).sum()
    s_y  = block_np.sum(axis=3)
    s_ty = (block_np * t).sum(axis=3)
    denom = (T * s_t2 - s_t * s_t); denom = np.where(denom <= 0, 1e-12, denom)
    b = (T * s_ty - s_t * s_y) / denom
    a = (s_y / T) - b * (s_t / T)
    return block_np - (a[..., None] + b[..., None] * t)

def alff_falff_np(block_dt: np.ndarray, TR: float, low: float, high: float) -> Tuple[np.ndarray,np.ndarray]:
    T = block_dt.shape[-1]
    F = np.fft.rfft(block_dt, axis=-1)
    P = (F.real**2 + F.imag**2)
    freqs = np.fft.rfftfreq(T, d=TR)
    band  = (freqs >= low) & (freqs <= high)
    band_power  = P[..., band].sum(axis=-1)
    total_power = P.sum(axis=-1) + 1e-12
    ALFF  = np.sqrt(np.maximum(band_power, 0.0)).astype(np.float32)
    fALFF = (ALFF / np.sqrt(total_power)).astype(np.float32)
    return ALFF, fALFF

@torch.no_grad()
def reho_tile_torch(x_dt_tile: "torch.Tensor", neigh: int, chunk_t: int) -> "torch.Tensor":
    Xp, Yp, Zp, T = x_dt_tile.shape
    device, dtype = x_dt_tile.device, x_dt_tile.dtype
    r = neigh // 2
    k = neigh ** 3
    W = torch.zeros((k, 1, neigh, neigh, neigh), device=device, dtype=dtype)
    idx = 0
    for dz in range(neigh):
        for dy in range(neigh):
            for dx in range(neigh):
                W[idx, 0, dz, dy, dx] = 1.0; idx += 1
    xT = x_dt_tile.permute(3, 2, 1, 0).unsqueeze(1).contiguous()
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
    Wmap = (12.0 * S / denom).clamp(0.0, 1.0)
    return Wmap.permute(2, 1, 0).contiguous()

def save_indices_nifti(alff: np.ndarray, falff: np.ndarray, reho: np.ndarray,
                       affine: np.ndarray, ref_header: nib.nifti1.Nifti1Header, out_path: Path):
    vol = np.stack([alff, falff, reho], axis=-1).astype(np.float32)  # (X,Y,Z,3)
    hdr = ref_header.copy()
    hdr.set_data_dtype(np.float32)
    try: hdr.set_intent('vector', (), '')
    except Exception: pass
    nib.save(nib.Nifti1Image(vol, affine=affine, header=hdr), str(out_path))

# ----------------------------- Processamento --------------------------------

def process_one_file(in_path: Path, out_path: Path, TR: float, low: float, high: float,
                     neighbor: int, device: "torch.device", chunk_t: int,
                     patch_xyz=(64,64,64), max_split_mb: int = 32):
    img = nib.load(str(in_path), mmap=True)
    if img.ndim != 4:
        raise ValueError(f"{in_path}: esperado 4D (X,Y,Z,T); recebi {img.shape}")
    data = img.dataobj
    X, Y, Z, T = img.shape

    if _HAS_TORCH and device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{int(max_split_mb)},expandable_segments:True"
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False

    ALFF_out  = np.zeros((X, Y, Z), dtype=np.float32)
    fALFF_out = np.zeros((X, Y, Z), dtype=np.float32)
    ReHo_out  = np.zeros((X, Y, Z), dtype=np.float32)

    r = neighbor // 2
    patch = as_tuple3(patch_xyz)
    halo  = (r, r, r)

    outer = tqdm(total=((X + patch[0]-1)//patch[0]) * ((Y + patch[1]-1)//patch[1]) * ((Z + patch[2]-1)//patch[2]),
                 desc=f"Tiles {in_path.name}", unit="tile", leave=False)

    for core, pad, crop in iter_tiles_xyz(X, Y, Z, patch, halo):
        block_np = np.asarray(data[pad[0], pad[1], pad[2], :], dtype=np.float32)
        block_dt = detrend_linear_np(block_np)
        del block_np

        # ALFF / fALFF
        try:
            if _HAS_TORCH and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    x = torch.from_numpy(block_dt).to(device, non_blocking=True)
                    Freq = torch.fft.rfft(x, dim=-1)
                    P = (Freq.real**2 + Freq.imag**2)
                    freqs = torch.fft.rfftfreq(block_dt.shape[-1], d=TR, device=device)
                    band = (freqs >= low) & (freqs <= high)
                    band_power  = P[..., band].sum(dim=-1)
                    total_power = P.sum(dim=-1).clamp_min(1e-12)
                    ALFF_t  = torch.sqrt(torch.clamp(band_power, min=0))
                    fALFF_t = ALFF_t / torch.sqrt(total_power)
                    ALFF_tile  = ALFF_t.float().cpu().numpy()
                    fALFF_tile = fALFF_t.float().cpu().numpy()
                    del x, Freq, P, freqs, band, band_power, total_power, ALFF_t, fALFF_t
                    torch.cuda.empty_cache()
            else:
                ALFF_tile, fALFF_tile = alff_falff_np(block_dt, TR, low, high)
        except RuntimeError:
            ALFF_tile, fALFF_tile = alff_falff_np(block_dt, TR, low, high)

        # ReHo
        if _HAS_TORCH:
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    xdt_t = torch.from_numpy(block_dt).to(device, non_blocking=True)
                    reho_full = reho_tile_torch(xdt_t, neigh=neighbor, chunk_t=chunk_t).float().cpu().numpy()
                    del xdt_t; torch.cuda.empty_cache()
            else:
                xdt_t = torch.from_numpy(block_dt)
                reho_full = reho_tile_torch(xdt_t, neigh=neighbor, chunk_t=chunk_t).numpy()
                del xdt_t
        else:
            raise RuntimeError(f"PyTorch não encontrado: {TORCH_ERR}")

        # write outputs
        cx, cy, cz = crop
        (x0, x1), (y0, y1), (z0, z1) = (
            (core[0].start, core[0].stop),
            (core[1].start, core[1].stop),
            (core[2].start, core[2].stop),
        )
        exp = (x1-x0, y1-y0, z1-z0)
        assert ALFF_tile[cx,cy,cz].shape == exp and fALFF_tile[cx,cy,cz].shape == exp and reho_full[cx,cy,cz].shape == exp, \
            f"Crop != destino em {in_path.name}"
        ALFF_out[x0:x1, y0:y1, z0:z1]  = ALFF_tile[cx,cy,cz]
        fALFF_out[x0:x1, y0:y1, z0:z1] = fALFF_tile[cx,cy,cz]
        ReHo_out[x0:x1, y0:y1, z0:z1]  = reho_full[cx,cy,cz]

        del block_dt, ALFF_tile, fALFF_tile, reho_full
        outer.update(1)

    outer.close()
    save_indices_nifti(ALFF_out, fALFF_out, ReHo_out, img.affine, img.header, out_path)

# ---------------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="fMRI 4D -> (X,Y,Z,3) [ALFF,fALFF,ReHo] (compatível ds002748).")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--nii", type=str, help="Arquivo único .nii.gz (4D)")
    m.add_argument("--in_dir", type=str, help="Pasta BIDS de entrada")

    ap.add_argument("--out", type=str, help="Saída .nii.gz (modo arquivo único)")
    ap.add_argument("--out_dir", type=str, required=True, help="Pasta raiz de saída")
    ap.add_argument("--dataset_name", type=str, default=None, help="Nome da subpasta do dataset (default=basename de --in_dir)")

    ap.add_argument("--tr", type=float, default=None, help="TR em segundos (ou use --auto_tr)")
    ap.add_argument("--auto_tr", action="store_true", help="Detectar TR do sidecar JSON quando disponível")
    ap.add_argument("--low", default=0.01, type=float, help="Freq baixa (Hz) para ALFF/fALFF [0.01]")
    ap.add_argument("--high", default=0.10, type=float, help="Freq alta  (Hz) para ALFF/fALFF [0.10]")
    ap.add_argument("--neighbor", default=3, type=int, help="Vizinhança cúbica ReHo (recomendado 3)")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="Acelerador [auto]")
    ap.add_argument("--chunk_t", default=32, type=int, help="Tamanho do lote no tempo p/ ReHo [32]")
    ap.add_argument("--suffix", default="_3d", type=str, help="Sufixo do arquivo de saída [_3d]")
    ap.add_argument("--skip_if_exists", action="store_true", help="Pula se saída já existir")
    ap.add_argument("--patch", nargs=3, type=int, default=[64,64,64], help="Patch espacial X Y Z [64 64 64]")
    ap.add_argument("--max_split_mb", type=int, default=32, help="Hint ao alocador (GPU) [32]")

    args = ap.parse_args()

    if not _HAS_TORCH:
        raise RuntimeError(f"PyTorch não encontrado: {TORCH_ERR}")

    device = get_device(args.device)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # MODO ARQUIVO ÚNICO
    if args.nii:
        in_path = Path(args.nii)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        dataset_name = infer_dataset_name(in_path.parent.parent, args.dataset_name)  # heurística
        out_path = Path(args.out) if args.out else make_out_path(in_path, in_path.parent.parent, out_root, dataset_name, suffix=args.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        TR = load_tr_from_sidecar(in_path) if args.auto_tr else None
        TR = TR if (TR is not None) else args.tr
        if TR is None:
            raise ValueError("Defina --tr ou use --auto_tr (sidecar JSON BIDS).")

        process_one_file(in_path, out_path, TR, args.low, args.high, args.neighbor,
                         device, args.chunk_t, patch_xyz=args.patch, max_split_mb=args.max_split_mb)
        print(f"[OK] {out_path}")
        return

    # MODO PASTA (BIDS)
    in_root = Path(args.in_dir)
    if not in_root.exists():
        raise FileNotFoundError(in_root)

    dataset_name = infer_dataset_name(in_root, args.dataset_name)
    files = find_bold_in_func(in_root)

    if not files:
        print("Nenhum run de repouso encontrado (sub-*/func/*_bold.nii.gz).")
        return

    with tqdm(total=len(files), desc="Arquivos", unit="arq") as pbar:
        for f in files:
            TR = load_tr_from_sidecar(f) if args.auto_tr else None
            TR = TR if (TR is not None) else args.tr
            if TR is None:
                print(f"[ERRO] Sem TR para {f} (use --tr ou --auto_tr).")
                pbar.update(1); continue

            out_path = make_out_path(f, in_root, out_root, dataset_name, suffix=args.suffix)
            if args.skip_if_exists and out_path.exists():
                pbar.set_postfix_str(f"skip {f.name}")
                pbar.update(1)
                continue

            try:
                process_one_file(f, out_path, TR, args.low, args.high, args.neighbor,
                                 device, args.chunk_t, patch_xyz=args.patch, max_split_mb=args.max_split_mb)
                pbar.set_postfix_str(f"ok {f.name}")
            except Exception:
                pbar.write(f"[ERRO] {f}")
                pbar.write(traceback.format_exc())
                pbar.set_postfix_str(f"erro {f.name}")
            finally:
                pbar.update(1)

if __name__ == "__main__":
    main()
