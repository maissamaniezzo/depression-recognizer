"""Unified BIDS -> NIfTI 4D (X,Y,Z,3) converter.

Combines the previous ``ds002748_to_3d`` and ``ds005917_to_3d`` scripts
into a single entry point. Layout differences are handled by the
``--bids-layout`` flag (see :data:`depression_recognizer.preprocessing.io_utils.LAYOUTS`).

Usage
-----
Single file::

    python -m depression_recognizer.preprocessing.bids_to_3d \\
        --nii path/to/sub-01_task-rest_bold.nii.gz \\
        --out path/to/sub-01_task-rest_bold_3d.nii.gz \\
        --tr 2.5 --device auto

Whole dataset (BIDS folder)::

    python -m depression_recognizer.preprocessing.bids_to_3d \\
        --in_dir data/ds002748 \\
        --out_dir data/dataset_3d \\
        --bids-layout ds002748 \\
        --auto-tr
"""
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from .. import config
from ..utils.optional_imports import optional_torch
from .indices import (
    alff_falff_np,
    alff_falff_torch,
    detrend_linear_np,
    reho_tile_torch,
)
from .io_utils import (
    LAYOUTS,
    find_bold_files,
    infer_dataset_name,
    load_tr_from_sidecar,
    make_out_path,
)


# ----------------------------- Helpers -------------------------------------

def get_device(dev_arg: str):
    torch, _ = optional_torch()
    if dev_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu or auto.")
        print("GPUs visíveis:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
        print("Device atual:", torch.cuda.current_device())
        return torch.device("cuda")
    if dev_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def as_tuple3(x) -> Tuple[int, int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return tuple(int(v) for v in x)  # type: ignore[return-value]
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


def save_indices_nifti(
    alff: np.ndarray,
    falff: np.ndarray,
    reho: np.ndarray,
    affine: np.ndarray,
    ref_header: nib.nifti1.Nifti1Header,
    out_path: Path,
) -> None:
    vol = np.stack([alff, falff, reho], axis=-1).astype(np.float32)
    hdr = ref_header.copy()
    hdr.set_data_dtype(np.float32)
    try:
        hdr.set_intent("vector", (), "")
    except Exception:
        pass
    nib.save(nib.Nifti1Image(vol, affine=affine, header=hdr), str(out_path))


# ----------------------------- Per-file worker ------------------------------

def process_one_file(
    in_path: Path,
    out_path: Path,
    TR: float,
    low: float,
    high: float,
    neighbor: int,
    device,
    chunk_t: int,
    patch_xyz=(64, 64, 64),
    max_split_mb: int = 32,
) -> None:
    torch, _ = optional_torch()

    img = nib.load(str(in_path), mmap=True)
    if img.ndim != 4:
        raise ValueError(f"{in_path}: esperado 4D (X,Y,Z,T); recebi {img.shape}")
    data = img.dataobj
    X, Y, Z, T = img.shape

    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"max_split_size_mb:{int(max_split_mb)},expandable_segments:True"
        )
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False

    ALFF_out = np.zeros((X, Y, Z), dtype=np.float32)
    fALFF_out = np.zeros((X, Y, Z), dtype=np.float32)
    ReHo_out = np.zeros((X, Y, Z), dtype=np.float32)

    r = neighbor // 2
    patch = as_tuple3(patch_xyz)
    halo = (r, r, r)

    outer = tqdm(
        total=((X + patch[0] - 1) // patch[0])
        * ((Y + patch[1] - 1) // patch[1])
        * ((Z + patch[2] - 1) // patch[2]),
        desc=f"Tiles {in_path.name}",
        unit="tile",
        leave=False,
    )

    for core, pad, crop in iter_tiles_xyz(X, Y, Z, patch, halo):
        block_np = np.asarray(data[pad[0], pad[1], pad[2], :], dtype=np.float32)
        block_dt = detrend_linear_np(block_np)
        del block_np

        # ---- ALFF / fALFF ----
        try:
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    ALFF_tile, fALFF_tile = alff_falff_torch(
                        block_dt, TR=TR, low=low, high=high, device=device
                    )
                    torch.cuda.empty_cache()
            else:
                ALFF_tile, fALFF_tile = alff_falff_np(block_dt, TR, low, high)
        except RuntimeError:
            ALFF_tile, fALFF_tile = alff_falff_np(block_dt, TR, low, high)

        # ---- ReHo ----
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                xdt_t = torch.from_numpy(block_dt).to(device, non_blocking=True)
                reho_full = reho_tile_torch(xdt_t, neigh=neighbor, chunk_t=chunk_t)
                reho_full = reho_full.float().cpu().numpy()
                del xdt_t
                torch.cuda.empty_cache()
        else:
            xdt_t = torch.from_numpy(block_dt)
            reho_full = reho_tile_torch(xdt_t, neigh=neighbor, chunk_t=chunk_t).numpy()
            del xdt_t

        # ---- write tiles into outputs ----
        cx, cy, cz = crop
        (x0, x1), (y0, y1), (z0, z1) = (
            (core[0].start, core[0].stop),
            (core[1].start, core[1].stop),
            (core[2].start, core[2].stop),
        )
        exp = (x1 - x0, y1 - y0, z1 - z0)
        assert (
            ALFF_tile[cx, cy, cz].shape == exp
            and fALFF_tile[cx, cy, cz].shape == exp
            and reho_full[cx, cy, cz].shape == exp
        ), f"Crop != destino em {in_path.name}"

        ALFF_out[x0:x1, y0:y1, z0:z1]  = ALFF_tile[cx, cy, cz]
        fALFF_out[x0:x1, y0:y1, z0:z1] = fALFF_tile[cx, cy, cz]
        ReHo_out[x0:x1, y0:y1, z0:z1]  = reho_full[cx, cy, cz]

        del block_dt, ALFF_tile, fALFF_tile, reho_full
        outer.update(1)

    outer.close()
    save_indices_nifti(ALFF_out, fALFF_out, ReHo_out, img.affine, img.header, out_path)


# ----------------------------- CLI ----------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "fMRI 4D (.nii.gz) -> NIfTI 4D (X,Y,Z,3) [ALFF, fALFF, ReHo]."
        ),
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nii", type=str, help="Arquivo único .nii.gz (4D)")
    mode.add_argument("--in-dir", type=str, dest="in_dir", help="Pasta BIDS de entrada")

    p.add_argument("--out", type=str, help="Saída .nii.gz (modo arquivo único)")
    p.add_argument(
        "--out-dir", type=str, dest="out_dir", required=True, help="Pasta raiz de saída"
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Subpasta do dataset (default = basename de --in-dir)",
    )
    p.add_argument(
        "--bids-layout",
        type=str,
        default="ds002748",
        choices=sorted(LAYOUTS.keys()),
        help="Layout BIDS a usar (default ds002748).",
    )

    p.add_argument("--tr", type=float, default=config.DEFAULT_TR, help="TR em segundos")
    p.add_argument(
        "--auto-tr",
        action="store_true",
        help="Detectar TR do sidecar JSON quando disponível",
    )
    p.add_argument("--low", type=float, default=config.DEFAULT_LOW, help="Freq baixa (Hz) p/ ALFF/fALFF")
    p.add_argument("--high", type=float, default=config.DEFAULT_HIGH, help="Freq alta  (Hz) p/ ALFF/fALFF")
    p.add_argument("--neighbor", type=int, default=config.DEFAULT_NEIGHBOR, help="Vizinhança cúbica ReHo")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Acelerador",
    )
    p.add_argument("--chunk-t", type=int, default=config.DEFAULT_CHUNK_T, dest="chunk_t", help="Chunk no tempo para ReHo")
    p.add_argument("--suffix", default="_3d", help="Sufixo do arquivo de saída")
    p.add_argument("--skip-if-exists", action="store_true", dest="skip_if_exists", help="Pula se saída já existir")
    p.add_argument(
        "--patch",
        nargs=3,
        type=int,
        default=list(config.DEFAULT_PATCH),
        help="Patch espacial X Y Z",
    )
    p.add_argument("--max-split-mb", type=int, default=32, dest="max_split_mb", help="Hint ao alocador (GPU)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    torch, _ = optional_torch()
    device = get_device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- single-file mode ----
    if args.nii:
        in_path = Path(args.nii)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        dataset_name = infer_dataset_name(in_path.parent.parent, args.dataset_name)
        out_path = (
            Path(args.out)
            if args.out
            else make_out_path(
                in_path, in_path.parent.parent, out_root, dataset_name, args.bids_layout,
                suffix=args.suffix,
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        TR = load_tr_from_sidecar(in_path) if args.auto_tr else None
        TR = TR if TR is not None else args.tr
        if TR is None:
            raise ValueError("Defina --tr ou use --auto-tr (sidecar JSON BIDS).")

        process_one_file(
            in_path, out_path, TR, args.low, args.high, args.neighbor,
            device, args.chunk_t, patch_xyz=args.patch, max_split_mb=args.max_split_mb,
        )
        print(f"[OK] {out_path}")
        return

    # ---- folder mode ----
    in_root = Path(args.in_dir)
    if not in_root.exists():
        raise FileNotFoundError(in_root)

    dataset_name = infer_dataset_name(in_root, args.dataset_name)
    files = find_bold_files(in_root, args.bids_layout)
    if not files:
        print(f"Nenhum arquivo BOLD encontrado para o layout {args.bids_layout!r} em {in_root}.")
        return

    with tqdm(total=len(files), desc="Arquivos", unit="arq") as pbar:
        for f in files:
            TR = load_tr_from_sidecar(f) if args.auto_tr else None
            TR = TR if TR is not None else args.tr
            if TR is None:
                print(f"[ERRO] Sem TR para {f} (use --tr ou --auto-tr).")
                pbar.update(1)
                continue

            out_path = make_out_path(
                f, in_root, out_root, dataset_name, args.bids_layout, suffix=args.suffix,
            )
            if args.skip_if_exists and out_path.exists():
                pbar.set_postfix_str(f"skip {f.name}")
                pbar.update(1)
                continue

            try:
                process_one_file(
                    f, out_path, TR, args.low, args.high, args.neighbor,
                    device, args.chunk_t, patch_xyz=args.patch, max_split_mb=args.max_split_mb,
                )
                pbar.set_postfix_str(f"ok {f.name}")
            except Exception:
                pbar.write(f"[ERRO] {f}")
                pbar.write(traceback.format_exc())
                pbar.set_postfix_str(f"erro {f.name}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()
