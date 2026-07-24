"""Stratified train/validation split of ``dataset_3d``.

Reads ``participants.tsv`` from the permitted dataset roots (only
``ds002748``, ``ds005917`` and the ``ds005817`` alias) and produces
``train-test-validation/manifest.csv`` plus symlinked/copied NIfTI files
organized as ``{split}/{control,depr}/``.

Usage
-----

::

    python -m depression_recognizer.data.split \\
        --base-dir data/dataset_3d \\
        --val-ratio 0.2 --link
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from glob import glob

# Only these roots are considered (ds005817 is treated as a typo/alias of ds005917)
HARD_ALLOWED_DATASETS = ["ds002748", "ds005917", "ds005817"]

# Sub-trees where NIfTI files live for each dataset.
DATASET_PATTERNS: dict[str, tuple[str, ...]] = {
    "ds002748": ("func",),               # dataset_3d/ds002748/sub-XX/func/**/*.nii(.gz)
    "ds005917": ("ses-b0", "func"),      # dataset_3d/ds005917/sub-XX/ses-b0/func/**/*.nii(.gz)
}


# ----------------------------- I/O -----------------------------------------

def _read_participants_tsv(tsv_path: str) -> dict[str, str]:
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames_lower = [c.lower() for c in (reader.fieldnames or [])]
        id_candidates = ["participant_id", "participant", "subject_id", "subject", "id"]
        try:
            id_col = next(
                reader.fieldnames[fieldnames_lower.index(c)]
                for c in id_candidates
                if c in fieldnames_lower
            )
        except StopIteration:
            id_col = reader.fieldnames[0]
        if "group" not in fieldnames_lower:
            raise ValueError(f"A coluna 'group' não foi encontrada em {tsv_path}")
        group_col = reader.fieldnames[fieldnames_lower.index("group")]

        mapping: dict[str, str] = {}
        for row in reader:
            pid_raw = (row.get(id_col) or "").strip()
            if not pid_raw:
                continue
            pid = pid_raw if pid_raw.startswith("sub-") else f"sub-{pid_raw}"
            group = (row.get(group_col) or "").strip().lower()
            if group in {"control", "depr"}:
                mapping[pid] = group
        return mapping


def read_all_participants(
    base_dir: str, datasets: list[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]], list]:
    """Read participants.tsv only from the allowed dataset roots."""
    mapping: dict[str, str] = {}
    per_dataset_mapping: dict[str, dict[str, str]] = {}
    seen: dict[str, list] = {}
    conflicts: list = []

    for ds in datasets:
        ds_root = os.path.join(base_dir, ds)
        tsv_path = os.path.join(ds_root, "participants.tsv")
        if not os.path.isdir(ds_root) or not os.path.isfile(tsv_path):
            continue
        try:
            ds_map = _read_participants_tsv(tsv_path)
        except Exception as exc:
            raise RuntimeError(f"Erro ao ler {tsv_path}: {exc}") from exc
        per_dataset_mapping[ds] = ds_map
        for pid, grp in ds_map.items():
            seen.setdefault(pid, []).append((ds, grp))

    for pid, lst in seen.items():
        groups = {g for _, g in lst}
        if len(groups) == 1:
            mapping[pid] = next(iter(groups))
        else:
            first_ds = sorted(lst, key=lambda x: x[0])[0]
            mapping[pid] = first_ds[1]
            conflicts.append((pid, groups, [ds for ds, _ in lst]))
    return mapping, per_dataset_mapping, conflicts


def find_subject_dirs(base_dir: str, datasets: list[str]) -> set[str]:
    """Collect all ``sub-XX`` folders within the allowed dataset roots."""
    subjects: set[str] = set()
    for ds in datasets:
        root = os.path.join(base_dir, ds)
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            if entry.startswith("sub-") and os.path.isdir(os.path.join(root, entry)):
                subjects.add(entry)
    return subjects


def find_subject_files(
    base_dir: str, subject: str, datasets: list[str], debug: bool = False
) -> list[str]:
    """Find NIfTI files for a subject, only within the allowed roots."""
    exts = ("*.nii.gz", "*.nii")
    files: set[str] = set()
    checked: list[str] = []
    for ds in datasets:
        tail = DATASET_PATTERNS.get(ds)
        if not tail:
            continue
        candidate_dir = os.path.join(base_dir, ds, subject, *tail)
        checked.append(candidate_dir)
        if os.path.isdir(candidate_dir):
            for ext in exts:
                files.update(glob(os.path.join(candidate_dir, "**", ext), recursive=True))
    if debug:
        print(f"[DEBUG] {subject}: verificados {len(checked)} diretórios")
        for p in checked:
            print("   -", p)
        print(f"[DEBUG] {subject}: encontrados {len(files)} NIfTI")
    return sorted(files)


# ----------------------------- Split ---------------------------------------

def _copy_or_link(src: str, dst: str, use_link: bool) -> None:
    if use_link:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def stratified_split(
    subjects_by_group: dict[str, list[str]], val_ratio: float, seed: int = 42
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    train_set: set[str] = set()
    val_set: set[str] = set()
    for subs in subjects_by_group.values():
        subs = list(subs)
        rng.shuffle(subs)
        n = len(subs)
        val_n = max(1, round(n * val_ratio)) if n > 0 else 0
        val_set.update(subs[:val_n])
        train_set.update(subs[val_n:])
    return train_set, val_set


# ----------------------------- CLI -----------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split estratificado em dataset_3d usando apenas ds002748 e ds005917."
    )
    p.add_argument("--base-dir", required=True, dest="base_dir", help="Caminho para dataset_3d")
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--link", action="store_true", help="Criar symlinks em vez de copiar")
    p.add_argument("--debug", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    base_dir = os.path.abspath(args.base_dir)

    datasets = [ds for ds in HARD_ALLOWED_DATASETS if os.path.isdir(os.path.join(base_dir, ds))]
    normalized: list[str] = []
    for ds in datasets:
        if ds == "ds005817":
            normalized.append("ds005917" if os.path.isdir(os.path.join(base_dir, "ds005917")) else "ds005817")
        else:
            normalized.append(ds)
    datasets = list(dict.fromkeys(normalized))

    if args.debug:
        print("[DEBUG] Raízes consideradas:", datasets)
    if not datasets:
        raise SystemExit("Nenhuma das raízes permitidas (ds002748, ds005917) foi encontrada em base_dir.")

    mapping, _, conflicts = read_all_participants(base_dir, datasets)
    subjects_on_disk = find_subject_dirs(base_dir, datasets)
    subjects_on_disk_set = set(subjects_on_disk)

    subjects_by_group: dict[str, list[str]] = {"control": [], "depr": []}
    missing_in_disk: list[str] = []
    missing_in_tsv: list[str] = []

    for pid, grp in mapping.items():
        if pid in subjects_on_disk_set:
            subjects_by_group[grp].append(pid)
        else:
            missing_in_disk.append(pid)
    for pid in subjects_on_disk:
        if pid not in mapping:
            missing_in_tsv.append(pid)

    train_set, val_set = stratified_split(subjects_by_group, args.val_ratio, args.seed)

    out_root = os.path.join(base_dir, "train-test-validation")
    paths = {
        ("train", "control"):       os.path.join(out_root, "train", "control"),
        ("train", "depr"):          os.path.join(out_root, "train", "depr"),
        ("validation", "control"):  os.path.join(out_root, "validation", "control"),
        ("validation", "depr"):     os.path.join(out_root, "validation", "depr"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    manifest_rows: list[dict] = []

    def process_split(split_name: str, subjects_set: set[str]) -> None:
        for pid in sorted(subjects_set):
            group = mapping.get(pid)
            if group not in {"control", "depr"}:
                continue
            src_files = find_subject_files(base_dir, pid, datasets, debug=args.debug)
            if not src_files:
                examples = [os.path.join(base_dir, ds, pid, *DATASET_PATTERNS.get(ds, ())) for ds in datasets]
                print(f"[AVISO] Nenhum .nii(.gz) encontrado para {pid}. Verificados:\n  - " + "\n  - ".join(examples))
                continue
            dest_dir = paths[(split_name, group)]
            for src in src_files:
                dst_name = f"{pid}__{os.path.basename(src)}"
                dst = os.path.join(dest_dir, dst_name)
                _copy_or_link(src, dst, use_link=args.link)
            for src in src_files:
                dest = os.path.join(paths[(split_name, group)], f"{pid}__{os.path.basename(src)}")
                manifest_rows.append({
                    "participant_id": pid,
                    "group": group,
                    "split": split_name,
                    "src_path": os.path.relpath(src, base_dir),
                    "dest_path": os.path.relpath(dest, base_dir),
                })

    process_split("train", train_set)
    process_split("validation", val_set)

    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["participant_id", "group", "split", "src_path", "dest_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    def _count_by_group(subjects_set: set[str]) -> dict[str, int]:
        d = {"control": 0, "depr": 0}
        for pid in subjects_set:
            g = mapping.get(pid)
            if g in d:
                d[g] += 1
        return d

    train_counts = _count_by_group(train_set)
    val_counts = _count_by_group(val_set)

    print("\n=== RESUMO ===")
    print(f"Base: {base_dir}")
    print(f"Datasets usados: {', '.join(datasets)}")
    print(f"Total no TSV: control={len(subjects_by_group['control'])} depr={len(subjects_by_group['depr'])}")
    print(f"Train: control={train_counts['control']} depr={train_counts['depr']}  (total={len(train_set)})")
    print(f"Validation: control={val_counts['control']} depr={val_counts['depr']}  (total={len(val_set)})")
    print(f"Manifesto salvo em: {manifest_path}")

    if conflicts:
        print("\n[Aviso] Conflitos de grupo:")
        for pid, groups, ds_list in conflicts:
            print(f"  - {pid}: grupos {sorted(groups)} em datasets {sorted(ds_list)}")
    if missing_in_disk:
        print("\n[Aviso] Sujeitos no TSV sem pasta correspondente:")
        for pid in sorted(missing_in_disk):
            print("  -", pid)
    if missing_in_tsv:
        print("\n[Aviso] Pastas sub-XX sem linha no participants.tsv:")
        for pid in sorted(missing_in_tsv):
            print("  -", pid)


if __name__ == "__main__":
    main()
