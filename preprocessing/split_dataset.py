#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split estratificado (train/validation) em dataset_3d, lendo apenas ds002748 e ds005917
(caso exista uma pasta 'ds005817', é tratada como alias de ds005917).

Estrutura:
  dataset_3d/
    ds002748/
      participants.tsv   ('group' ∈ {'control','depr'})
      sub-XX/func/*.nii[.gz]
    ds005917/ (ou ds005817/)
      participants.tsv   ('group' ∈ {'control','depr'})
      sub-YY/ses-b0/func/*.nii[.gz]

Saída:
  dataset_3d/train-test-validation/...
  + manifest.csv: (participant_id, group, split, src_path, dest_path)

  Usage:
    python preprocessing/split_dataset.py --base_dir ./dataset_3d
"""

import argparse
import csv
import os
import random
import shutil
from glob import glob

# Somente estas raízes são consideradas
HARD_ALLOWED_DATASETS = ["ds002748", "ds005917", "ds005817"]  # 817 tratado como alias

# Subcaminhos onde ficam os NIfTI por dataset
DATASET_PATTERNS = {
    "ds002748": ("func",),               # dataset_3d/ds002748/sub-XX/func/**/*.nii(.gz)
    "ds005917": ("ses-b0", "func"),      # dataset_3d/ds005917/sub-XX/ses-b0/func/**/*.nii(.gz)
}

def read_participants_tsv(tsv_path):
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames_lower = [c.lower() for c in (reader.fieldnames or [])]
        id_candidates = ["participant_id", "participant", "subject_id", "subject", "id"]
        id_col = next((reader.fieldnames[fieldnames_lower.index(c)]
                       for c in id_candidates if c in fieldnames_lower), reader.fieldnames[0])
        if "group" not in fieldnames_lower:
            raise ValueError(f"A coluna 'group' não foi encontrada em {tsv_path}")
        group_col = reader.fieldnames[fieldnames_lower.index("group")]
        mapping = {}
        for row in reader:
            pid_raw = (row.get(id_col) or "").strip()
            if not pid_raw:
                continue
            pid = pid_raw if pid_raw.startswith("sub-") else f"sub-{pid_raw}"
            group = (row.get(group_col) or "").strip().lower()
            if group in {"control", "depr"}:
                mapping[pid] = group
        return mapping

def read_all_participants(base_dir, datasets):
    """
    Lê participants.tsv apenas das raízes permitidas.
    Retorna: mapping_global, per_dataset_mapping, conflicts
    """
    mapping = {}
    per_dataset_mapping = {}
    seen = {}
    conflicts = []

    for ds in datasets:
        ds_root = os.path.join(base_dir, ds)
        tsv_path = os.path.join(ds_root, "participants.tsv")
        if not os.path.isdir(ds_root) or not os.path.isfile(tsv_path):
            continue
        try:
            ds_map = read_participants_tsv(tsv_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao ler {tsv_path}: {e}")
        per_dataset_mapping[ds] = ds_map
        for pid, grp in ds_map.items():
            seen.setdefault(pid, []).append((ds, grp))

    for pid, lst in seen.items():
        groups = {g for _, g in lst}
        if len(groups) == 1:
            mapping[pid] = next(iter(groups))
        else:
            # Conflito: mantém o primeiro por ordem do nome do dataset (determinístico)
            first_ds = sorted(lst, key=lambda x: x[0])[0]
            mapping[pid] = first_ds[1]
            conflicts.append((pid, groups, [ds for ds, _ in lst]))
    return mapping, per_dataset_mapping, conflicts

def find_subject_dirs(base_dir, datasets):
    """Coleta todos os sub-XX existentes apenas dentro das raízes permitidas."""
    subjects = set()
    for ds in datasets:
        root = os.path.join(base_dir, ds)
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            if entry.startswith("sub-") and os.path.isdir(os.path.join(root, entry)):
                subjects.add(entry)
    return subjects

def find_subject_files(base_dir, subject, datasets, debug=False):
    """
    Busca .nii.gz e .nii apenas nas raízes permitidas e nos subcaminhos definidos em DATASET_PATTERNS.
    NÃO percorre nada fora de ds002748/ds005917(/ds005817).
    """
    exts = ("*.nii.gz", "*.nii")
    files = set()
    checked = []
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
        print(f"[DEBUG] {subject}: verificados {len(checked)} diretórios:")
        for p in checked:
            print("   -", p)
        print(f"[DEBUG] {subject}: encontrados {len(files)} NIfTI")
    return sorted(files)

def make_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_or_link(src, dst, use_link=False):
    if use_link:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)

def stratified_split(subjects_by_group, val_ratio, seed=42):
    rng = random.Random(seed)
    train_set, val_set = set(), set()
    for group, subs in subjects_by_group.items():
        subs = list(subs)
        rng.shuffle(subs)
        n = len(subs)
        val_n = max(1, round(n * val_ratio)) if n > 0 else 0
        val_part = set(subs[:val_n])
        train_part = set(subs[val_n:])
        val_set |= val_part
        train_set |= train_part
    return train_set, val_set

def main():
    parser = argparse.ArgumentParser(
        description="Split estratificado em dataset_3d usando apenas ds002748 e ds005917."
    )
    parser.add_argument("--base_dir", required=True, help="Caminho para dataset_3d")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proporção para validation (padrão: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Semente randômica (padrão: 42)")
    parser.add_argument("--link", action="store_true", help="Criar symlinks em vez de copiar")
    parser.add_argument("--debug", action="store_true", help="Imprime caminhos verificados")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)

    # Considera apenas raízes permitidas que existam de fato
    datasets = [ds for ds in HARD_ALLOWED_DATASETS if os.path.isdir(os.path.join(base_dir, ds))]
    # Alias: se existir ds005817 mas não ds005917, tratamos como 917 para consistência no relatório
    normalized_datasets = []
    for ds in datasets:
        if ds == "ds005817":
            normalized_datasets.append("ds005917" if os.path.isdir(os.path.join(base_dir, "ds005917")) else "ds005817")
        else:
            normalized_datasets.append(ds)
    datasets = list(dict.fromkeys(normalized_datasets))  # remove duplicatas mantendo ordem

    if args.debug:
        print("[DEBUG] Raízes consideradas:", datasets)

    if not datasets:
        raise SystemExit("Nenhuma das raízes permitidas (ds002748, ds005917) foi encontrada em base_dir.")

    # Lê participants.tsv apenas dessas raízes
    mapping, per_dataset_mapping, conflicts = read_all_participants(base_dir, datasets)

    # Sujeitos presentes no disco (apenas nessas raízes)
    subjects_on_disk = find_subject_dirs(base_dir, datasets)
    subjects_on_disk_set = set(subjects_on_disk)

    # Agrupa por rótulo apenas para quem existe no disco
    subjects_by_group = {"control": [], "depr": []}
    missing_in_disk, missing_in_tsv = [], []

    for pid, grp in mapping.items():
        if pid in subjects_on_disk_set:
            subjects_by_group[grp].append(pid)
        else:
            missing_in_disk.append(pid)

    for pid in subjects_on_disk:
        if pid not in mapping:
            missing_in_tsv.append(pid)

    # Split
    train_set, val_set = stratified_split(subjects_by_group, args.val_ratio, args.seed)

    # Saída no mesmo nível das raízes
    out_root = os.path.join(base_dir, "train-test-validation")
    paths = {
        ("train", "control"): os.path.join(out_root, "train", "control"),
        ("train", "depr"):    os.path.join(out_root, "train", "depr"),
        ("validation", "control"): os.path.join(out_root, "validation", "control"),
        ("validation", "depr"):    os.path.join(out_root, "validation", "depr"),
    }
    for p in paths.values():
        make_dir(p)

    manifest_rows = []

    def process_split(split_name, subjects_set):
        for pid in sorted(subjects_set):
            group = mapping.get(pid)
            if group not in {"control", "depr"}:
                continue
            src_files = find_subject_files(base_dir, pid, datasets, debug=args.debug)
            if not src_files:
                # Mostra exemplos de caminhos esperados (somente dentro das raízes permitidas)
                examples = []
                for ds in datasets:
                    tail = DATASET_PATTERNS.get(ds, ())
                    examples.append(os.path.join(base_dir, ds, pid, *tail))
                print(f"[AVISO] Nenhum .nii(.gz) encontrado para {pid}. Verificados:\n  - " + "\n  - ".join(examples))
                continue
            dest_dir = paths[(split_name, group)]
            for src in src_files:
                dst_name = f"{pid}__{os.path.basename(src)}"
                dst = os.path.join(dest_dir, dst_name)
                copy_or_link(src, dst, use_link=args.link)
            # 1 linha por arquivo no manifest (mantém rastreabilidade)
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

    # Manifest
    make_dir(out_root)
    manifest_path = os.path.join(out_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "group", "split", "src_path", "dest_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Resumo
    def count_by_group(subjects_set):
        d = {"control": 0, "depr": 0}
        for pid in subjects_set:
            g = mapping.get(pid)
            if g in d:
                d[g] += 1
        return d

    train_counts = count_by_group(train_set)
    val_counts = count_by_group(val_set)
    total_control = len(subjects_by_group["control"])
    total_depr = len(subjects_by_group["depr"])

    print("\n=== RESUMO ===")
    print(f"Base: {base_dir}")
    print(f"Datasets usados: {', '.join(datasets)}")
    print(f"Total no TSV (união nas raízes permitidas): control={total_control} depr={total_depr}")
    print(f"Train: control={train_counts['control']} depr={train_counts['depr']}  (total={len(train_set)})")
    print(f"Validation: control={val_counts['control']} depr={val_counts['depr']}  (total={len(val_set)})")
    print(f"Manifesto salvo em: {manifest_path}")

    if conflicts:
        print("\n[Aviso] Conflitos de grupo para o mesmo participant_id entre datasets:")
        for pid, groups, ds_list in conflicts:
            print(f"  - {pid}: grupos {sorted(groups)} em datasets {sorted(ds_list)} "
                  f"(mantido o primeiro por ordem do nome do dataset)")

    if missing_in_disk:
        print("\n[Aviso] Sujeitos presentes em participants.tsv (das raízes permitidas) mas sem pasta correspondente:")
        for pid in sorted(missing_in_disk):
            print("  -", pid)

    if missing_in_tsv:
        print("\n[Aviso] Pastas 'sub-XX' no disco (dentro das raízes permitidas) sem linha em nenhum participants.tsv:")
        for pid in sorted(missing_in_tsv):
            print("  -", pid)

if __name__ == "__main__":
    main()
