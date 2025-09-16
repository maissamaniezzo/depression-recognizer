#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cria split estratificado (train/validation) em dataset BIDS-like:
- Estrutura esperada:
  dataset_3d/
    participants.tsv   (coluna obrigatória: 'group', valores: 'control' ou 'depr')
    sub-XX/
      func/
        *.nii.gz       (um ou mais arquivos por sujeito)

Saída:
  dataset_3d/train-test-validation/
    train/
      control/
        sub-XX__arquivo.nii.gz
      depr/
        sub-YY__arquivo.nii.gz
    validation/
      control/
      depr/
  + manifest.csv com (participant_id, group, split, src_path, dest_path)

Uso:
  python split_dataset.py --base_dir dataset_3d --val_ratio 0.2 --seed 42 [--link]

Obs:
- Por padrão copia arquivos; com --link cria symlinks.
- Faz split estratificado sem depender de scikit-learn.
"""

import argparse
import csv
import os
import random
import shutil
from glob import glob

def read_participants_tsv(tsv_path):
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"Não encontrei: {tsv_path}")

    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames_lower = [c.lower() for c in reader.fieldnames or []]
        # Detecta coluna com o ID do participante
        # BIDS costuma usar 'participant_id'
        id_candidates = ["participant_id", "participant", "subject_id", "subject", "id"]
        id_col = None
        for cand in id_candidates:
            if cand in fieldnames_lower:
                id_col = reader.fieldnames[fieldnames_lower.index(cand)]
                break
        if id_col is None:
            # fallback: tenta a primeira coluna
            id_col = reader.fieldnames[0]

        # Coluna de grupo (obrigatória segundo o enunciado)
        if "group" not in fieldnames_lower:
            raise ValueError("A coluna 'group' não foi encontrada em participants.tsv")
        group_col = reader.fieldnames[fieldnames_lower.index("group")]

        mapping = {}
        for row in reader:
            pid_raw = (row.get(id_col) or "").strip()
            if not pid_raw:
                continue
            # Normaliza 'sub-XX'
            if not pid_raw.startswith("sub-"):
                pid = f"sub-{pid_raw}"
            else:
                pid = pid_raw
            group = (row.get(group_col) or "").strip().lower()
            if group not in {"control", "depr"}:
                # ignora linhas com grupos inesperados
                continue
            mapping[pid] = group
        return mapping

def find_subject_files(base_dir, subject):
    """Encontra todos .nii.gz sob sub-XX/func (recursivo)."""
    func_dir = os.path.join(base_dir, subject, "func")
    if not os.path.isdir(func_dir):
        return []
    # recursivo para cobrir subníveis
    files = glob(os.path.join(func_dir, "**", "*.nii.gz"), recursive=True)
    return sorted(files)

def make_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_or_link(src, dst, use_link=False):
    if use_link:
        # remove destino se já existir (p/ recriar o link)
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
    parser = argparse.ArgumentParser(description="Split estratificado train/validation em dataset_3d.")
    parser.add_argument("--base_dir", required=True, help="Caminho para dataset_3d")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proporção para validation (padrão: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Semente randômica (padrão: 42)")
    parser.add_argument("--link", action="store_true", help="Criar links simbólicos em vez de copiar")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    tsv_path = os.path.join(base_dir, "participants.tsv")
    mapping = read_participants_tsv(tsv_path)

    # Filtra apenas sujeitos que existem como pastas sub-XX
    subjects_on_disk = [d for d in os.listdir(base_dir) if d.startswith("sub-") and os.path.isdir(os.path.join(base_dir, d))]
    subjects_on_disk_set = set(subjects_on_disk)

    # Agrupa por rótulo, somente sujeitos presentes de fato no disco
    subjects_by_group = {"control": [], "depr": []}
    missing_in_disk = []
    missing_in_tsv = []

    for pid, grp in mapping.items():
        if pid in subjects_on_disk_set:
            subjects_by_group[grp].append(pid)
        else:
            missing_in_disk.append(pid)

    # Também avisa se há sub-XX sem linha em participants.tsv
    for pid in subjects_on_disk:
        if pid not in mapping:
            missing_in_tsv.append(pid)

    # Split estratificado
    train_set, val_set = stratified_split(subjects_by_group, args.val_ratio, args.seed)

    # Pastas de saída
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
    # Função auxiliar para processar um conjunto
    def process_split(split_name, subjects_set):
        for pid in sorted(subjects_set):
            group = mapping.get(pid, None)
            if group not in {"control", "depr"}:
                continue
            src_files = find_subject_files(base_dir, pid)
            if not src_files:
                print(f"[AVISO] Nenhum .nii.gz encontrado para {pid} em {os.path.join(base_dir, pid, 'func')}")
                continue
            dest_dir = paths[(split_name, group)]
            for src in src_files:
                # Prefixa com o subject para evitar colisão de nomes
                dst_name = f"{pid}__{os.path.basename(src)}"
                dst = os.path.join(dest_dir, dst_name)
                copy_or_link(src, dst, use_link=args.link)
                manifest_rows.append({
                    "participant_id": pid,
                    "group": group,
                    "split": split_name,
                    "src_path": os.path.relpath(src, base_dir),
                    "dest_path": os.path.relpath(dst, base_dir),
                })

    process_split("train", train_set)
    process_split("validation", val_set)

    # Salva manifesto
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
            g = mapping.get(pid, None)
            if g in d:
                d[g] += 1
        return d

    train_counts = count_by_group(train_set)
    val_counts = count_by_group(val_set)

    print("\n=== RESUMO ===")
    print(f"Base: {base_dir}")
    print(f"Total no TSV: control={len(subjects_by_group['control']) + (1 if False else 0)} depr={len(subjects_by_group['depr'])}")
    print(f"Train: control={train_counts['control']} depr={train_counts['depr']}  (total={len(train_set)})")
    print(f"Validation: control={val_counts['control']} depr={val_counts['depr']}  (total={len(val_set)})")
    print(f"Manifesto salvo em: {manifest_path}")

    if missing_in_disk:
        print("\n[Aviso] Sujeitos presentes em participants.tsv mas sem pasta 'sub-XX' no disco:")
        for pid in sorted(missing_in_disk):
            print("  -", pid)

    if missing_in_tsv:
        print("\n[Aviso] Pastas 'sub-XX' sem linha correspondente em participants.tsv:")
        for pid in sorted(missing_in_tsv):
            print("  -", pid)

if __name__ == "__main__":
    main()
