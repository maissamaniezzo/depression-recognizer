# depression-recognizer

> Fine-tuning **Med3D ResNet18** (Tencent MedicalNet) on resting-state fMRI
> to classify **depression vs control**.
>
> The pipeline computes three voxel-wise indices — **ALFF**, **fALFF** and
> **ReHo** — for each subject, stacks them as channels of a 3D NIfTI
> volume, and trains a 3D ResNet to predict the clinical label.

The project started as a TFG (UNIFEI) study; this repository is the
**re-organised, package-ified version** of the original code.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Quick start (Make)](#quick-start-make)
5. [Manual step-by-step](#manual-step-by-step)
6. [Datasets](#datasets)
7. [Pretrained weights](#pretrained-weights)
8. [Configuration & environment variables](#configuration--environment-variables)
9. [Output artifacts](#output-artifacts)
10. [Development](#development)
11. [References](#references)

---

## Pipeline overview

```
fMRI 4D (.nii.gz) ──► [preprocessing] ──► (X,Y,Z,3) NIfTI  [ALFF, fALFF, ReHo]
                                                       │
                                                       ▼
                                  [data.split]  ──►  train/validation folders + manifest.csv
                                                       │
                                                       ▼
                              [modeling.train]  ──►  Med3D ResNet18 fine-tune ──► checkpoint .pt
                                                       │
                                                       ▼
                              [modeling.infer]  ──►  metrics.json + predictions.csv
```

The classification head replaces MedicalNet's segmentation head
(`conv_seg`) with `AdaptiveAvgPool3d → Flatten → Dropout → Linear`.

---

## Repository layout

```
depression-recognizer/
├── Makefile                     # one-shot tasks (install, preprocess, train, ...)
├── pyproject.toml               # installable package (PEP 621)
├── README.md
├── .gitignore
└── src/depression_recognizer/
    ├── __init__.py
    ├── config.py                # paths + defaults (env-var overridable)
    ├── utils/
    │   ├── paths.py             # NIfTI / path helpers
    │   └── optional_imports.py  # deferred torch import
    ├── preprocessing/
    │   ├── bids_to_3d.py        # CLI: fMRI 4D → (X,Y,Z,3) NIfTI
    │   ├── indices.py           # detrend, ALFF/fALFF, ReHo
    │   └── io_utils.py          # BIDS discovery, sidecar JSON, output paths
    ├── data/
    │   ├── dataset.py           # PyTorch Dataset + manifest reader
    │   └── split.py             # stratified train/validation split (CLI)
    └── modeling/
        ├── model.py             # Med3D ResNet18 builder + freeze helpers
        ├── train.py             # training CLI
        └── infer.py             # evaluation CLI
```

The previous folders (`preprocessing/`, `model/`, `model-verification/`,
`study_code/`) and the two duplicated preprocessing scripts
(`ds002748_to_3d.py`, `ds005917_to_3d.py`) were replaced by this
package.

---

## Installation

Tested on **Python 3.10+**. CUDA is optional but recommended
(Med3D ResNet18 inference benefits massively from GPU).

```bash
# 1. Clone
git clone https://github.com/<your-org>/depression-recognizer.git
cd depression-recognizer

# 2. Create a venv and install the package (CUDA build by default)
make install

# Or, for a CPU-only install
make install-cpu
```

The package is `src/`-layout and exposes four console scripts:

| script                         | module                                              |
| ------------------------------ | --------------------------------------------------- |
| `dep-recognizer-preprocess`    | `depression_recognizer.preprocessing.bids_to_3d`    |
| `dep-recognizer-split`         | `depression_recognizer.data.split`                  |
| `dep-recognizer-train`         | `depression_recognizer.modeling.train`              |
| `dep-recognizer-infer`         | `depression_recognizer.modeling.infer`              |

You can use either the console script or `python -m ...`.

---

## Quick start (Make)

```bash
# (one-off) get the raw OpenNeuro datasets and the MedicalNet backbone
make download-ds002748 DS_DS002748=./data/ds002748
make download-ds005917 DS_DS005917=./data/ds005917
make download-medicalnet

# (one-off) pretrained Med3D ResNet18 weights
make download-weights

# Preprocess both datasets into data/dataset_3d/
make preprocess-all DEVICE=cuda

# Stratified train/validation split
make split

# Train (CPU-friendly defaults; tune on your hardware)
make train DEVICE=cuda

# Evaluate on the validation split
make infer SPLIT=validation
```

`make help` lists all available targets.

---

## Manual step-by-step

If you prefer to drive the pipeline without `make`:

```bash
# 1. Preprocess OpenNeuro ds002748 (no session folder)
python -m depression_recognizer.preprocessing.bids_to_3d \
    --in-dir  data/ds002748 \
    --out-dir data/dataset_3d \
    --bids-layout ds002748 \
    --dataset-name ds002748 \
    --tr 2.5 --neighbor 3 --auto-tr --device auto

# 2. Preprocess OpenNeuro ds005917 (baseline session b0, rest task)
python -m depression_recognizer.preprocessing.bids_to_3d \
    --in-dir  data/ds005917 \
    --out-dir data/dataset_3d \
    --bids-layout ds005917 \
    --dataset-name ds005917 \
    --tr 2.5 --neighbor 3 --auto-tr --device auto

# 3. Stratified train/validation split (writes manifest.csv)
python -m depression_recognizer.data.split \
    --base-dir data/dataset_3d --val-ratio 0.2 --seed 42 --link

# 4. Train
python -m depression_recognizer.modeling.train \
    --data-root data/dataset_3d/train-test-validation \
    --warmup-epochs 100 --main-epochs 2000

# 5. Evaluate
python -m depression_recognizer.modeling.infer \
    --data-root data/dataset_3d/train-test-validation \
    --checkpoint model/checkpoints/med3d_resnet18.pt \
    --split validation
```

A full preprocessing pass on a single subject takes a few minutes on
CPU and seconds on GPU.

---

## Datasets

| Dataset      | Subjects | Layout                                                  |
| ------------ | -------- | ------------------------------------------------------- |
| `ds002748`   | 72       | `sub-*/func/*_bold.nii.gz` (no session)                 |
| `ds005917`   | 130+     | `sub-*/ses-b0/func/*task-rest_bold.nii.gz`              |
| `ds005817`   | alias    | Treated as `ds005917` (typo / older naming)             |

Only the two well-known controls/depressed cohorts are used. Any other
folders under `data/dataset_3d/` are ignored by the splitter.

Each `participants.tsv` must contain a `group` column with values
`control` or `depr` (also accepts `controle`, `hc`, `healthy`,
`depression`, `depressed`, `patient`, `patients` — see
`config.LABELS_MAP`).

---

## Pretrained weights

The backbone is the **3D ResNet18** released by Tencent MedicalNet,
pretrained on 23 medical-imaging datasets. Two ways to obtain it:

```bash
# via the Makefile (downloads from the official HF mirror)
make download-weights

# or manually from https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18
# place the .pth at model/resnet_18_23dataset.pth
```

`MedicalNet`'s own repo must be cloned too — it provides
`models/resnet.py` which the package imports at runtime:

```bash
git clone https://github.com/Tencent/MedicalNet.git
```

The path can be overridden with `DEPREC_MEDICALNET_DIR`.

---

## Configuration & environment variables

All defaults live in `src/depression_recognizer/config.py` and can be
overridden via environment variables — useful for CI or when sharing
the repo across hosts.

| Variable                    | Default                                | Purpose                          |
| --------------------------- | -------------------------------------- | -------------------------------- |
| `DEPREC_DATA_DIR`           | `<repo>/data`                          | Root for raw + processed data    |
| `DEPREC_DATASET_3D_DIR`     | `<repo>/data/dataset_3d`               | 3D-index NIfTI outputs           |
| `DEPREC_SPLIT_DIR`          | `<repo>/data/dataset_3d/train-test-validation` | Split folder & manifest.csv |
| `DEPREC_MODEL_DIR`          | `<repo>/model`                         | Where to look for `resnet_18_…` |
| `DEPREC_CHECKPOINTS_DIR`    | `<repo>/model/checkpoints`             | Where trained models go          |
| `DEPREC_RESULTS_DIR`        | `<repo>/model-verification/results`    | Where evaluation CSVs go         |
| `DEPREC_MEDICALNET_DIR`     | `<repo>/MedicalNet`                    | Path to cloned MedicalNet repo   |
| `DEPREC_MEDICALNET_WEIGHTS` | `<repo>/model/resnet_18_23dataset.pth` | Pretrained backbone weights      |
| `DEPREC_TR`                 | `2.5`                                  | fMRI RepetitionTime              |
| `DEPREC_BATCH_SIZE`         | `2`                                    | Training batch size              |
| `DEPREC_SEED`               | `1337`                                 | Random seed                      |

---

## Output artifacts

| Where                                          | What                                                     |
| ---------------------------------------------- | -------------------------------------------------------- |
| `data/dataset_3d/<dataset>/sub-XX/.../_3d.nii.gz` | NIfTI 4D `(X,Y,Z,3)` with `[ALFF, fALFF, ReHo]` channels |
| `data/dataset_3d/train-test-validation/manifest.csv` | Index of every sample (subject, group, split, paths) |
| `data/dataset_3d/train-test-validation/{train,validation}/{control,depr}/` | Symlinked/copied NIfTI files (input to the model) |
| `model/checkpoints/*.pt`                       | Trained model weights                                    |
| `model-verification/results/<split>_predictions.csv` | Per-sample predictions (id, label, pred, probs)     |
| `model-verification/results/<split>_metrics.json`   | Acc, confusion matrix, precision/recall/F1, ROC-AUC |

The reported metrics for the joint training run (26 subjects,
~54 % accuracy, macro-F1 ≈ 0.51) are a starting point, not a
benchmark. With more preprocessing (motion correction, band-pass
filtering, nuisance regression) and a held-out test set, the same
backbone should reach substantially better numbers.

---

## Development

```bash
make install-dev     # adds ruff, mypy, pytest
make lint
make format
make typecheck
make clean
```

`ruff` is configured in `pyproject.toml` (`line-length = 100`,
`target-version = "py310"`).

---

## References

* Zang et al., 2007 — *Altered baseline brain activity in children with
  ADHD revealed by resting-state functional MRI*. **ALFF**.
* Zou et al., 2008 — *An improved approach to detection of amplitude of
  low-frequency fluctuation (ALFF) for resting-state fMRI*. **fALFF**.
* Zang et al., 2004 — *Regional homogeneity approach to fMRI data
  analysis*. **ReHo / Kendall's W**.
* Chen et al., 2019 — *MedicalNet: Transfer Learning for 3D Medical
  Image Analysis* (Tencent).

---

## License

MIT (see `pyproject.toml`).
