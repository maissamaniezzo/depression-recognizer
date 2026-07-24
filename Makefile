# =============================================================================
# depression-recognizer — Makefile
# -----------------------------------------------------------------------------
# Common tasks: install, preprocess, split, train, infer, lint, clean.
# Run on Windows via Git Bash / WSL / MSYS2 with GNU make.
#
#     make help
# =============================================================================

# ---- Configuration (override with `make VAR=value` or env vars) ----

PYTHON        ?= python
VENV          ?= .venv
VENV_PY       := $(VENV)/Scripts/python.exe
PIP           := $(VENV_PY) -m pip

# Datasets (relative to repo root unless absolute)
DS_DS002748   ?= ./data/ds002748
DS_DS005917   ?= ./data/ds005917
DATASET_3D    ?= ./data/dataset_3d
SPLIT_DIR     ?= $(DATASET_3D)/train-test-validation

# Preprocessing
TR            ?= 2.5
NEIGHBOR      ?= 3
DEVICE        ?= auto
LOW           ?= 0.01
HIGH          ?= 0.10
CHUNK_T       ?= 32
PATCH         ?= 64 64 64

# Training
WARMUP_EPOCHS ?= 100
MAIN_EPOCHS   ?= 2000
BATCH_SIZE    ?= 2

# Inference
SPLIT         ?= validation
CKPT          ?= model/checkpoints/med3d_resnet18.pt
RESULTS_DIR   ?= model-verification/results

# MedicalNet location
MEDICALNET_DIR ?= ./MedicalNet
MEDICALNET_PTH ?= model/resnet_18_23dataset.pth

# ---- Phony targets ----
.PHONY: help install install-cpu install-cuda install-dev \
        preprocess-ds002748 preprocess-ds005917 preprocess-all \
        split train infer verify \
        lint format typecheck clean clean-data clean-venv clean-all \
        download-ds002748 download-ds005917 download-medicalnet download-weights

# ---- Help ----
help: ## Show this help
	@echo depression-recognizer - common targets
	@echo.
	@$(PYTHON) scripts\make_help.py $(MAKEFILE_LIST)

# ---- Install ----
$(VENV)/Scripts/python.exe:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/Scripts/python.exe ## Create venv and install package (CUDA by default)
	$(PIP) install -e ".[cu121]"

install-cpu: $(VENV)/Scripts/python.exe ## Install with CPU-only torch
	$(PIP) install -e ".[cpu]"

install-cuda: install ## alias for the default install

install-dev: ## Install dev tools (lint, test, typecheck)
	$(PIP) install -e ".[cu121,dev]"

# ---- Datasets ----
download-ds002748: ## Clone OpenNeuro ds002748 into ./data/ds002748
	@mkdir -p ./data
	cd ./data && git clone https://github.com/OpenNeuroDatasets/ds002748.git

download-ds005917: ## Clone OpenNeuro ds005917 into ./data/ds005917
	@mkdir -p ./data
	cd ./data && git clone https://github.com/OpenNeuroDatasets/ds005917.git

download-medicalnet: ## Clone Tencent MedicalNet (needed for the 3D ResNet)
	git clone https://github.com/Tencent/MedicalNet.git

download-weights: ## Download pretrained Med3D ResNet18 weights via the HF mirror
	$(VENV_PY) -c "import urllib.request, os; \
url='https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18_23dataset.pth'; \
os.makedirs('model', exist_ok=True); \
urllib.request.urlretrieve(url, '$(MEDICALNET_PTH)') if not os.path.exists('$(MEDICALNET_PTH)') else print('already exists')"

# ---- Preprocessing ----
preprocess-ds002748: ## fMRI -> (X,Y,Z,3) NIfTI for ds002748
	$(VENV_PY) -m depression_recognizer.preprocessing.bids_to_3d \
	    --in-dir $(DS_DS002748) \
	    --out-dir $(DATASET_3D) \
	    --bids-layout ds002748 \
	    --dataset-name ds002748 \
	    --tr $(TR) --neighbor $(NEIGHBOR) --low $(LOW) --high $(HIGH) \
	    --chunk-t $(CHUNK_T) --device $(DEVICE) \
	    --patch $(PATCH) --auto-tr --skip-if-exists

preprocess-ds005917: ## fMRI -> (X,Y,Z,3) NIfTI for ds005917 (ses-b0 baseline)
	$(VENV_PY) -m depression_recognizer.preprocessing.bids_to_3d \
	    --in-dir $(DS_DS005917) \
	    --out-dir $(DATASET_3D) \
	    --bids-layout ds005917 \
	    --dataset-name ds005917 \
	    --tr $(TR) --neighbor $(NEIGHBOR) --low $(LOW) --high $(HIGH) \
	    --chunk-t $(CHUNK_T) --device $(DEVICE) \
	    --patch $(PATCH) --auto-tr --skip-if-exists

preprocess-all: preprocess-ds002748 preprocess-ds005917 ## Run preprocessing for both datasets

# ---- Split ----
split: ## Build the stratified train/validation split (manifest.csv)
	$(VENV_PY) -m depression_recognizer.data.split \
	    --base-dir $(DATASET_3D) --val-ratio 0.2 --seed 42 --link

# ---- Train / infer ----
train: ## Fine-tune Med3D ResNet18 on the prepared split
	DEPREC_MEDICALNET_DIR=$(MEDICALNET_DIR) \
	DEPREC_MEDICALNET_WEIGHTS=$(MEDICALNET_PTH) \
	$(VENV_PY) -m depression_recognizer.modeling.train \
	    --data-root $(SPLIT_DIR) \
	    --warmup-epochs $(WARMUP_EPOCHS) --main-epochs $(MAIN_EPOCHS) \
	    --batch-size $(BATCH_SIZE) \
	    --output $(CKPT)

infer: ## Evaluate a trained checkpoint on --split (default validation)
	DEPREC_MEDICALNET_DIR=$(MEDICALNET_DIR) \
	DEPREC_MEDICALNET_WEIGHTS=$(MEDICALNET_PTH) \
	$(VENV_PY) -m depression_recognizer.modeling.infer \
	    --data-root $(SPLIT_DIR) \
	    --checkpoint $(CKPT) \
	    --split $(SPLIT) \
	    --out-dir $(RESULTS_DIR)

verify: infer ## alias for infer

# ---- Quality ----
lint: ## Run ruff
	$(VENV_PY) -m ruff check src

format: ## Format with ruff
	$(VENV_PY) -m ruff format src
	$(VENV_PY) -m ruff check --fix src

typecheck: ## Run mypy (best-effort)
	$(VENV_PY) -m mypy src || true

# ---- Clean ----
clean: ## Remove generated artifacts (checkpoints, results, caches)
	rm -rf model/checkpoints/* model-verification/results/* 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache .pytest_cache 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data: ## Remove generated dataset_3d (keeps raw OpenNeuro clones)
	rm -rf $(DATASET_3D)

clean-venv: ## Remove the virtualenv
	rm -rf $(VENV)

clean-all: clean clean-data clean-venv ## Remove everything (venv + data + caches)
