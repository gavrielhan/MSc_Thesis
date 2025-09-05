## MSc_Thesis

Fine-tuning and evaluating JEPA-based vision transformers for retinal image analysis across multiple datasets (IDRID, Messidor, PAPILA). The repo includes utilities for dataset preparation, LoRA-based fine-tuning, validation-driven model selection, and feature-based KNN evaluation.

### Key features
- LoRA fine-tuning on retina datasets with Vision Transformer encoder
- Stratified 20% validation split from the training set per dataset
- Best-epoch selection by validation macro ROC AUC; corresponding test metrics recorded
- Optional image preprocessing: crop black borders, autocontrast, equalize
- KNN evaluation on frozen features
- Lightweight logs per-epoch 
- Default batch size is 1 to accommodate memory constraints 

### Repository layout (selected)
- `JEPA/ijepa_finetune_crossdatasets.py`: Main fine-tune/eval entrypoint across datasets
- `JEPA/ijepa_retina_pretrain.py`, `JEPA/ijepa_retina_ft.py`: Retina-specific pretrain/fine-tune flows
- `JEPA/test_retina_pretrain.py`: Evaluation helpers
- `JEPA/prepare_retina_dataset.py`: Dataset preparation/utilities
- `JEPA/ijepa/src/`: Model, training, masks, transforms, and utilities for JEPA

## Environment setup
1) Create environment (conda recommended):
```bash
conda create -n retina-jepa python=3.10 -y
conda activate retina-jepa
```
2) Install PyTorch (choose the right CUDA/Metal build if using GPU):
```bash
pip install torch torchvision torchaudio
```
3) Install Python deps:
```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm wandb
```

## Data preparation
Datasets are expected under a single root configured in the scripts (see CONFIG keys). For cross-dataset fine-tuning:
- IDRID: `IDRID/B. Disease Grading/`
- Messidor: `messidor/` with `IMAGES/`, `messidor_data.csv`, and `messidor-2.csv`
- PAPILA: `PAPILA/PapilaDB-.../FundusImages/` and provided `HelpCode/kfold/Test 1/`

You can adapt or run `JEPA/prepare_retina_dataset.py` to align file structures if needed.

## Fine-tuning and evaluation (cross-datasets)
Entrypoint: `JEPA/ijepa_finetune_crossdatasets.py`

### Strategies
- `retina_feature_finetune`: fine-tune from a retina checkpoint
- `imagenet_finetune`: fine-tune from ImageNet-22k checkpoint
- `retina_pretrain_finetune`: fine-tune from a retina pretrain checkpoint
- `retina_feature_knn`: frozen feature KNN evaluation

### Validation-driven selection
- From the training set, a stratified 20% validation split is created per dataset.
- At each epoch, validation metrics are computed; the epoch with highest validation macro ROC AUC is tracked.
- The test metrics recorded at that best-validation epoch are persisted as the final test report.

### Preprocessing
Configurable, applied before resizing/normalization:
- Crop black borders (threshold + margin)
- Optional `autocontrast` and `equalize`

### Important CONFIG keys (within the script)
- `img_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`
- LoRA: `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`
- Optimization: `batch_size` (defaults to 1 [[memory:4844974]]), `lr`, `weight_decay`, `epochs`, `num_workers`
- Datasets: `external_root`
- Preprocessing: `preprocess_crop_black`, `preprocess_autocontrast`, `preprocess_equalize`, `preprocess_crop_threshold`, `preprocess_crop_margin`

### Run examples
Fine-tune with ImageNet encoder and evaluate on IDRID (default):
```bash
python JEPA/ijepa_finetune_crossdatasets.py --strategy imagenet_finetune
```
Run KNN feature evaluation:
```bash
python JEPA/ijepa_finetune_crossdatasets.py --strategy retina_feature_knn
```
Disable W&B (default telemetry is disabled, logging only if initialized):
```bash
WANDB_DISABLED=true python JEPA/ijepa_finetune_crossdatasets.py --strategy imagenet_finetune
```

## Outputs
- Checkpoints, JSON results, and plots are saved under the paths configured inside the script (`OUTPUT_DIRS`).
- Per-epoch checkpoint includes states and last computed metrics.
- Result JSON includes (among others):
  - `train_losses`
  - `val_last_aucs`, `val_last_pr_aucs`, `val_last_f1s`
  - `best_val_macro_auc`, `best_val_epoch`, `best_val_aucs`, `best_val_pr_aucs`, `best_val_f1s`
  - `best_test_aucs_at_best_val`, `best_test_pr_aucs_at_best_val`, `best_test_f1s_at_best_val`, `best_test_confusion_matrix_at_best_val`
  - final `aucs`, `pr_aucs`, `f1s`, `confusion_matrix`, `all_labels`, `all_probs`

## GNN risk forecasting (CGM + sleep + EHR diagnosis)
Build temporal heterogeneous graphs from CGM, sleep embeddings, and diagnosis events, then train a HeteroGNN to forecast condition risk and time-to-event.

- **Graph construction** (`graph_gen_split.py`, `mk_graphs.py`):
  - **Windows**: sliding 3-day windows over CGM; patient splits into train/val/test at the patient level.
  - **Nodes**: `patient` (concatenated patient/sleep or demographics), `signature` (NMF CGM signatures), `condition` (one-hot conditions).
  - **Edges**:
    - `patient -> signature`: weights from least-squares projection of daily CGM onto NMF signatures.
    - `patient has condition`: label edges at/around diagnosis; attributes schedule risk approaching diagnosis; negatives get baseline risk for diabetes when focused.
    - `patient follows patient`: temporal links across windows with simple age-based weights.
  - **Artifacts**: saves `split/train_graphs_3d.pt`, `split/val_graphs_3d.pt`, `split/test_graphs_3d.pt`, and `diagnosis_mapping.json`.

- **Model** (`model.py`, `train.py`, `eval.py`, `link_prediction_head.py`):
  - **Encoder**: `HeteroGAT` with `HeteroConv` over relation-specific `GATConv`/`GCNConv`.
  - **Heads**: time-aware cosine link predictor (optionally with tte), `CoxHead` for survival, `JointHead` for uncertainty-weighted multi-tasking.
  - **Training**: per-window batches, weighted BCE on horizon positives + MSE on edge attributes; optional pseudo-labels; gradient clipping.
  - **Metrics**: PR AUC/ROC AUC (patient-condition), C-index (per condition), optional absolute risk calibration via lifelines.

- **Run**:
```bash
# Generate graphs and splits
python graph_gen_split.py

# Train + evaluate GNN (uses train/val/test graphs from split/)
python model.py
```

- **Dependencies**: `torch_geometric` (install per official instructions: `https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html`), `lifelines`, and internal `LabData` loaders for CGM/body measures.

## Reproducibility
- Global random seeds are set in the scripts; CUDA deterministic flags are used when applicable.

## Troubleshooting
- If GPU/Metal memory is tight, keep `batch_size=1` [[memory:4844974]] and `num_workers=0`.
- If images look too dark or centered with large black borders, enable crop or tone adjustments in preprocessing.
- If W&B connectivity is an issue, run without initializing W&B (sweep mode) or set `WANDB_DISABLED=true`.
