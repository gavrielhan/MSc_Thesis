#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_recall_curve
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import glob
import re
import wandb

# IJepa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer

# ---------- Output Directories ----------
OUTPUT_DIRS = {
    'checkpoints': "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/",
    'results': "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/",
    'images': os.path.join('outputs', 'images'),
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ---------- Config ----------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable W&B telemetry to avoid connection issues
os.environ["WANDB_TELEMETRY_DISABLED"] = "true"

# Memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# For IDRID and Messidor: use single split (no folds)
# For PAPILA: use existing fold structure
USE_SINGLE_SPLIT = True  # Set to True for IDRID/Messidor, False for PAPILA

CONFIG = {
    'img_size': (616, 616),
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'use_lora': True,
    'lora_r': 16,  # Match the pretrained checkpoint
    'lora_alpha': 16,  # Match the pretrained checkpoint
    'lora_dropout': 0.2,  # Match the pretrained checkpoint
    'batch_size': 1,  # Minimal batch size to save memory
    'num_workers': 0,  # No workers to save memory
    'lr': 1e-4,  # Reduced learning rate for more stable fine-tuning
    'weight_decay': 1e-2,  # Increased weight decay to prevent overfitting
    'epochs': 30,  # Increased epochs for better convergence
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets',
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Memory cleanup function
def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Function to manually count images in a dataset
def count_images_in_dataset(dataset, dataset_name):
    """Manually count images and black images in a dataset"""
    total_checked = 0
    black_count = 0

    print(f"Manually checking {dataset_name} dataset...")
    for i in range(len(dataset)):
        try:
            # Get the original image path and load it directly
            if hasattr(dataset, 'df') and hasattr(dataset, 'img_dir'):
                # For Messidor dataset (check first since it also has label_col)
                if 'image_id' in dataset.df.columns:
                    row = dataset.df.iloc[i]
                    img_name = row['image_id']
                    img_path = os.path.join(dataset.img_dir, img_name)
                # For PAPILA dataset
                elif 'Patient ID' in dataset.df.columns:
                    row = dataset.df.iloc[i]
                    pid = row['Patient ID']
                    eye = row['eyeID']
                    img_name = f"RET{pid}{eye}.jpg"
                    img_path = os.path.join(dataset.img_dir, img_name)
                # For IDRID dataset
                elif hasattr(dataset, 'label_col'):
                    row = dataset.df.iloc[i]
                    img_name = row['Image name']
                    if not img_name.endswith('.jpg'):
                        img_name = img_name + '.jpg'
                    img_path = os.path.join(dataset.img_dir, img_name)
                else:
                    # Fallback to checking transformed image
                    img, label = dataset[i]
                    total_checked += 1
                    if hasattr(img, 'numpy'):
                        img_array = img.numpy()
                    else:
                        img_array = np.array(img)
                    # For normalized images, check if they're close to zero
                    if np.mean(img_array) < 0.1:  # Much lower threshold for normalized images
                        black_count += 1
                    continue

                # Load original image and check
                original_img = Image.open(img_path).convert('RGB')
                img_array = np.array(original_img)
                total_checked += 1

                # Check if image is mostly black (mean < 30 for RGB images)
                if np.mean(img_array) < 30:
                    black_count += 1
                    print(f"Warning: {dataset_name} image {img_name} is very dark (mean: {np.mean(img_array):.1f})")

            else:
                # Fallback for other dataset types
                img, label = dataset[i]
                total_checked += 1
                if hasattr(img, 'numpy'):
                    img_array = img.numpy()
                else:
                    img_array = np.array(img)
                if np.mean(img_array) < 0.1:
                    black_count += 1

        except Exception as e:
            print(f"Error checking {dataset_name} sample {i}: {e}")
            black_count += 1

    return total_checked, black_count


# ---------- Data Utilities ----------
class PapilaDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.black_image_count = 0
        self.total_images_checked = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['Patient ID']
        eye = row['eyeID']
        label = int(row['Diagnosis'])
        img_name = f"RET{pid}{eye}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            # Check if image is mostly black
            img_array = np.array(img)
            self.total_images_checked += 1
            if np.mean(img_array) < 10:  # Very dark image
                self.black_image_count += 1
                print(f"Warning: Image {img_name} is very dark (mean: {np.mean(img_array):.1f})")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', CONFIG['img_size'], 'black')
            self.black_image_count += 1
        img = self.transform(img)
        return img, label


class IDRIDDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.black_image_count = 0
        self.total_images_checked = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image name']
        # Add .jpg extension if not present
        if not img_name.endswith('.jpg'):
            img_name = img_name + '.jpg'
        label = int(row[self.label_col])

        # Ensure label is in valid range
        if label < 0 or label > 1:
            print(f"Warning: Invalid label {label} for image {img_name}. Fixing to 0.")
            label = 0

        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            # Check if image is mostly black
            img_array = np.array(img)
            self.total_images_checked += 1
            if np.mean(img_array) < 10:  # Very dark image
                self.black_image_count += 1
                print(f"Warning: Image {img_name} is very dark (mean: {np.mean(img_array):.1f})")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', CONFIG['img_size'], 'black')
            self.black_image_count += 1
        img = self.transform(img)
        return img, label


class MessidorDataset(Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.black_image_count = 0
        self.total_images_checked = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        label = int(row[self.label_col])
        img_path = os.path.join(self.img_dir, img_name)

        # Check if image file exists (case-insensitive)
        if not os.path.exists(img_path):
            # Try with different case extensions
            base_name = os.path.splitext(img_name)[0]
            possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

            found_file = None
            for ext in possible_extensions:
                alt_path = os.path.join(self.img_dir, base_name + ext)
                if os.path.exists(alt_path):
                    found_file = alt_path
                    break

            if found_file:
                img_path = found_file
            else:
                print(f"Image file not found: {img_path}")
                img = Image.new('RGB', CONFIG['img_size'], 'black')
                self.black_image_count += 1
                img = self.transform(img)
                return img, label

        try:
            img = Image.open(img_path).convert('RGB')
            # Check if image is mostly black
            img_array = np.array(img)
            self.total_images_checked += 1
            if np.mean(img_array) < 10:  # Very dark image
                self.black_image_count += 1
                print(f"Warning: Image {img_name} is very dark (mean: {np.mean(img_array):.1f})")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', CONFIG['img_size'], 'black')
            self.black_image_count += 1
        img = self.transform(img)
        return img, label


# ---------- Model Utilities ----------
class ClassificationHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, 0]
        return self.head(x)


def build_encoder(config):
    enc = VisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_chans=3,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    # Note: VisionTransformer doesn't support gradient_checkpointing_enable()
    # Memory optimization is handled through reduced batch size and image size
    return enc


# Accept checkpoint path directly
def load_pretrained_encoder(enc, ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        enc_state = state.get('enc', state)

        # Load ALL parameters including LoRA parameters
        # The encoder state dict contains all parameters including LoRA
        filtered = enc_state

        # Handle input channel mismatch
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 6 and enc.patch_embed.proj.weight.shape[1] == 3:
            filtered['patch_embed.proj.weight'] = w[:, :3]

        # Load state dict and report what was loaded
        missing_keys, unexpected_keys = enc.load_state_dict(filtered, strict=False)
        logger.info("Pretrained weights loaded into encoder")
        logger.info(f"Loaded {len(filtered)} parameters")
        logger.info(f"Missing keys: {len(missing_keys)}")
        logger.info(f"Unexpected keys: {len(unexpected_keys)}")

        # Debug: Check if LoRA parameters were loaded
        lora_params_loaded = [k for k in filtered.keys() if k.startswith('lora_')]
        logger.info(f"LoRA parameters loaded: {len(lora_params_loaded)}")
        if lora_params_loaded:
            logger.info(f"Sample LoRA params: {lora_params_loaded[:5]}")
        else:
            logger.warning("No LoRA parameters found in checkpoint!")
            logger.info(f"Available parameter prefixes: {list(set([k.split('.')[0] for k in filtered.keys()]))}")
            logger.info(f"First 10 loaded parameters: {list(filtered.keys())[:10]}")
            logger.info(f"Total parameters in checkpoint: {len(filtered)}")
    else:
        logger.info("No pretrained checkpoint found; skipping load")


def freeze_lora_params(model):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


# ---------- Evaluation Utilities ----------
def evaluate_and_plot_rocauc(all_labels, all_probs, n_classes, dataset_name):
    auc_values = []
    for c in range(n_classes):
        y_true = (all_labels == c).astype(int)
        y_score = all_probs[:, c]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
        auc_values.append(auc)
        print(f"{dataset_name} ROC AUC for class {c}: {auc:.4f}")
    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar([f'class_{c}' for c in range(n_classes)], auc_values, color='orange', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title(f'ROC AUC per Class ({dataset_name})')
    for bar, auc in zip(bars, auc_values):
        y = max(bar.get_height() + 0.05, 0.05)
        plt.text(bar.get_x() + bar.get_width() / 2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'roc_auc_per_class_{dataset_name.lower()}.png')
    plt.close()


# ---------- Strategy Configurations ----------
STRATEGIES = [
    {
        "name": "retina_feature_finetune",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth",
        "eval_type": "finetune"
    },
    {
        "name": "imagenet_finetune",
        "ckpt": "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar",
        "eval_type": "finetune"
    },
    {
        "name": "retina_pretrain_finetune",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_pretrain_newrun.pth",
        "eval_type": "finetune"
    },
    {
        "name": "retina_feature_knn",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth",
        "eval_type": "knn"
    }
]

# Strategy selection - change this index to select different strategies
# 0: retina_feature_finetune
# 1: imagenet_finetune
# 2: retina_pretrain_finetune
# 3: retina_feature_knn
STRATEGY_INDEX = 0  # Change this to select different strategies


# ---------- Utility: Save/Load Results ----------
def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# ---------- KNN Evaluation ----------
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            if feats.ndim == 3:
                feats = feats[:, 0]  # Use CLS token
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def knn_evaluate(encoder, train_loader, test_loader, n_classes, device, k=5):
    X_train, y_train = extract_features(encoder, train_loader, device)
    X_test, y_test = extract_features(encoder, test_loader, device)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_prob = knn.predict_proba(X_test)
    aucs = []
    for c in range(n_classes):
        y_true = (y_test == c).astype(int)
        y_score = y_prob[:, c]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
    return aucs, y_test, y_prob


# ---------- Main Training/Eval Loop (Refactored) ----------
def main(strategy_name=None, fold=0, sweep_mode=False):
    import os

    # Initialize W&B
    if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
        # Normal mode - initialize W&B with fixed config
        wandb.init(
            project="retina-lora-finetuning_strat1",
            config={
                "dataset": "messidor",
                "strategy": strategy_name or "imagenet_finetune",
                "lora_r": CONFIG['lora_r'],
                "lora_alpha": CONFIG['lora_alpha'],
                "lr": CONFIG['lr'],
                "batch_size": CONFIG['batch_size'],
                "epochs": CONFIG['epochs'],
                "weight_decay": CONFIG['weight_decay'],
            },
            name=f"messidor_lora_r{CONFIG['lora_r']}_lr{CONFIG['lr']}"
        )
    else:
        # Sweep mode - don't use W&B at all
        print("SWEEP MODE: Running without W&B logging")
        # Don't try to update wandb.config since W&B isn't initialized

    # User selects dataset for fine-tuning and testing
    train_dataset = "messidor"  # Start with IDRID
    test_dataset = "messidor"

    if sweep_mode:
        print(f"? SWEEP MODE: Running {strategy_name} with config:")
        print(f"  LoRA r: {CONFIG['lora_r']}, alpha: {CONFIG['lora_alpha']}, dropout: {CONFIG['lora_dropout']}")
        print(f"  LR: {CONFIG['lr']}, weight_decay: {CONFIG['weight_decay']}, epochs: {CONFIG['epochs']}")
    else:
        print("Selected dataset for fine-tuning (train): ", train_dataset)
        print("Selected dataset for testing (eval): ", test_dataset)
        print("Using single split (no folds) to avoid data leakage")

    trsf = build_transforms()
    n_folds = 5
    seeds = [42, 43, 44, 45, 46]  # For reproducible random splits per fold

    # Prepare dataset splits
    if train_dataset == 'idrid':
        root = os.path.join(CONFIG['external_root'], 'IDRID', 'B. Disease Grading')
        img_dir = os.path.join(root, '1. Original Images', 'a. Training Set')
        label_csv = os.path.join(root, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        df = pd.read_csv(label_csv)
        # Clean up columns: strip whitespace, drop all-NaN and Unnamed columns
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        # Binarize both relevant columns, robust to missing columns
        for col in ['Retinopathy grade', 'Risk of macular edema']:
            if col in df.columns:
                df[col] = (df[col] > 0).astype(int)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")
        label_col = 'Retinopathy grade'

        # IDRID already has predefined splits - use them
        # Load training and test splits from IDRID's predefined structure
        train_csv = os.path.join(root, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        test_csv = os.path.join(root, '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')

        if os.path.exists(test_csv):
            # Use predefined splits
            df_train = pd.read_csv(train_csv)
            df_test = pd.read_csv(test_csv)
            # Clean and binarize both
            for df_split in [df_train, df_test]:
                df_split.columns = df_split.columns.str.strip()
                df_split = df_split.loc[:, ~df_split.columns.str.contains('^Unnamed')]
                df_split = df_split.dropna(axis=1, how='all')
                for col in ['Retinopathy grade', 'Risk of macular edema']:
                    if col in df_split.columns:
                        df_split[col] = (df_split[col] > 0).astype(int)

            # Debug: Check label ranges
            print(f"Training set label range: {df_train[label_col].min()} to {df_train[label_col].max()}")
            print(f"Test set label range: {df_test[label_col].min()} to {df_test[label_col].max()}")
            print(f"Training set label distribution: {df_train[label_col].value_counts().sort_index().to_dict()}")
            print(f"Test set label distribution: {df_test[label_col].value_counts().sort_index().to_dict()}")

            # Ensure labels are in correct range (0 to n_classes-1)
            for df_split, split_name in [(df_train, 'train'), (df_test, 'test')]:
                if df_split[label_col].min() < 0 or df_split[label_col].max() >= 2:
                    print(f"Warning: {split_name} set has labels outside [0,1] range. Fixing...")
                    df_split[label_col] = df_split[label_col].clip(0, 1).astype(int)
                    print(
                        f"After fixing: {split_name} set label range: {df_split[label_col].min()} to {df_split[label_col].max()}")

            splits = [(df_train, df_test)]  # Use predefined splits
            n_folds = 1
            print("Using IDRID's predefined train/test splits")
        else:
            # Fallback to creating splits if predefined ones don't exist
            indices = np.arange(len(df))
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            shuffled = rng.permutation(indices)
            split = int(0.8 * len(df))
            train_idx = shuffled[:split]
            test_idx = shuffled[split:]
            splits = [(train_idx, test_idx)]  # Single split
            n_folds = 1
            print("Created single train/test split for IDRID")
        n_classes_idrid = 2  # Always binary after binarization

    elif train_dataset == 'papila':
        # Keep existing fold structure for PAPILA
        root = os.path.join(CONFIG['external_root'], 'PAPILA',
                            'PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0')
        img_dir = os.path.join(root, 'FundusImages')
        kfold_dir = os.path.join(root, 'HelpCode', 'kfold', 'Test 1')
        splits = []
        for fold in range(n_folds):
            train_xlsx = os.path.join(kfold_dir, 'Train', f'test_1_train_index_fold_{fold}.xlsx')
            test_xlsx = os.path.join(kfold_dir, 'Test', f'test_1_test_index_fold_{fold}.xlsx')
            splits.append((train_xlsx, test_xlsx))
        n_classes_papila = 3  # 0, 1, 2

    elif train_dataset == 'messidor':
        root = os.path.join(CONFIG['external_root'], 'messidor')
        img_dir = os.path.join(root, 'IMAGES')
        data_csv = os.path.join(root, 'messidor_data.csv')
        patient_csv = os.path.join(root, 'messidor-2.csv')
        df = pd.read_csv(data_csv)
        # Binarize adjudicated_dr_grade
        df['adjudicated_dr_grade'] = (df['adjudicated_dr_grade'] != 0).astype(int)
        for col in ['adjudicated_dme', 'adjudicated_gradable']:
            df[col] = (df[col] != 0).astype(int)
        df_pat = pd.read_csv(patient_csv)
        df_pat.columns = df_pat.columns.str.strip()
        col = df_pat.columns[0]  # Should be the 'left;right' column
        patient_groups = []
        for _, row in df_pat.iterrows():
            imgs = [img.strip() for img in str(row[col]).split(';') if img.strip() and img.strip().lower() != 'nan']
            if imgs:
                patient_groups.append(imgs)
        n_pat = len(patient_groups)

        # Messidor doesn't have predefined splits - create them
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        shuffled_groups = patient_groups.copy()
        rng.shuffle(shuffled_groups)
        split = int(0.8 * n_pat)
        train_pat = shuffled_groups[:split]
        test_pat = shuffled_groups[split:]
        train_imgs = set([img for group in train_pat for img in group])
        test_imgs = set([img for group in test_pat for img in group])
        splits = [(train_imgs, test_imgs, df)]  # Single split
        n_folds = 1

        # Save the split for later use in inference
        split_info = {
            'train_patients': train_pat,
            'test_patients': test_pat,
            'train_images': list(train_imgs),
            'test_images': list(test_imgs),
            'seed': 42
        }
        split_path = os.path.join(OUTPUT_DIRS['results'], 'messidor_split.json')
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Messidor split saved to: {split_path}")
        n_classes_messidor = 2  # Binary after binarization
    else:
        raise NotImplementedError(f"Dataset not implemented: {train_dataset}")

    # Select the strategy to run
    if strategy_name:
        # Find strategy by name
        strategy_select = None
        for strategy in STRATEGIES:
            if strategy['name'] == strategy_name:
                strategy_select = strategy
                break
        if strategy_select is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    else:
        strategy_select = STRATEGIES[STRATEGY_INDEX]
    print(f"\n=== Running strategy: {strategy_select['name']} ===")

    if not os.path.isfile(strategy_select['ckpt']):
        print(f"Checkpoint for strategy '{strategy_select['name']}' not found at {strategy_select['ckpt']}. Exiting.")
        return

    # Handle different dataset types
    if strategy_name and fold is not None:
        # Use provided fold for sweep mode
        folds_to_run = [fold]
        print(f"Running fold {fold} for strategy {strategy_select['name']} (sweep mode)")
    elif train_dataset in ['idrid', 'messidor']:
        # Single split datasets - always use fold 0
        fold = 0
        print(f"Running single split (fold 0) for strategy {strategy_select['name']} on {train_dataset}")
        folds_to_run = [0]
    elif train_dataset == 'papila':
        # PAPILA: run all folds sequentially with fresh models
        print(f"Running PAPILA sequentially for all folds with fresh models")
        folds_to_run = list(range(n_folds))  # Run all 5 folds
    else:
        fold = 0
        print(f"Running fold {fold} for strategy {strategy_select['name']}")
        folds_to_run = [0]

    # Run each fold (for PAPILA this will be multiple folds, for others just one)
    for fold in folds_to_run:
        print(f"\n--- Processing fold {fold} ---")

        # --- Main fold logic ---
        result_path = os.path.join(OUTPUT_DIRS['results'],
                                   f"results_{strategy_select['name']}_{train_dataset}_fold{fold}.json")
        ckpt_path = os.path.join(OUTPUT_DIRS['checkpoints'],
                                 f"model_{strategy_select['name']}_{train_dataset}_fold{fold}.pth")
        time_ckpt_pattern = os.path.join(OUTPUT_DIRS['checkpoints'],
                                         f"model_{strategy_select['name']}_{train_dataset}_fold{fold}_epoch*_timechkpt.pth")
        time_ckpt_interval = 41400  # 11.5 hours in seconds
        start_epoch = 1
        last_time_ckpt = None

        # Prepare train/test datasets for this fold
        if train_dataset == 'idrid':
            df_train, df_test = splits[0]  # Single split
            train_ds = IDRIDDataset(df_train, img_dir, trsf, label_col=label_col)
            test_ds = IDRIDDataset(df_test, img_dir, trsf, label_col=label_col)

            # Check a few images to see if they're loading properly
            print("Checking image loading...")
            for i in range(min(5, len(train_ds))):
                img, label = train_ds[i]
                print(f"Sample {i}: Image shape {img.shape}, Label {label}")

            print(f"Train dataset: {len(train_ds)} images")
            print(f"Test dataset: {len(test_ds)} images")

            n_classes = n_classes_idrid
        elif train_dataset == 'papila':
            train_xlsx, test_xlsx = splits[fold]  # Use specific fold
            df_train = pd.read_excel(train_xlsx)
            df_test = pd.read_excel(test_xlsx)
            train_ds = PapilaDataset(df_train, img_dir, trsf)
            test_ds = PapilaDataset(df_test, img_dir, trsf)
            n_classes = n_classes_papila
        elif train_dataset == 'messidor':
            train_imgs, test_imgs, df_all = splits[0]  # Single split
            label_col = 'adjudicated_dr_grade'
            df_train = df_all[df_all['image_id'].isin(train_imgs)].reset_index(drop=True)
            df_test = df_all[df_all['image_id'].isin(test_imgs)].reset_index(drop=True)
            train_ds = MessidorDataset(df_train, img_dir, trsf, label_col=label_col)
            test_ds = MessidorDataset(df_test, img_dir, trsf, label_col=label_col)
            n_classes = n_classes_messidor
        else:
            raise NotImplementedError

        # Check images at the beginning
        print(f"\nImage Loading Summary:")
        train_total, train_black = count_images_in_dataset(train_ds, "train")
        test_total, test_black = count_images_in_dataset(test_ds, "test")
        print(f"Train dataset - Total checked: {train_total}, Black images: {train_black}")
        print(f"Test dataset - Total checked: {test_total}, Black images: {test_black}")

        # If all images are black, there's a serious problem
        if train_black == train_total and test_black == test_total:
            error_msg = "All images are black - cannot proceed with training!"
            print(f"ERROR: {error_msg}")
            print("Possible causes:")
            print("1. Image paths are incorrect")
            print("2. Images are actually black/damaged")
            print("3. Image loading is failing")
            print("4. Threshold for 'black' detection is too strict")

            if sweep_mode:
                # In sweep mode, just return instead of raising exception
                print("SWEEP MODE: Logging error and continuing with next trial")
                return
            else:
                raise RuntimeError(error_msg)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        # Use num_workers=0 for test_loader to ensure black image tracking works properly
        # (When using multiple workers, tracking attributes don't propagate back to main process)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0,
                                 pin_memory=True)

        # Calculate class weights to handle imbalance
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        train_labels = np.array(train_labels)
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        print(f"Class distribution in training: {class_counts}")
        print(f"Class weights: {class_weights.cpu().numpy()}")
        print(f"Class weight ratio (Class 1 / Class 0): {class_weights[1].item() / class_weights[0].item():.3f}")

        # Debug: Compare train vs test distributions
        test_labels = []
        for _, labels in test_loader:
            test_labels.extend(labels.numpy())
        test_labels = np.array(test_labels)
        test_class_counts = np.bincount(test_labels)
        print(f"Class distribution in test: {test_class_counts}")

        train_ratio = class_counts[1] / class_counts[0] if class_counts[0] > 0 else float('inf')
        test_ratio = test_class_counts[1] / test_class_counts[0] if test_class_counts[0] > 0 else float('inf')
        print(f"Train Class 1/Class 0 ratio: {train_ratio:.3f}")
        print(f"Test Class 1/Class 0 ratio: {test_ratio:.3f}")
        print(f"Ratio difference: {abs(train_ratio - test_ratio):.3f}")

        # Debug: Sample a few images to check basic statistics
        print("\nComparing train vs test image statistics...")
        train_imgs_sample = []
        test_imgs_sample = []

        # Sample from train loader
        for i, (imgs, _) in enumerate(train_loader):
            train_imgs_sample.append(imgs.cpu().numpy())
            if i >= 5:  # Sample first few batches
                break

        # Sample from test loader
        for i, (imgs, _) in enumerate(test_loader):
            test_imgs_sample.append(imgs.cpu().numpy())
            if i >= 5:  # Sample first few batches
                break

        if train_imgs_sample and test_imgs_sample:
            train_imgs = np.concatenate(train_imgs_sample, axis=0)
            test_imgs = np.concatenate(test_imgs_sample, axis=0)

            print(f"Train images: mean={np.mean(train_imgs):.4f}, std={np.std(train_imgs):.4f}")
            print(f"Test images: mean={np.mean(test_imgs):.4f}, std={np.std(test_imgs):.4f}")
            print(f"Mean difference: {abs(np.mean(train_imgs) - np.mean(test_imgs)):.4f}")
            print(f"Std difference: {abs(np.std(train_imgs) - np.std(test_imgs)):.4f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        results = {"fold": fold, "strategy": strategy_select['name'], "train_dataset": train_dataset,
                   "test_dataset": test_dataset}

        # Resume logic and strategy handling
        if strategy_select['eval_type'] == "finetune":
            # For PAPILA: always start with fresh model (no checkpoint loading)
            # For other datasets: try to load checkpoint if available
            if train_dataset == 'papila':
                print(f"Starting PAPILA fold {fold} with fresh model (no checkpoint loading)")
                model = nn.Sequential(build_encoder(CONFIG), ClassificationHead(CONFIG['embed_dim'], n_classes)).to(
                    DEVICE)

                # FREEZE all parameters except LoRA and classification head
                freeze_lora_params(model[0])  # Freeze encoder parameters except LoRA
                # Classification head should remain trainable
                for param in model[1].parameters():
                    param.requires_grad = True

                # Debug: Check if LoRA parameters exist after freezing
                lora_params = [name for name, _ in model[0].named_parameters() if 'lora_' in name]
                trainable_lora_params = [name for name, param in model[0].named_parameters() if
                                         'lora_' in name and param.requires_grad]
                print(f"LoRA parameters found: {len(lora_params)}")
                print(f"Trainable LoRA parameters: {len(trainable_lora_params)}")
                if len(lora_params) == 0:
                    print("WARNING: No LoRA parameters found! The model might not be using LoRA.")
                if len(trainable_lora_params) == 0:
                    print("WARNING: No trainable LoRA parameters! LoRA might not be working.")

                # Debug: Check LoRA parameter values to verify they're loaded
                if len(lora_params) > 0:
                    sample_lora_param = next(p for n, p in model[0].named_parameters() if 'lora_' in n)
                    print(
                        f"Sample LoRA parameter mean: {sample_lora_param.mean().item():.6f}, std: {sample_lora_param.std().item():.6f}")
                    print(f"Strategy: {strategy_select['name']} - LoRA should be different for different strategies!")

                optimizer = AdamW([
                    {'params': [p for n, p in model[0].named_parameters() if p.requires_grad],
                     'weight_decay': CONFIG['weight_decay'], 'lr': CONFIG['lr']},  # LoRA LR
                    {'params': model[1].parameters(), 'weight_decay': 0.0, 'lr': CONFIG['lr'] * 2}
                    # Higher LR for classification head
                ])
                # Use warmup + cosine annealing to help escape local optimum
                from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
                warmup_epochs = 1
                main_epochs = CONFIG['epochs'] - warmup_epochs
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
                cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs)
                train_losses = []
                start_epoch = 1
                last_time_ckpt = time.time()
            else:
                # For IDRID/Messidor: load pretrained model from strategy checkpoint
                print(f"Loading pretrained model from: {strategy_select['ckpt']}")

                # Load the pretrained model from strategy checkpoint
                pretrained_checkpoint = torch.load(strategy_select['ckpt'], map_location=DEVICE, weights_only=False)

                # Build fresh model for fine-tuning
                model = nn.Sequential(build_encoder(CONFIG), ClassificationHead(CONFIG['embed_dim'], n_classes)).to(
                    DEVICE)

                # Load pretrained weights into encoder using the proper function
                load_pretrained_encoder(model[0], strategy_select['ckpt'])

                print("Pretrained encoder loaded successfully")

                # FREEZE all parameters except LoRA and classification head
                freeze_lora_params(model[0])  # Freeze encoder parameters except LoRA
                # Classification head should remain trainable
                for param in model[1].parameters():
                    param.requires_grad = True

                # Debug: Check if LoRA parameters exist after freezing
                lora_params = [name for name, _ in model[0].named_parameters() if 'lora_' in name]
                trainable_lora_params = [name for name, param in model[0].named_parameters() if
                                         'lora_' in name and param.requires_grad]
                print(f"LoRA parameters found: {len(lora_params)}")
                print(f"Trainable LoRA parameters: {len(trainable_lora_params)}")
                if len(lora_params) == 0:
                    print("WARNING: No LoRA parameters found! The model might not be using LoRA.")
                if len(trainable_lora_params) == 0:
                    print("WARNING: No trainable LoRA parameters! LoRA might not be working.")

                # Debug: Check LoRA parameter values to verify they're loaded
                if len(lora_params) > 0:
                    sample_lora_param = next(p for n, p in model[0].named_parameters() if 'lora_' in n)
                    print(
                        f"Sample LoRA parameter mean: {sample_lora_param.mean().item():.6f}, std: {sample_lora_param.std().item():.6f}")
                    print(f"Strategy: {strategy_select['name']} - LoRA should be different for different strategies!")

                # Initialize fresh optimizer and scheduler for fine-tuning
                optimizer = AdamW([
                    {'params': [p for n, p in model[0].named_parameters() if p.requires_grad],
                     'weight_decay': CONFIG['weight_decay'], 'lr': CONFIG['lr']},  # LoRA LR
                    {'params': model[1].parameters(), 'weight_decay': 0.0, 'lr': CONFIG['lr'] * 3}
                    # Higher LR for classification head
                ])
                # Use warmup + cosine annealing to help escape local optimum
                from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
                warmup_epochs = 1
                main_epochs = CONFIG['epochs'] - warmup_epochs
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
                cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs)
                train_losses = []
                start_epoch = 1
                last_time_ckpt = time.time()

            fold_start_time = time.time()
            for epoch in range(start_epoch, CONFIG['epochs'] + 1):
                model.train()
                train_loss = 0
                n_samples = 0

                # Debug: Check if LoRA parameters are being updated
                if epoch == 1:
                    print("Checking LoRA parameter updates...")
                    lora_params_before = {}
                    for name, param in model[0].named_parameters():
                        if 'lora_' in name:
                            lora_params_before[name] = param.data.clone()

                    # Track classification head initial weights
                    head_weight_before = model[1].head.weight.data.clone()
                    head_bias_before = model[1].head.bias.data.clone()

                # Debug: Print which parameters are being trained
                if epoch == 1:
                    print("Trainable parameters (requires_grad=True):")
                    lora_param_count = 0
                    total_params = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            print(f"  {name} | shape: {tuple(param.shape)}")
                            if 'lora_' in name:
                                lora_param_count += 1
                            total_params += param.numel()
                    print(f"Total LoRA parameters being trained: {lora_param_count}")
                    print(f"Total trainable parameters: {total_params:,}")

                    # Debug: Check if any parameters require gradients
                    if total_params == 0:
                        print("WARNING: No parameters require gradients! This will cause training to fail.")
                        print("Checking all parameters:")
                        for name, param in model.named_parameters():
                            print(f"  {name}: requires_grad={param.requires_grad}")
                        raise RuntimeError("No trainable parameters found!")

                    # Debug: Print classification head initial weights
                    print("Classification head initial weights:")
                    print(
                        f"  Weight mean: {model[1].head.weight.mean().item():.4f}, std: {model[1].head.weight.std().item():.4f}")
                    print(f"  Bias: {model[1].head.bias.data.cpu().numpy()}")

                    # Debug: Check dataset sizes and class distribution
                    print(f"Dataset sizes - Train: {len(train_ds)}, Test: {len(test_ds)}")
                    # Use a separate loader with num_workers=0 to avoid affecting tracking
                    train_loader_debug = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
                    test_loader_debug = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

                    train_labels = []
                    for _, labels in train_loader_debug:
                        train_labels.extend(labels.numpy())
                    train_labels = np.array(train_labels)

                    test_labels = []
                    for _, labels in test_loader_debug:
                        test_labels.extend(labels.numpy())
                    test_labels = np.array(test_labels)

                    print(f"Train class distribution: {np.bincount(train_labels)}")
                    print(f"Test class distribution: {np.bincount(test_labels)}")

                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()

                    # Use regular forward pass (gradient checkpointing was causing issues)
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    loss.backward()

                    # Debug: Print logits and loss for first few batches in first epoch
                    if epoch == 1 and n_samples < 10:
                        print(
                            f"Batch {n_samples}: logits={logits.detach().cpu().numpy()}, labels={labels.cpu().numpy()}, loss={loss.item():.4f}")

                    # Debug: Check gradient norms
                    if epoch == 1 and n_samples == 0:
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        print(f"Gradient norm: {total_norm:.6f}")

                    # Add gradient clipping to prevent oscillation (disabled for debugging)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                    optimizer.step()
                    batch_size = imgs.size(0)
                    train_loss += loss.item() * batch_size
                    n_samples += batch_size

                # Debug: Check LoRA parameter changes after first epoch
                if epoch == 1:
                    total_lora_change = 0.0
                    for name, param in model[0].named_parameters():
                        if 'lora_' in name:
                            change = torch.norm(param.data - lora_params_before[name]).item()
                            total_lora_change += change
                    print(f"Total LoRA change after first epoch: {total_lora_change:.6f}")

                    # Debug: Check classification head changes
                    head_weight_change = torch.norm(model[1].head.weight.data - head_weight_before).item()
                    head_bias_change = torch.norm(model[1].head.bias.data - head_bias_before).item()
                    print(f"Head weight change: {head_weight_change:.6f}, Head bias change: {head_bias_change:.6f}")

                    # Log parameter changes to W&B (only if not in sweep mode)
                    if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                        wandb.log({
                            "head_weight_change": head_weight_change,
                            "head_bias_change": head_bias_change,
                            "total_lora_change": total_lora_change,
                        })

                train_loss /= n_samples if n_samples > 0 else 1
                train_losses.append(train_loss)
                lora_lr = optimizer.param_groups[0]['lr']

                # Evaluate on train set to track learning progress
                model.eval()
                train_labels = []
                train_probs = []
                with torch.no_grad():
                    for imgs, labels in train_loader:
                        imgs = imgs.to(DEVICE)
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        train_probs.append(probs)
                        train_labels.append(labels.numpy())
                train_probs_np = np.concatenate(train_probs, axis=0)
                train_labels_np = np.concatenate(train_labels, axis=0)

                # Compute train metrics
                train_aucs = []
                train_pr_aucs = []
                for c in range(n_classes):
                    y_true = (train_labels_np == c).astype(int)
                    y_score = train_probs_np[:, c]
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auc = float('nan')
                    try:
                        pr_auc = average_precision_score(y_true, y_score)
                    except ValueError:
                        pr_auc = float('nan')
                    train_aucs.append(auc)
                    train_pr_aucs.append(pr_auc)

                # Print train metrics
                print(f"  Train ROC AUC - Class 0: {train_aucs[0]:.4f}, Class 1: {train_aucs[1]:.4f}")
                print(f"  Train PR AUC - Class 0: {train_pr_aucs[0]:.4f}, Class 1: {train_pr_aucs[1]:.4f}")

                # Log train metrics to W&B (only if not in sweep mode)
                if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                    wandb.log({
                        "train_roc_auc_class_0": train_aucs[0],
                        "train_roc_auc_class_1": train_aucs[1],
                        "train_pr_auc_class_0": train_pr_aucs[0],
                        "train_pr_auc_class_1": train_pr_aucs[1],
                    })

                # Early stopping check (stop if validation AUC hasn't improved for 5 epochs)
                if len(train_losses) > 5:
                    # Check if validation performance is improving
                    if 'best_val_auc' not in locals():
                        best_val_auc = 0
                        patience_counter = 0

                    current_val_auc = np.mean(aucs)  # Average AUC across classes
                    if current_val_auc > best_val_auc:
                        best_val_auc = current_val_auc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Also check for overfitting (train improving but test degrading)
                    if len(train_losses) > 10:
                        if train_avg_auc > 0.8 and test_avg_auc < best_val_auc - 0.05:
                            print(f"Early stopping at epoch {epoch} - overfitting detected!")
                            print(
                                f"Train AUC: {train_avg_auc:.4f}, Test AUC: {test_avg_auc:.4f}, Best Test AUC: {best_val_auc:.4f}")
                            break

                    if patience_counter >= 5:
                        print(f"Early stopping at epoch {epoch} - validation AUC not improving for 5 epochs")
                        print(f"Best validation AUC: {best_val_auc:.4f}")
                        break
                head_lr = optimizer.param_groups[1]['lr']
                print(
                    f"{strategy_select['name']} Fold {fold} Epoch {epoch} Train Loss: {train_loss:.4f} LoRA_LR: {lora_lr:.2e} Head_LR: {head_lr:.2e}")

                # Log to W&B (only if not in sweep mode)
                if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "lora_lr": lora_lr,
                        "head_lr": head_lr,
                        "fold": fold,
                        "strategy": strategy_select['name']
                    })

                # Step the appropriate scheduler based on epoch
                if epoch <= warmup_epochs:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                # Clean up memory after each epoch
                cleanup_memory()

                # Evaluate on test set after each epoch
                model.eval()  # Set to evaluation mode
                all_labels = []
                all_probs = []
                with torch.no_grad():  # Disable gradient computation
                    for imgs, labels in test_loader:
                        imgs = imgs.to(DEVICE)
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        all_labels.append(labels.numpy())
                all_probs_np = np.concatenate(all_probs, axis=0)
                all_labels_np = np.concatenate(all_labels, axis=0)

                # Compute metrics
                aucs = []
                pr_aucs = []
                f1s = []
                conf_mat = confusion_matrix(all_labels_np, np.argmax(all_probs_np, axis=1),
                                            labels=list(range(n_classes)))
                for c in range(n_classes):
                    y_true = (all_labels_np == c).astype(int)
                    y_score = all_probs_np[:, c]
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auc = float('nan')
                    try:
                        pr_auc = average_precision_score(y_true, y_score)
                    except ValueError:
                        pr_auc = float('nan')
                    try:
                        f1 = f1_score(y_true, np.argmax(all_probs_np, axis=1) == c)
                    except ValueError:
                        f1 = float('nan')
                    aucs.append(auc)
                    pr_aucs.append(pr_auc)
                    f1s.append(f1)

                # Print ROC AUC for each class
                print(f"  Test ROC AUC - Class 0: {aucs[0]:.4f}, Class 1: {aucs[1]:.4f}")
                print(f"  Test PR AUC - Class 0: {pr_aucs[0]:.4f}, Class 1: {pr_aucs[1]:.4f}")
                print(f"  Test F1 - Class 0: {f1s[0]:.4f}, Class 1: {f1s[1]:.4f}")

                # Compare train vs test performance
                train_avg_auc = np.mean(train_aucs)
                test_avg_auc = np.mean(aucs)
                train_avg_pr_auc = np.mean(train_pr_aucs)
                test_avg_pr_auc = np.mean(pr_aucs)

                print(f"  Performance Summary:")
                print(f"    Train Avg ROC AUC: {train_avg_auc:.4f}, Test Avg ROC AUC: {test_avg_auc:.4f}")
                print(f"    Train Avg PR AUC: {train_avg_pr_auc:.4f}, Test Avg PR AUC: {test_avg_pr_auc:.4f}")

                # Learning diagnosis
                if train_avg_auc < 0.6:
                    print(
                        f"    ??  WARNING: Train ROC AUC is very low ({train_avg_auc:.4f}) - model may not be learning!")
                elif train_avg_auc > 0.8 and test_avg_auc < 0.6:
                    print(
                        f"    ??  WARNING: Train ROC AUC ({train_avg_auc:.4f}) >> Test ROC AUC ({test_avg_auc:.4f}) - overfitting!")
                elif train_avg_auc > 0.7 and test_avg_auc > 0.7:
                    print(f"    ? GOOD: Both train and test performance are good!")
                else:
                    print(f"    ? Learning in progress...")

                # Debug: Show prediction distributions and class balance
                predictions = np.argmax(all_probs_np, axis=1)
                print(f"  Test Predictions: Class 0: {np.sum(predictions == 0)}, Class 1: {np.sum(predictions == 1)}")
                print(
                    f"  Test True Labels: Class 0: {np.sum(all_labels_np == 0)}, Class 1: {np.sum(all_labels_np == 1)}")
                print(
                    f"  Test Probabilities - Class 0 mean: {np.mean(all_probs_np[:, 0]):.4f}, Class 1 mean: {np.mean(all_probs_np[:, 1]):.4f}")

                # Debug: Show probability distribution
                prob_diffs = all_probs_np[:, 1] - all_probs_np[:, 0]  # Class 1 - Class 0
                print(f"  Probability differences (Class 1 - Class 0):")
                print(f"    Mean: {np.mean(prob_diffs):.4f}, Std: {np.std(prob_diffs):.4f}")
                print(f"    Min: {np.min(prob_diffs):.4f}, Max: {np.max(prob_diffs):.4f}")
                print(f"    Samples with Class 1 > Class 0: {np.sum(prob_diffs > 0)}/{len(prob_diffs)}")

                # Log test metrics to W&B (only if not in sweep mode)
                if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                    wandb.log({
                        "test_roc_auc_class_0": aucs[0],
                        "test_roc_auc_class_1": aucs[1],
                        "test_pr_auc_class_0": pr_aucs[0],
                        "test_pr_auc_class_1": pr_aucs[1],
                        "test_f1_class_0": f1s[0],
                        "test_f1_class_1": f1s[1],
                        "train_avg_roc_auc": train_avg_auc,
                        "test_avg_roc_auc": test_avg_auc,
                        "train_avg_pr_auc": train_avg_pr_auc,
                        "test_avg_pr_auc": test_avg_pr_auc,
                        "prob_std": np.std(prob_diffs),
                        "prob_mean": np.mean(prob_diffs),
                        "class_0_predictions": np.sum(predictions == 0),
                        "class_1_predictions": np.sum(predictions == 1),
                        "prob_min": np.min(prob_diffs),
                        "prob_max": np.max(prob_diffs),
                        "class_0_prob_mean": np.mean(all_probs_np[:, 0]),
                        "class_1_prob_mean": np.mean(all_probs_np[:, 1]),
                    })

                # Clean up memory after evaluation
                cleanup_memory()

                # Save checkpoint every epoch (with all states and metrics)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                    'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                    'train_losses': train_losses,
                    'aucs': aucs,
                    'pr_aucs': pr_aucs,
                    'f1s': f1s,
                    'confusion_matrix': conf_mat,
                    'all_labels': all_labels_np,
                    'all_probs': all_probs_np
                }, ckpt_path)

                # Time-based checkpointing
                now = time.time()
                if now - last_time_ckpt >= time_ckpt_interval:
                    time_ckpt_path = os.path.join(OUTPUT_DIRS['checkpoints'],
                                                  f"model_{strategy_select['name']}_{train_dataset}_fold{fold}_epoch{epoch}_timechkpt.pth")
                    # Remove old time-based checkpoints for this fold
                    for old_ckpt in glob.glob(time_ckpt_pattern):
                        try:
                            os.remove(old_ckpt)
                        except Exception:
                            pass
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                        'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                        'train_losses': train_losses,
                        'aucs': aucs,
                        'pr_aucs': pr_aucs,
                        'f1s': f1s,
                        'confusion_matrix': conf_mat,
                        'all_labels': all_labels_np,
                        'all_probs': all_probs_np
                    }, time_ckpt_path)
                    print(f"Time-based checkpoint saved at {time_ckpt_path}")
                    last_time_ckpt = now

                # Final evaluation for this fold (repeat to ensure metrics are up to date)
                model.eval()  # Ensure model is in evaluation mode
                all_labels = []
                all_probs = []
                with torch.no_grad():  # Disable gradient computation
                    for imgs, labels in test_loader:
                        imgs = imgs.to(DEVICE)
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        all_labels.append(labels.numpy())
                all_probs_np = np.concatenate(all_probs, axis=0)
                all_labels_np = np.concatenate(all_labels, axis=0)
                aucs = []
                pr_aucs = []
                f1s = []
                conf_mat = confusion_matrix(all_labels_np, np.argmax(all_probs_np, axis=1),
                                            labels=list(range(n_classes)))
                for c in range(n_classes):
                    y_true = (all_labels_np == c).astype(int)
                    y_score = all_probs_np[:, c]
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auc = float('nan')
                    try:
                        pr_auc = average_precision_score(y_true, y_score)
                    except ValueError:
                        pr_auc = float('nan')
                    try:
                        f1 = f1_score(y_true, np.argmax(all_probs_np, axis=1) == c)
                    except ValueError:
                        f1 = float('nan')
                    aucs.append(auc)
                    pr_aucs.append(pr_auc)
                    f1s.append(f1)
                results["train_losses"] = train_losses
                results["aucs"] = aucs
                results["pr_aucs"] = pr_aucs
                results["f1s"] = f1s
                results["confusion_matrix"] = conf_mat.tolist()
                results["all_labels"] = all_labels_np.tolist()
                results["all_probs"] = all_probs_np.tolist()
            # Save ROC AUC plot
            plt.figure(figsize=(6, 4))
            bars = plt.bar([f'class_{c}' for c in range(n_classes)], aucs, color='orange', alpha=0.7)
            plt.ylim(0, 1)
            plt.ylabel('ROC AUC')
            plt.title(f'ROC AUC per Class ({strategy_select["name"]} Fold {fold})')
            for bar, auc in zip(bars, aucs):
                y = max(bar.get_height() + 0.05, 0.05)
                plt.text(bar.get_x() + bar.get_width() / 2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10,
                         fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIRS['images'],
                                     f'roc_auc_{strategy_select["name"]}_{train_dataset}_fold{fold}.png'))
            plt.close()
            # Save loss curve
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Train Loss ({strategy_select["name"]} Fold {fold})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIRS['images'],
                                     f'loss_curve_{strategy_select["name"]}_{train_dataset}_fold{fold}.png'))
            plt.close()
            # Save results
            save_json(results, result_path)

            # Finish W&B run (only if not in sweep mode)
            if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                wandb.finish()
        elif strategy_select['eval_type'] == "knn":
            # Only use encoder, extract features, run KNN
            enc = build_encoder(CONFIG)
            load_pretrained_encoder(enc, strategy_select['ckpt'])
            enc.to(DEVICE)
            freeze_lora_params(enc)
            aucs, all_labels, all_probs = knn_evaluate(enc, train_loader, test_loader, n_classes, DEVICE, k=5)
            results["aucs"] = aucs
            results["all_labels"] = all_labels.tolist()
            results["all_probs"] = all_probs.tolist()
            # Save ROC AUC plot
            plt.figure(figsize=(6, 4))
            bars = plt.bar([f'class_{c}' for c in range(n_classes)], aucs, color='blue', alpha=0.7)
            plt.ylim(0, 1)
            plt.ylabel('ROC AUC')
            plt.title(f'KNN ROC AUC per Class ({strategy_select["name"]} Fold {fold})')
            for bar, auc in zip(bars, aucs):
                y = max(bar.get_height() + 0.05, 0.05)
                plt.text(bar.get_x() + bar.get_width() / 2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10,
                         fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIRS['images'],
                                     f'roc_auc_{strategy_select["name"]}_{train_dataset}_fold{fold}.png'))
            plt.close()
            # Save results
            save_json(results, result_path)

            # Print image loading summary for KNN
            print(f"\nImage Loading Summary (KNN):")
            train_total, train_black = count_images_in_dataset(train_ds, "train")
            test_total, test_black = count_images_in_dataset(test_ds, "test")
            print(f"Train dataset - Total checked: {train_total}, Black images: {train_black}")
            print(f"Test dataset - Total checked: {test_total}, Black images: {test_black}")

            # Finish W&B run (only if not in sweep mode)
            if not sweep_mode and not os.environ.get('WANDB_SWEEP_MODE'):
                wandb.finish()
        else:
            raise NotImplementedError(f"Unknown eval_type: {strategy_select['eval_type']}")

    # After processing all folds for this strategy, break to avoid running more strategies
    # break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LoRA Fine-tuning with Cross-dataset Evaluation')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--strategy', type=str,
                        choices=['retina_feature_finetune', 'imagenet_finetune', 'scratch'],
                        help='Strategy to use')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--sweep_mode', action='store_true', help='Run in sweep mode')

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_from_file = json.load(f)
            CONFIG.update(config_from_file)

    # Set strategy if provided
    if args.strategy:
        strategy_name = args.strategy
    else:
        strategy_name = None

    # Set fold if provided
    if args.fold is not None:
        fold_number = args.fold
    else:
        fold_number = 0

    # Run main function
    main(strategy_name=strategy_name, fold=fold_number, sweep_mode=args.sweep_mode)