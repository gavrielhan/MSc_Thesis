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
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import glob

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

# Manual settings for inference
STRATEGY_NAME = "retina_feature_finetune"  # Change this to your strategy
TRAIN_DATASET = "idrid"  # Change this to your training dataset
FOLD_NUMBER = 2  # Change this to your fold number
EVAL_DATASET = "idrid"  # Change this to dataset you want to evaluate on

# Manual checkpoint path - specify the exact checkpoint file you want to use
CHECKPOINT_PATH = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/model_retina_feature_finetune_idrid_fold2.pth"
# Alternative: leave empty to auto-construct from strategy/dataset/fold
# CHECKPOINT_PATH = ""

CONFIG = {
    'img_size': (616, 616),
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.2,
    'batch_size': 8,  # Larger batch size for faster inference
    'num_workers': 4,  # More workers for faster data loading
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets',
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- Data Utilities ----------
class PapilaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

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
        except:
            img = Image.new('RGB', CONFIG['img_size'], 'black')
        img = self.transform(img)
        return img, label


class IDRIDDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image name']
        label = int(row[self.label_col])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', CONFIG['img_size'], 'black')
        img = self.transform(img)
        return img, label


class MessidorDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        label = int(row[self.label_col])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', CONFIG['img_size'], 'black')
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
    return enc


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


# ---------- Evaluation Utilities ----------
def evaluate_model(model, dataloader, n_classes, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
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
    conf_mat = confusion_matrix(all_labels_np, np.argmax(all_probs_np, axis=1), labels=list(range(n_classes)))

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

    return aucs, pr_aucs, f1s, conf_mat, all_labels_np, all_probs_np


def plot_roc_auc(aucs, dataset_name, strategy_name, fold):
    plt.figure(figsize=(6, 4))
    bars = plt.bar([f'class_{c}' for c in range(len(aucs))], aucs, color='orange', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title(f'ROC AUC per Class ({strategy_name} Fold {fold} on {dataset_name})')
    for bar, auc in zip(bars, aucs):
        y = max(bar.get_height() + 0.05, 0.05)
        plt.text(bar.get_x() + bar.get_width() / 2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRS['images'], f'inference_roc_auc_{strategy_name}_{dataset_name}_fold{fold}.png'))
    plt.close()


# ---------- Main Inference Function ----------
def main():
    print(f"=== Fast Inference Script ===")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Training Dataset: {TRAIN_DATASET}")
    print(f"Fold: {FOLD_NUMBER}")
    print(f"Evaluation Dataset: {EVAL_DATASET}")
    print(f"Device: {DEVICE}")

    # Load checkpoint - use manual path if provided, otherwise auto-construct
    if CHECKPOINT_PATH:
        ckpt_path = CHECKPOINT_PATH
        print(f"Using manual checkpoint path: {ckpt_path}")
    else:
        ckpt_path = os.path.join(OUTPUT_DIRS['checkpoints'],
                                 f"model_{STRATEGY_NAME}_{TRAIN_DATASET}_fold{FOLD_NUMBER}.pth")
        print(f"Using auto-constructed checkpoint path: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Determine number of classes based on dataset
    if EVAL_DATASET == 'idrid':
        n_classes = 2
    elif EVAL_DATASET == 'papila':
        n_classes = 3
    elif EVAL_DATASET == 'messidor':
        n_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {EVAL_DATASET}")

    # Build model
    model = nn.Sequential(build_encoder(CONFIG), ClassificationHead(CONFIG['embed_dim'], n_classes)).to(DEVICE)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully!")

    # Prepare evaluation dataset - ONLY test set
    trsf = build_transforms()

    if EVAL_DATASET == 'idrid':
        root = os.path.join(CONFIG['external_root'], 'IDRID', 'B. Disease Grading')
        img_dir = os.path.join(root, '1. Original Images', 'a. Training Set')
        label_csv = os.path.join(root, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        df = pd.read_csv(label_csv)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        for col in ['Retinopathy grade', 'Risk of macular edema']:
            if col in df.columns:
                df[col] = (df[col] > 0).astype(int)
        label_col = 'Retinopathy grade'

        # Create test split for the specific fold
        indices = np.arange(len(df))
        rng = np.random.default_rng(42 + FOLD_NUMBER)  # Use fold-specific seed
        shuffled = rng.permutation(indices)
        split = int(0.8 * len(df))
        test_idx = shuffled[split:]  # Only test set
        df_test = df.iloc[test_idx].reset_index(drop=True)
        dataset = IDRIDDataset(df_test, img_dir, trsf, label_col=label_col)

    elif EVAL_DATASET == 'papila':
        root = os.path.join(CONFIG['external_root'], 'PAPILA',
                            'PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0')
        img_dir = os.path.join(root, 'FundusImages')
        kfold_dir = os.path.join(root, 'HelpCode', 'kfold', 'Test 1')
        test_xlsx = os.path.join(kfold_dir, 'Test', f'test_1_test_index_fold_{FOLD_NUMBER}.xlsx')
        df_test = pd.read_excel(test_xlsx)
        dataset = PapilaDataset(df_test, img_dir, trsf)

    elif EVAL_DATASET == 'messidor':
        root = os.path.join(CONFIG['external_root'], 'messidor')
        img_dir = os.path.join(root, 'IMAGES')
        data_csv = os.path.join(root, 'messidor_data.csv')
        patient_csv = os.path.join(root, 'messidor-2.csv')
        df = pd.read_csv(data_csv)
        df['adjudicated_dr_grade'] = (df['adjudicated_dr_grade'] != 0).astype(int)
        df_pat = pd.read_csv(patient_csv)
        df_pat.columns = df_pat.columns.str.strip()
        col = df_pat.columns[0]
        patient_groups = []
        for _, row in df_pat.iterrows():
            imgs = [img.strip() for img in str(row[col]).split(';') if img.strip() and img.strip().lower() != 'nan']
            if imgs:
                patient_groups.append(imgs)
        n_pat = len(patient_groups)
        rng = np.random.default_rng(42 + FOLD_NUMBER)
        shuffled_groups = patient_groups.copy()
        rng.shuffle(shuffled_groups)
        split = int(0.8 * n_pat)
        test_pat = shuffled_groups[split:]  # Only test patients
        test_imgs = set([img for group in test_pat for img in group])
        df_test = df[df['image_id'].isin(test_imgs)].reset_index(drop=True)
        label_col = 'adjudicated_dr_grade'
        dataset = MessidorDataset(df_test, img_dir, trsf, label_col=label_col)

    else:
        raise ValueError(f"Unknown evaluation dataset: {EVAL_DATASET}")

    # Create dataloader with optimized settings
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    print(f"Evaluating on {len(dataset)} test samples...")

    # Run inference
    aucs, pr_aucs, f1s, conf_mat, all_labels, all_probs = evaluate_model(model, dataloader, n_classes, DEVICE)

    # Print results
    print(f"\n=== Results for {STRATEGY_NAME} Fold {FOLD_NUMBER} on {EVAL_DATASET} ===")
    for c in range(n_classes):
        print(f"Class {c}:")
        print(f"  ROC AUC: {aucs[c]:.4f}")
        print(f"  PR AUC: {pr_aucs[c]:.4f}")
        print(f"  F1 Score: {f1s[c]:.4f}")

    print(f"\nConfusion Matrix:")
    print(conf_mat)

    # Save results
    results = {
        "strategy": STRATEGY_NAME,
        "train_dataset": TRAIN_DATASET,
        "eval_dataset": EVAL_DATASET,
        "fold": FOLD_NUMBER,
        "aucs": aucs,
        "pr_aucs": pr_aucs,
        "f1s": f1s,
        "confusion_matrix": conf_mat.tolist(),
        "all_labels": all_labels.tolist(),
        "all_probs": all_probs.tolist()
    }

    result_path = os.path.join(OUTPUT_DIRS['results'],
                               f"inference_results_{STRATEGY_NAME}_{EVAL_DATASET}_fold{FOLD_NUMBER}.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {result_path}")

    # Plot ROC AUC
    plot_roc_auc(aucs, EVAL_DATASET, STRATEGY_NAME, FOLD_NUMBER)
    print(f"ROC AUC plot saved to: {OUTPUT_DIRS['images']}")


if __name__ == '__main__':
    main()