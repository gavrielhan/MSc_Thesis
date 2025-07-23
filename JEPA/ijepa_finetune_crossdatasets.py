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
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# IJepa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer

# ---------- Output Directories ----------
OUTPUT_DIRS = {
    'checkpoints': "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/",
    'results': os.path.join('outputs', 'results'),
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
    'batch_size': 2,
    'num_workers': 2,
    'lr': 2e-4,
    'weight_decay': 1e-2,
    'epochs': 30,
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
class PapilaDataset(Dataset):
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

class IDRIDDataset(Dataset):
    def __init__(self, df, img_dir, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_name']
        label = int(row[self.label_col])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', CONFIG['img_size'], 'black')
        img = self.transform(img)
        return img, label

class MessidorDataset(Dataset):
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

# Accept checkpoint path directly
def load_pretrained_encoder(enc, ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        enc_state = state.get('enc', state)
        filtered = {k: v for k, v in enc_state.items()
                    if k.startswith('patch_embed.') or k.startswith('blocks.')
                    or k in ('cls_token','pos_embed')}
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 6 and enc.patch_embed.proj.weight.shape[1] == 3:
            filtered['patch_embed.proj.weight'] = w[:, :3]
        enc.load_state_dict(filtered, strict=False)
        logger.info("Pretrained weights loaded into encoder")
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
    plt.figure(figsize=(6,4))
    bars = plt.bar([f'class_{c}' for c in range(n_classes)], auc_values, color='orange', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title(f'ROC AUC per Class ({dataset_name})')
    for bar, auc in zip(bars, auc_values):
        y = max(bar.get_height() + 0.05, 0.05)
        plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'roc_auc_per_class_{dataset_name.lower()}.png')
    plt.close()

# ---------- Strategy Configurations ----------
STRATEGIES = [
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
        "name": "retina_feature_finetune",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth",
        "eval_type": "finetune"
    },
    {
        "name": "retina_feature_knn",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth",
        "eval_type": "knn"
    }
]

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
def main():
    # User selects dataset for fine-tuning and testing
    train_dataset = "idrid"
    test_dataset = "idrid"
    print("Selected dataset for fine-tuning (train): ", train_dataset)
    print("Selected dataset for testing (eval): ", test_dataset)

    trsf = build_transforms()
    n_folds = 5
    seeds = [42, 43, 44, 45, 46]  # For reproducible random splits per fold

    # Prepare dataset splits for k-fold or random splits with different seeds
    if train_dataset == 'idrid':
        root = os.path.join(CONFIG['external_root'], 'IDRID', 'B. Disease Grading')
        img_dir = os.path.join(root, '1. Original Images', 'a. Training Set')
        label_csv = os.path.join(root, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        df = pd.read_csv(label_csv)
        # Binarize both relevant columns
        for col in ['Retinopathy grade', 'Risk of macular edema']:
            df[col] = (df[col] > 0).astype(int)
        label_col = 'Retinopathy grade'
        indices = np.arange(len(df))
        splits = []
        for fold, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(indices)
            split = int(0.8 * len(df))
            train_idx = shuffled[:split]
            test_idx = shuffled[split:]
            splits.append((train_idx, test_idx))
        n_classes_idrid = 2  # Always binary after binarization
    elif train_dataset == 'papila':
        root = os.path.join(CONFIG['external_root'], 'PAPILA', 'PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0')
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
        patient_groups = []
        for _, row in df_pat.iterrows():
            left = str(row['left']) if not pd.isna(row['left']) else None
            right = str(row['right']) if not pd.isna(row['right']) else None
            imgs = []
            if left: imgs.append(left)
            if right: imgs.append(right)
            if imgs:
                patient_groups.append(imgs)
        n_pat = len(patient_groups)
        splits = []
        for fold, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            shuffled_groups = patient_groups.copy()
            rng.shuffle(shuffled_groups)
            split = int(0.8 * n_pat)
            train_pat = shuffled_groups[:split]
            test_pat = shuffled_groups[split:]
            train_imgs = set([img for group in train_pat for img in group])
            test_imgs = set([img for group in test_pat for img in group])
            splits.append((train_imgs, test_imgs, df))
        n_classes_messidor = 2  # Binary after binarization
    else:
        raise NotImplementedError(f"K-fold not implemented for dataset: {train_dataset}")

    for strategy in STRATEGIES:
        if not os.path.isfile(strategy['ckpt']):
            print(f"Checkpoint for strategy '{strategy['name']}' not found at {strategy['ckpt']}. Skipping this strategy.")
            continue
        print(f"\n=== Running strategy: {strategy['name']} ===")
        for fold in range(n_folds):
            result_path = os.path.join(OUTPUT_DIRS['results'], f"results_{strategy['name']}_{train_dataset}_fold{fold}.json")
            ckpt_path = os.path.join(OUTPUT_DIRS['checkpoints'], f"model_{strategy['name']}_{train_dataset}_fold{fold}.pth")
            if os.path.exists(result_path):
                print(f"Skipping {strategy['name']} fold {fold} (already done)")
                continue
            print(f"\n--- Fold {fold} ---")
            # Prepare train/test datasets for this fold
            if train_dataset == 'idrid':
                train_idx, test_idx = splits[fold]
                df_train = df.iloc[train_idx].reset_index(drop=True)
                df_test = df.iloc[test_idx].reset_index(drop=True)
                train_ds = IDRIDDataset(df_train, img_dir, trsf, label_col=label_col)
                test_ds = IDRIDDataset(df_test, img_dir, trsf, label_col=label_col)
                n_classes = n_classes_idrid
            elif train_dataset == 'papila':
                train_xlsx, test_xlsx = splits[fold]
                df_train = pd.read_excel(train_xlsx)
                df_test = pd.read_excel(test_xlsx)
                train_ds = PapilaDataset(df_train, img_dir, trsf)
                test_ds = PapilaDataset(df_test, img_dir, trsf)
                n_classes = n_classes_papila
            elif train_dataset == 'messidor':
                train_imgs, test_imgs, df_all = splits[fold]
                label_col = 'adjudicated_dr_grade'
                df_train = df_all[df_all['image_id'].isin(train_imgs)].reset_index(drop=True)
                df_test = df_all[df_all['image_id'].isin(test_imgs)].reset_index(drop=True)
                train_ds = MessidorDataset(df_train, img_dir, trsf, label_col=label_col)
                test_ds = MessidorDataset(df_test, img_dir, trsf, label_col=label_col)
                n_classes = n_classes_messidor
            else:
                raise NotImplementedError
            train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
            # Build encoder and load checkpoint
            enc = build_encoder(CONFIG)
            # Load checkpoint for this strategy
            load_pretrained_encoder(enc, strategy['ckpt'])
            enc.to(DEVICE)
            freeze_lora_params(enc)
            results = {"fold": fold, "strategy": strategy['name'], "train_dataset": train_dataset, "test_dataset": test_dataset}
            if strategy['eval_type'] == "finetune":
                head = ClassificationHead(CONFIG['embed_dim'], n_classes).to(DEVICE)
                model = nn.Sequential(enc, head)
                optimizer = AdamW([
                    {'params': [p for n, p in enc.named_parameters() if p.requires_grad], 'weight_decay': CONFIG['weight_decay']},
                    {'params': head.parameters(), 'weight_decay': 0.0}
                ], lr=CONFIG['lr'])
                scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
                criterion = nn.CrossEntropyLoss()
                train_losses = []
                for epoch in range(1, CONFIG['epochs']+1):
                    model.train()
                    train_loss = 0
                    n_samples = 0
                    for imgs, labels in train_loader:
                        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        batch_size = imgs.size(0)
                        train_loss += loss.item() * batch_size
                        n_samples += batch_size
                    train_loss /= n_samples if n_samples > 0 else 1
                    train_losses.append(train_loss)
                    print(f"{strategy['name']} Fold {fold} Epoch {epoch} Train Loss: {train_loss:.4f}")
                    scheduler.step()
                    # Save checkpoint every epoch
                    torch.save(model.state_dict(), ckpt_path)
                # Evaluate
                model.eval()
                all_labels = []
                all_probs = []
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs = imgs.to(DEVICE)
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        all_labels.append(labels.numpy())
                all_probs = np.concatenate(all_probs, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                aucs = []
                for c in range(n_classes):
                    y_true = (all_labels == c).astype(int)
                    y_score = all_probs[:, c]
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auc = float('nan')
                    aucs.append(auc)
                results["train_losses"] = train_losses
                results["aucs"] = aucs
                results["all_labels"] = all_labels.tolist()
                results["all_probs"] = all_probs.tolist()
                # Save ROC AUC plot
                plt.figure(figsize=(6,4))
                bars = plt.bar([f'class_{c}' for c in range(n_classes)], aucs, color='orange', alpha=0.7)
                plt.ylim(0, 1)
                plt.ylabel('ROC AUC')
                plt.title(f'ROC AUC per Class ({strategy["name"]} Fold {fold})')
                for bar, auc in zip(bars, aucs):
                    y = max(bar.get_height() + 0.05, 0.05)
                    plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIRS['images'], f'roc_auc_{strategy["name"]}_{train_dataset}_fold{fold}.png'))
                plt.close()
                # Save loss curve
                plt.figure()
                plt.plot(train_losses, label='Train Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Train Loss ({strategy["name"]} Fold {fold})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIRS['images'], f'loss_curve_{strategy["name"]}_{train_dataset}_fold{fold}.png'))
                plt.close()
                # Save results
                save_json(results, result_path)
            elif strategy['eval_type'] == "knn":
                # Only use encoder, extract features, run KNN
                aucs, all_labels, all_probs = knn_evaluate(enc, train_loader, test_loader, n_classes, DEVICE, k=5)
                results["aucs"] = aucs
                results["all_labels"] = all_labels.tolist()
                results["all_probs"] = all_probs.tolist()
                # Save ROC AUC plot
                plt.figure(figsize=(6,4))
                bars = plt.bar([f'class_{c}' for c in range(n_classes)], aucs, color='blue', alpha=0.7)
                plt.ylim(0, 1)
                plt.ylabel('ROC AUC')
                plt.title(f'KNN ROC AUC per Class ({strategy["name"]} Fold {fold})')
                for bar, auc in zip(bars, aucs):
                    y = max(bar.get_height() + 0.05, 0.05)
                    plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIRS['images'], f'roc_auc_{strategy["name"]}_{train_dataset}_fold{fold}.png'))
                plt.close()
                # Save results
                save_json(results, result_path)
            else:
                raise NotImplementedError(f"Unknown eval_type: {strategy['eval_type']}")

if __name__ == '__main__':
    main()