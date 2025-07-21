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

# IJepa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer

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
    'pretrained_ckpt': os.path.expanduser('~/checkpoint_pretrain.pth'),
    'n_classes': 3,
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets/PAPILA/PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0',
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
        pid = row['Patient ID']  # Use correct column name with space
        eye = row['eyeID']
        label = int(row['Diagnosis'])
        # Compose filename: e.g., RET293OS.jpg
        img_name = f"RET{pid}{eye}.jpg"
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

def load_pretrained_encoder(enc, config):
    ckpt = config.get('pretrained_ckpt', None)
    if ckpt and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=DEVICE)
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

# ---------- Main Training/Eval ----------
CHECKPOINT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_papila_finetune.pth'
MAX_TRAIN_SECONDS = 11.5 * 3600  # 11.5 hours in seconds

def main():
    # Paths
    root = CONFIG['external_root']
    img_dir = os.path.join(root, 'FundusImages')
    kfold_dir = os.path.join(root, 'HelpCode', 'kfold', 'Test 1')

    start_epoch = 1
    elapsed_time = 0


    # --- Clinical Data Baseline ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    clinical_dir = os.path.join(root, "ClinicalData")
    od_path = os.path.join(clinical_dir, 'patient_data_od.xlsx')
    os_path = os.path.join(clinical_dir, 'patient_data_os.xlsx')
    df_od = pd.read_excel(od_path, header=1)
    df_os = pd.read_excel(os_path, header=1)
    df_od = df_od.rename(columns={str(df_od.columns[0]): 'ID'})
    df_os = df_os.rename(columns={str(df_os.columns[0]): 'ID'})
    df_od = df_od.iloc[1:].reset_index(drop=True)
    df_os = df_os.iloc[1:].reset_index(drop=True)
    df_od['eyeID'] = 'OD'
    df_os['eyeID'] = 'OS'
    df_clin = pd.concat([df_od, df_os], ignore_index=True)
    df_clin = df_clin.dropna(subset=['Age', 'Gender', 'Diagnosis'])
    n_folds = 5
    roc_aucs_clin_folds = []
    for fold in range(n_folds):
        print(f"\n=== Clinical Baseline Fold {fold} ===")
        train_xlsx = os.path.join(kfold_dir, 'Train', f'test_1_train_index_fold_{fold}.xlsx')
        test_xlsx = os.path.join(kfold_dir, 'Test', f'test_1_test_index_fold_{fold}.xlsx')
        df_train = pd.read_excel(train_xlsx)
        df_test = pd.read_excel(test_xlsx)
        def get_keys(df):
            return set(zip(df['Patient ID'].astype(str), df['eyeID'].astype(str)))
        train_keys = get_keys(df_train)
        test_keys = get_keys(df_test)
        clin_keys = list(zip(df_clin['ID'].astype(str).str.lstrip('#'), df_clin['eyeID'].astype(str)))
        is_train = np.array([k in train_keys for k in clin_keys])
        is_test = np.array([k in test_keys for k in clin_keys])
        X = df_clin[['Age', 'Gender']].values
        y = df_clin['Diagnosis'].astype(int).values
        train_idx = np.where(is_train)[0]
        test_idx = np.where(is_test)[0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(multi_class='ovr', max_iter=1000)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        roc_aucs_clin = {}
        for c in range(CONFIG['n_classes']):
            y_true = (y_test == c).astype(int)
            y_score = y_proba[:, c]
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float('nan')
            roc_aucs_clin[f'class_{c}'] = auc
        print("Clinical Baseline ROC AUCs:")
        for c, auc in roc_aucs_clin.items():
            print(f"ROC AUC for {c}: {auc:.4f}")
        roc_aucs_clin_folds.append(roc_aucs_clin)

    # Average ROC AUCs across folds
    avg_roc_aucs_clin = {f'class_{c}': np.nanmean([fold[f'class_{c}'] for fold in roc_aucs_clin_folds]) for c in range(CONFIG['n_classes'])}
    print("\nAveraged ROC AUCs (Clinical Baseline):")
    for c, auc in avg_roc_aucs_clin.items():
        print(f"ROC AUC for {c}: {auc:.4f}")
    # Plot averaged ROC AUCs
    import matplotlib.pyplot as plt
    class_labels = list(avg_roc_aucs_clin.keys())
    auc_values_clin = [avg_roc_aucs_clin[c] for c in class_labels]
    auc_folds = np.array([[fold[c] for c in class_labels] for fold in roc_aucs_clin_folds])
    auc_stds = np.nanstd(auc_folds, axis=0)
    plt.figure(figsize=(6,4))
    bars = plt.bar(class_labels, auc_values_clin, color='salmon', yerr=auc_stds, capsize=8, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title('Averaged ROC AUC per Class (Clinical Data)')
    # Overlay black points for each fold
    for i, label in enumerate(class_labels):
        y_folds = auc_folds[:, i]
        x = np.full_like(y_folds, i, dtype=float)
        plt.scatter(x, y_folds, color='black', zorder=10)
    # Place value labels higher for visibility
    for bar, auc in zip(bars, auc_values_clin):
        y = max(bar.get_height() + 0.1, 0.05)
        plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('roc_auc_per_class_clinical_avg.png')
    plt.close()

    # --- Main Fine-Tuning Task (Deep Model) Cross-Validation ---
    import time
    CHECKPOINT_CV_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_papila_finetune_cv.pth'
    MAX_TRAIN_SECONDS = 11.5 * 3600
    n_folds = 5
    roc_aucs_model_folds = []
    start_fold = 0
    start_epoch = 1
    elapsed_time = 0
    # Resume logic
    if os.path.isfile(CHECKPOINT_CV_PATH):
        checkpoint = torch.load(CHECKPOINT_CV_PATH, map_location=DEVICE, weights_only=False)
        start_fold = checkpoint.get('fold', 0)
        start_epoch = checkpoint.get('epoch', 1)
        elapsed_time = 0
        roc_aucs_model_folds = checkpoint.get('roc_aucs_model_folds', [])
        print(f"Resuming from checkpoint: fold {start_fold}, epoch {start_epoch}, elapsed_time {elapsed_time:.2f}s")
    total_start_time = time.time()
    for fold in range(start_fold, n_folds):
        print(f"\n=== Deep Model Fold {fold} ===")
        train_xlsx = os.path.join(kfold_dir, 'Train', f'test_1_train_index_fold_{fold}.xlsx')
        test_xlsx = os.path.join(kfold_dir, 'Test', f'test_1_test_index_fold_{fold}.xlsx')
        df_train = pd.read_excel(train_xlsx)
        df_test = pd.read_excel(test_xlsx)
        trsf = build_transforms()
        train_ds = PapilaDataset(df_train, img_dir, trsf)
        test_ds = PapilaDataset(df_test, img_dir, trsf)
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        enc = build_encoder(CONFIG)
        load_pretrained_encoder(enc, CONFIG)
        enc.to(DEVICE)
        freeze_lora_params(enc)
        head = ClassificationHead(CONFIG['embed_dim'], CONFIG['n_classes']).to(DEVICE)
        model = nn.Sequential(enc, head)
        optimizer = AdamW([
            {'params': [p for n, p in enc.named_parameters() if p.requires_grad], 'weight_decay': CONFIG['weight_decay']},
            {'params': head.parameters(), 'weight_decay': 0.0}
        ], lr=CONFIG['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        criterion = nn.CrossEntropyLoss()
        for epoch in range(start_epoch if fold == start_fold else 1, CONFIG['epochs']+1):
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
                # Check for time limit
                total_elapsed = elapsed_time + (time.time() - total_start_time)
                if total_elapsed >= MAX_TRAIN_SECONDS:
                    torch.save({
                        'fold': fold,
                        'epoch': epoch,
                        'enc': enc.state_dict(),
                        'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'elapsed_time': total_elapsed,
                        'roc_aucs_model_folds': roc_aucs_model_folds
                    }, CHECKPOINT_CV_PATH)
                    print(f"Reached max training time. Checkpoint saved to {CHECKPOINT_CV_PATH}. Exiting.")
                    return
            train_loss /= n_samples if n_samples > 0 else 1
            print(f"Fold {fold} Epoch {epoch} Train Loss: {train_loss:.4f}")
            scheduler.step()
        # Evaluation
        model.eval()
        all_labels = []
        all_probs = []
        test_loss = 0
        n_test_samples = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                test_loss += loss.item() * imgs.size(0)
                n_test_samples += imgs.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        test_loss /= n_test_samples if n_test_samples > 0 else 1
        print(f"Fold {fold} Test Loss: {test_loss:.4f}")
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        roc_aucs = {}
        for c in range(CONFIG['n_classes']):
            y_true = (all_labels == c).astype(int)
            y_score = all_probs[:, c]
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float('nan')
            roc_aucs[f'class_{c}'] = auc
        print("Deep Model ROC AUCs:")
        for c, auc in roc_aucs.items():
            print(f"ROC AUC for {c}: {auc:.4f}")
        roc_aucs_model_folds.append(roc_aucs)
        # Reset start_epoch for next fold
        start_epoch = 1
        # Reset elapsed_time for next fold
        elapsed_time = 0
    # Average ROC AUCs across folds for deep model
    avg_roc_aucs_model = {f'class_{c}': np.nanmean([fold[f'class_{c}'] for fold in roc_aucs_model_folds]) for c in range(CONFIG['n_classes'])}
    auc_folds_model = np.array([[fold[c] for c in avg_roc_aucs_model.keys()] for fold in roc_aucs_model_folds])
    auc_stds_model = np.nanstd(auc_folds_model, axis=0)
    print("\nAveraged ROC AUCs (Deep Model):")
    for c, auc in avg_roc_aucs_model.items():
        print(f"ROC AUC for {c}: {auc:.4f}")
    # Plot averaged ROC AUCs for deep model
    import matplotlib.pyplot as plt
    class_labels = list(avg_roc_aucs_model.keys())
    auc_values_model = [avg_roc_aucs_model[c] for c in class_labels]
    plt.figure(figsize=(6,4))
    bars = plt.bar(class_labels, auc_values_model, color='skyblue', yerr=auc_stds_model, capsize=8, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title('Averaged ROC AUC per Class (Deep Model)')
    # Overlay black points for each fold
    for i, label in enumerate(class_labels):
        y_folds = auc_folds_model[:, i]
        x = np.full_like(y_folds, i, dtype=float)
        plt.scatter(x, y_folds, color='black', zorder=10)
    # Place value labels higher for visibility
    for bar, auc in zip(bars, auc_values_model):
        y = max(bar.get_height() + 0.1, 0.05)
        plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('roc_auc_per_class_model_avg.png')
    plt.close()

    # --- KNN on Retina Fine-Tuned Representations ---
    from sklearn.neighbors import KNeighborsClassifier
    KNN_CKPT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth'
    n_folds = 5
    roc_aucs_knn_folds = []
    for fold in range(n_folds):
        print(f"\n=== KNN on Retina FT Features Fold {fold} ===")
        train_xlsx = os.path.join(kfold_dir, 'Train', f'test_1_train_index_fold_{fold}.xlsx')
        test_xlsx = os.path.join(kfold_dir, 'Test', f'test_1_test_index_fold_{fold}.xlsx')
        df_train = pd.read_excel(train_xlsx)
        df_test = pd.read_excel(test_xlsx)
        trsf = build_transforms()
        train_ds = PapilaDataset(df_train, img_dir, trsf)
        test_ds = PapilaDataset(df_test, img_dir, trsf)
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        # Load encoder from retina_ft checkpoint
        enc = build_encoder(CONFIG)
        ckpt = torch.load(KNN_CKPT_PATH, map_location=DEVICE)
        enc.load_state_dict(ckpt['enc'])
        enc.to(DEVICE)
        enc.eval()
        # Extract features
        def extract_features(loader):
            features = []
            labels = []
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = imgs.to(DEVICE)
                    # Get CLS token (assume enc returns [B, N+1, D], take [:,0,:])
                    feats = enc(imgs)
                    if feats.ndim == 3:
                        feats = feats[:, 0, :]
                    features.append(feats.cpu().numpy())
                    labels.append(lbls.numpy())
            return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)
        X_train, y_train = extract_features(train_loader)
        X_test, y_test = extract_features(test_loader)
        # KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_proba = knn.predict_proba(X_test)
        roc_aucs_knn = {}
        for c in range(CONFIG['n_classes']):
            y_true = (y_test == c).astype(int)
            y_score = y_proba[:, c]
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float('nan')
            roc_aucs_knn[f'class_{c}'] = auc
        print("KNN ROC AUCs:")
        for c, auc in roc_aucs_knn.items():
            print(f"ROC AUC for {c}: {auc:.4f}")
        roc_aucs_knn_folds.append(roc_aucs_knn)
    # Average ROC AUCs across folds for KNN
    avg_roc_aucs_knn = {f'class_{c}': np.nanmean([fold[f'class_{c}'] for fold in roc_aucs_knn_folds]) for c in range(CONFIG['n_classes'])}
    auc_folds_knn = np.array([[fold[c] for c in avg_roc_aucs_knn.keys()] for fold in roc_aucs_knn_folds])
    auc_stds_knn = np.nanstd(auc_folds_knn, axis=0)
    print("\nAveraged ROC AUCs (KNN on Retina FT):")
    for c, auc in avg_roc_aucs_knn.items():
        print(f"ROC AUC for {c}: {auc:.4f}")
    # Plot averaged ROC AUCs for KNN
    import matplotlib.pyplot as plt
    class_labels = list(avg_roc_aucs_knn.keys())
    auc_values_knn = [avg_roc_aucs_knn[c] for c in class_labels]
    plt.figure(figsize=(6,4))
    bars = plt.bar(class_labels, auc_values_knn, color='green', yerr=auc_stds_knn, capsize=8, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title('Averaged ROC AUC per Class (KNN on Retina FT)')
    # Overlay black points for each fold
    for i, label in enumerate(class_labels):
        y_folds = auc_folds_knn[:, i]
        x = np.full_like(y_folds, i, dtype=float)
        plt.scatter(x, y_folds, color='black', zorder=10)
    # Place value labels higher for visibility
    for bar, auc in zip(bars, auc_values_knn):
        y = max(bar.get_height() + 0.1, 0.05)
        plt.text(bar.get_x() + bar.get_width()/2, y, f'{auc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('roc_auc_per_class_knn_avg.png')
    plt.close()

if __name__ == '__main__':
    main()