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
    'batch_size': 1,
    'num_workers': 2,
    'lr': 6e-4,
    'weight_decay': 1e-2,
    'epochs': 60,
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

def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# ---------- Main Training/Eval ----------
def main():
    # Paths
    root = CONFIG['external_root']
    img_dir = os.path.join(root, 'FundusImages')
    kfold_dir = os.path.join(root, 'HelpCode', 'kfold', 'Test 1')
    train_xlsx = os.path.join(kfold_dir, 'Train', 'test_1_train_index_fold_4.xlsx')
    test_xlsx = os.path.join(kfold_dir, 'Test', 'test_1_test_index_fold_4.xlsx')
    # Load splits
    df_train = pd.read_excel(train_xlsx)
    df_test = pd.read_excel(test_xlsx)
    # Build datasets
    trsf = build_transforms()
    train_ds = PapilaDataset(df_train, img_dir, trsf)
    test_ds = PapilaDataset(df_test, img_dir, trsf)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    # Build model
    enc = build_encoder(CONFIG)
    load_pretrained_encoder(enc, CONFIG)
    enc.to(DEVICE)
    head = ClassificationHead(CONFIG['embed_dim'], CONFIG['n_classes']).to(DEVICE)
    model = nn.Sequential(enc, head)
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(1, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        n_samples = 0
        for imgs, labels in tqdm(train_loader, desc=f"Train E{epoch}"):
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
        logger.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        scheduler.step()
    # Evaluation
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Compute ROC AUC for each class
    roc_aucs = {}
    for c in range(CONFIG['n_classes']):
        y_true = (all_labels == c).astype(int)
        y_score = all_probs[:, c]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
        roc_aucs[f'class_{c}'] = auc
    for c, auc in roc_aucs.items():
        print(f"ROC AUC for {c}: {auc:.4f}")

    # Plot ROC AUC per class
    import matplotlib.pyplot as plt
    class_labels = list(roc_aucs.keys())
    auc_values = [roc_aucs[c] for c in class_labels]
    plt.figure(figsize=(6,4))
    bars = plt.bar(class_labels, auc_values, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC per Class')
    for bar, auc in zip(bars, auc_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{auc:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('roc_auc_per_class.png')
    plt.close()

if __name__ == '__main__':
    main()