#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torchvision import transforms
from ijepa.src.models.vision_transformer import vit_huge
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from PIL import Image

# --- LoRA config (manual, not peft) ---
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# --- Load and filter data ---
bm = RetinaScanLoader().get_data()
df = bm.df.copy()
if 'patient_id' in df.columns:
    df = df.rename(columns={'patient_id': 'RegistrationCode'})
if 'date' not in df.columns:
    print("Warning: 'date' column not found in df!")
print('Original shape:', df.shape)

# Drop columns with >70% NaNs and then all remaining NaNs
nan_frac = df.isna().mean()
cols_to_keep = nan_frac[nan_frac <= 0.7].index.tolist()
df = df[cols_to_keep]
print('Columns kept:', cols_to_keep)
n_before = df.shape[0]
df = df.dropna()
print(f'Dropped {n_before - df.shape[0]} rows with NaNs. New shape: {df.shape}')

# Identify automorph feature columns
feature_cols = [c for c in df.columns if c.startswith('automorph_')]
print('Feature columns:', feature_cols)

# Merge with manifest to get image paths
BASE_DIR = os.path.expanduser("~/PycharmProjects/MSc_Thesis/JEPA")
MANIFEST_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
man_df = pd.read_csv(MANIFEST_CSV)

# Standardize and merge on available keys
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
if 'date' in man_df.columns:
    man_df['date'] = pd.to_datetime(man_df['date']).dt.strftime('%Y-%m-%d')
# Determine merge keys: always RegistrationCode, plus date if present in both
merge_keys = ['RegistrationCode']
if 'date' in df.columns and 'date' in man_df.columns:
    merge_keys.append('date')

merged = pd.merge(df, man_df, on=merge_keys, how='inner')
print('Merged shape:', merged.shape)

# Prepare features and targets (regress all automorph features)
X = merged[feature_cols].values.astype(np.float32)
y = X.copy()
paths = merged[['od_path', 'os_path']].values

# Split train/val
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X, y, paths, test_size=0.2, random_state=42
)
print('Train shape:', X_train.shape, 'Val shape:', X_val.shape)

# --- Image transforms ---
IMG_SIZE = (336, 336)
# Normalize each 3-channel eye separately in transform, then concatenate
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class FeatureImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, paths, transform):
        self.X = X
        self.y = y
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        od_path, os_path = self.paths[idx]
        od_img = Image.open(od_path).convert('RGB')
        os_img = Image.open(os_path).convert('RGB')
        od_img = self.transform(od_img)
        os_img = self.transform(os_img)
        # concatenate two eyes into 6-channel tensor
        img = torch.cat([od_img, os_img], dim=0)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, target

# DataLoaders
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = FeatureImageDataset(X_train, y_train, paths_train, transform)
val_dataset   = FeatureImageDataset(X_val,   y_val,   paths_val,   transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# --- Model: ViT backbone + regression head ---
VIT_PATCH_SIZE = 14
VIT_EMBED_DIM = 1280
PRETRAINED_CKPT = os.path.join(BASE_DIR, "pretrained_IN/IN22K-vit.h.14-900e.pth.tar")

backbone = vit_huge(
    patch_size=VIT_PATCH_SIZE,
    img_size=IMG_SIZE,
    in_chans=6,
    use_lora=USE_LORA,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)
# Load pretrained weights (if available)
if os.path.isfile(PRETRAINED_CKPT):
    state_dict = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
    # adapt first conv to 6 channels
    if hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'proj'):
        w = state_dict.get('patch_embed.proj.weight', None)
        if w is not None and w.shape[1] == 3 and backbone.patch_embed.proj.weight.shape[1] == 6:
            w6 = w.repeat(1, 2, 1, 1)[:, :6]
            state_dict['patch_embed.proj.weight'] = w6
    try:
        backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {PRETRAINED_CKPT}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("No pretrained checkpoint found or specified.")

class RetinaFeatureRegressor(nn.Module):
    def __init__(self, backbone, out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(VIT_EMBED_DIM, out_dim)
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        cls_token = feats[:, 0]
        return self.head(cls_token)

# Instantiate model
model = RetinaFeatureRegressor(backbone, out_dim=len(feature_cols)).to(DEVICE)

# --- Freeze all but LoRA adapters and head for LoRA-only tuning ---
for name, param in model.named_parameters():
    if 'lora_' in name or name.startswith('head.'):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Optimizer and loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# --- Training loop ---
EPOCHS = 10
for epoch in range(1, EPOCHS+1):
    model.train()
    train_losses = []
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f'Epoch {epoch} Train Loss: {np.mean(train_losses):.4f}')

    model.eval()
    val_losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, targets)
            val_losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    print(f'Epoch {epoch} Val Loss: {np.mean(val_losses):.4f}')
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    r2s = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])]
    print('R2 per feature:', dict(zip(feature_cols, np.round(r2s, 3))))

print('Fine-tuning complete.')
