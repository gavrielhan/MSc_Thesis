#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
import random
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from ijepa.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from ijepa.src.masks.multiblock import MaskCollator
from ijepa.src.masks.utils import apply_masks
import matplotlib.pyplot as plt


# ---------- Configuration & Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONFIG = {
    'seed': SEED,
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.2,
    'img_size': (336, 336),
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'predictor_embed_dim': 384,
    'pred_depth': 6,
    'batch_size': 4,
    'num_workers': 2,
    'lr': 3e-4,
    'weight_decay': 1e-2,
    'epochs': 12,
    'mask_scales': {'enc': (0.2, 0.6), 'pred': (0.2, 0.6)},
    'nenc': 2,
    'npred': 2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Save config for reproducibility
with open('config.json', 'w') as f:
    json.dump(CONFIG, f, indent=4)
logger.info("Saved training config to config.json")

# ---------- Data Loading & Splitting ----------
MANIFEST_CSV = os.path.expanduser(
    "~/PycharmProjects/MSc_Thesis/JEPA/retina_prevalent_future_diagnosis.csv"
)
man_df = pd.read_csv(MANIFEST_CSV)

# Patient-level split (if patient_id exists)
if 'patient_id' in man_df.columns:
    patients = man_df['patient_id'].unique()
    train_pats, val_pats = train_test_split(
        patients, test_size=0.2, random_state=SEED
    )
    train_df = man_df[man_df['patient_id'].isin(train_pats)].reset_index(drop=True)
    val_df   = man_df[man_df['patient_id'].isin(val_pats)].reset_index(drop=True)
else:
    train_df, val_df = train_test_split(
        man_df, test_size=0.2, random_state=SEED
    )

train_paths = train_df[['od_path','os_path']].values
val_paths   = val_df[['od_path','os_path']].values

# ---------- Compute Dataset Mean & Std ----------
class SimpleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        od, os_ = self.paths[idx]
        try:
            img = Image.open(od).convert('RGB')
        except:
            img = Image.new('RGB', CONFIG['img_size'], 'black')
        img = transforms.Resize(CONFIG['img_size'])(img)
        return transforms.ToTensor()(img)

stat_loader = DataLoader(
    SimpleDataset(train_paths),
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

means = torch.zeros(3)
sqs   = torch.zeros(3)
cnt   = 0
for batch in tqdm(stat_loader, desc="Computing dataset stats"):
    b, c, h, w = batch.shape
    batch = batch.view(b, c, -1)
    means += batch.mean(dim=2).sum(dim=0)
    sqs   += (batch**2).mean(dim=2).sum(dim=0)
    cnt   += b

means /= cnt
sqs   /= cnt
stds   = (sqs - means**2).sqrt()
mean   = means.tolist()
std    = stds.tolist()
logger.info(f"Dataset mean: {mean}, std: {std}")

# ---------- Transforms & Dataset ----------
transform = transforms.Compose([
    transforms.Resize(CONFIG['img_size']),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class RetinaImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        od_path, os_path = self.paths[idx]
        try:
            od_img = Image.open(od_path).convert('RGB')
        except:
            logger.warning(f"Could not load {od_path}, using black image")
            od_img = Image.new('RGB', CONFIG['img_size'], 'black')
        try:
            os_img = Image.open(os_path).convert('RGB')
        except:
            logger.warning(f"Could not load {os_path}, using black image")
            os_img = Image.new('RGB', CONFIG['img_size'], 'black')
        od_t = self.transform(od_img)
        os_t = self.transform(os_img)
        return torch.cat([od_t, os_t], dim=0)

mask_collator = MaskCollator(
    input_size=CONFIG['img_size'],
    patch_size=CONFIG['patch_size'],
    enc_mask_scale=CONFIG['mask_scales']['enc'],
    pred_mask_scale=CONFIG['mask_scales']['pred'],
    aspect_ratio=(0.2, 5.0),
    nenc=CONFIG['nenc'],
    npred=CONFIG['npred'],
    min_keep=2,
    allow_overlap=False
)

train_loader = DataLoader(
    RetinaImageDataset(train_paths, transform),
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    collate_fn=mask_collator
)
val_loader = DataLoader(
    RetinaImageDataset(val_paths, transform),
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    collate_fn=mask_collator
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Model & LoRA Setup ----------
encoder = VisionTransformer(
    img_size=CONFIG['img_size'],
    patch_size=CONFIG['patch_size'],
    in_chans=6,
    embed_dim=CONFIG['embed_dim'],
    depth=CONFIG['depth'],
    num_heads=CONFIG['num_heads'],
    use_lora=CONFIG['use_lora'],
    lora_r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout']
).to(DEVICE)

predictor = VisionTransformerPredictor(
    num_patches=(CONFIG['img_size'][0] // CONFIG['patch_size'])**2,
    embed_dim=CONFIG['embed_dim'],
    predictor_embed_dim=CONFIG['predictor_embed_dim'],
    depth=CONFIG['pred_depth'],
    num_heads=CONFIG['num_heads'],
    use_lora=CONFIG['use_lora'],
    lora_r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout']
).to(DEVICE)

# Load and filter pretrained weights
PRETRAINED_CKPT = os.path.expanduser(
    "~/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar"
)
if os.path.isfile(PRETRAINED_CKPT):
    ckpt = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
    filtered = {}
    for k, v in ckpt.items():
        if k.startswith('patch_embed.') or k.startswith('blocks.') or k in ('cls_token','pos_embed'):
            filtered[k] = v
    # Adapt first conv from 3?6 channels
    w = filtered.get('patch_embed.proj.weight', None)
    if w is not None and w.shape[1] == 3 and encoder.patch_embed.proj.weight.shape[1] == 6:
        filtered['patch_embed.proj.weight'] = w.repeat(1, 2, 1, 1)[:, :6]
    encoder.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded pretrained encoder weights from {PRETRAINED_CKPT}")
else:
    logger.info("No pretrained checkpoint found; training from scratch.")

# Freeze all except LoRA & predictor head
for p in encoder.parameters():
    p.requires_grad = False
for name, p in encoder.named_parameters():
    if 'lora_' in name:
        p.requires_grad = True

for p in predictor.parameters():
    p.requires_grad = False
for name, p in predictor.named_parameters():
    if 'lora_' in name or name.startswith('head.'):
        p.requires_grad = True

# Optimizer with weight-decay parameter grouping
decay, no_decay = [], []
for name, p in list(encoder.named_parameters()) + list(predictor.named_parameters()):
    if not p.requires_grad:
        continue
    if any(nd in name for nd in ('bias','norm','pos_embed','cls_token')):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = AdamW(
    [
        {'params': decay,    'weight_decay': CONFIG['weight_decay']},
        {'params': no_decay, 'weight_decay':   0.}
    ],
    lr=CONFIG['lr']
)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
scaler = torch.cuda.amp.GradScaler()

# ---------- Training & Validation Loops ----------
best_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, CONFIG['epochs'] + 1):
    encoder.train(); predictor.train()
    epoch_train, epoch_cos = [], []

    for imgs, masks_enc, masks_pred in tqdm(
        train_loader, desc=f"Epoch {epoch} [Train]"
    ):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks_enc = [m.to(DEVICE, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(DEVICE, non_blocking=True) for m in masks_pred]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            z     = encoder(imgs, masks_enc)
            preds = predictor(z, masks_enc, masks_pred)
            with torch.no_grad():
                h = encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h_masked = apply_masks(h, masks_pred).repeat(len(masks_enc), 1, 1)
            loss = F.mse_loss(preds, h_masked)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_train.append(loss.item())
        cos = nn.CosineSimilarity(dim=-1)(
            preds.flatten(1), h_masked.flatten(1)
        ).mean().item()
        epoch_cos.append(cos)

    mean_train = np.mean(epoch_train)
    mean_cos   = np.mean(epoch_cos)
    train_losses.append(mean_train)
    logger.info(f"Epoch {epoch} Train ? MSE: {mean_train:.4f}, Cosine: {mean_cos:.4f}")

    # Validation
    encoder.eval(); predictor.eval()
    epoch_val = []
    with torch.no_grad():
        for imgs, masks_enc, masks_pred in tqdm(
            val_loader, desc=f"Epoch {epoch} [Val]"
        ):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks_enc = [m.to(DEVICE, non_blocking=True) for m in masks_enc]
            masks_pred= [m.to(DEVICE, non_blocking=True) for m in masks_pred]

            with torch.cuda.amp.autocast():
                z     = encoder(imgs, masks_enc)
                preds = predictor(z, masks_enc, masks_pred)
                h = encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h_masked = apply_masks(h, masks_pred).repeat(len(masks_enc), 1, 1)
                loss = F.mse_loss(preds, h_masked)

            epoch_val.append(loss.item())

    mean_val = np.mean(epoch_val)
    val_losses.append(mean_val)
    logger.info(f"Epoch {epoch} Val   ? MSE: {mean_val:.4f}")

    scheduler.step()

    # Save best checkpoint
    if mean_val < best_val_loss:
        best_val_loss = mean_val
        torch.save({
            'epoch': epoch,
            'encoder_state':   encoder.state_dict(),
            'predictor_state': predictor.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'mean': mean,
            'std': std
        }, 'best_checkpoint.pth')
        logger.info(f"Saved new best checkpoint (epoch {epoch}).")

# ---------- Plot & Save LoRA Matrices ----------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, CONFIG['epochs']+1), train_losses, label='Train MSE')
plt.plot(range(1, CONFIG['epochs']+1), val_losses,   label='Val MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('I-JEPA LoRA Fine-Tuning Loss')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.close()
logger.info("Saved loss curve to loss_curve.png")

# Extract & save only LoRA matrices
lora_state = {}
for name, param in list(encoder.named_parameters()) + list(predictor.named_parameters()):
    if 'lora_A' in name or 'lora_B' in name:
        lora_state[name] = param.detach().cpu()
torch.save(lora_state, 'lora_matrices_ijepa.pth')
logger.info("Saved LoRA matrices to lora_matrices_ijepa.pth")

logger.info("Fine-tuning complete.")