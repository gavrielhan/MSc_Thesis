#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
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
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import inspect

# Import IJepa and optional loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from ijepa.src.masks.multiblock import MaskCollator
from ijepa.src.masks.utils import apply_masks
try:
    from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
except ImportError:
    RetinaScanLoader = None

# ---------- Flags to choose pipeline stages ----------
RUN_MASKED_PRETRAIN = False
RUN_SUPERVISED_FINETUNE = True

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
    'img_size': (616, 616),
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'predictor_embed_dim': 384,
    'pred_depth': 6,
    'batch_size': 2,
    'num_workers': 2,
    'lr': 3e-4,
    'weight_decay': 1e-2,
    'epochs': 20,
    'mask_scales': {'enc': (0.2, 0.6), 'pred': (0.2, 0.6)},
    'nenc': 2,
    'npred': 2,
    'manifest_csv': os.path.expanduser(
        "~/PycharmProjects/MSc_Thesis/JEPA/retina_prevalent_future_diagnosis.csv"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Save config
with open('config.json', 'w') as f:
    json.dump(CONFIG, f, indent=4)
logger.info("Configuration saved to config.json")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Helpers & Builders ----------
def compute_image_stats(paths, config):
    class SimpleDS(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            od, os_ = self.paths[idx]
            img = Image.open(od).convert('RGB')
            img = transforms.Resize(config['img_size'])(img)
            return transforms.ToTensor()(img)

    loader = DataLoader(
        SimpleDS(paths),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    sums = torch.zeros(3)
    sqs  = torch.zeros(3)
    cnt  = 0
    for batch in tqdm(loader, desc="Compute stats"):
        b, c, h, w = batch.shape
        batch = batch.view(b, c, -1)
        sums += batch.mean(dim=2).sum(dim=0)
        sqs  += (batch**2).mean(dim=2).sum(dim=0)
        cnt  += b
    mean = (sums / cnt).tolist()
    std  = ((sqs / cnt - (sums / cnt)**2).sqrt()).tolist()
    return mean, std


def build_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_encoder_and_predictor(config):
    enc = VisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_chans=6,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    pred = VisionTransformerPredictor(
        num_patches=(config['img_size'][0] // config['patch_size'])**2,
        embed_dim=config['embed_dim'],
        predictor_embed_dim=config['predictor_embed_dim'],
        depth=config['pred_depth'],
        num_heads=config['num_heads'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    return enc.to(DEVICE), pred.to(DEVICE)


def load_pretrained_encoder(enc, config):
    ckpt = config.get('pretrained_ckpt', None)
    if ckpt and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=DEVICE)
        filtered = {k: v for k, v in state.items()
                    if k.startswith('patch_embed.') or k.startswith('blocks.')
                    or k in ('cls_token','pos_embed')}
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 3 and enc.patch_embed.proj.weight.shape[1] == 6:
            filtered['patch_embed.proj.weight'] = w.repeat(1, 2, 1, 1)[:, :6]
        enc.load_state_dict(filtered, strict=False)
        logger.info("Pretrained weights loaded into encoder")
    else:
        logger.info("No pretrained checkpoint found; skipping load")


def freeze_lora_params(model):
    for p in model.parameters(): p.requires_grad = False
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True


def build_masked_dataloaders(paths, transform, config):
    ds = lambda p: RetinaImageDataset(p, transform)
    mask_coll = MaskCollator(
        input_size=config['img_size'],
        patch_size=config['patch_size'],
        enc_mask_scale=config['mask_scales']['enc'],
        pred_mask_scale=config['mask_scales']['pred'],
        aspect_ratio=(0.2,5.0),
        nenc=config['nenc'],
        npred=config['npred'],
        min_keep=2,
        allow_overlap=False
    )
    train_ds, val_ds = ds(paths['train']), ds(paths['val'])
    return (
        DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                   num_workers=config['num_workers'], pin_memory=True, collate_fn=mask_coll),
        DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False,
                   num_workers=config['num_workers'], pin_memory=True, collate_fn=mask_coll)
    )


class RetinaImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        od_path, os_path = self.paths[idx]
        try:
            od = Image.open(od_path).convert('RGB')
        except:
            od = Image.new('RGB', CONFIG['img_size'], 'black')
        try:
            os_ = Image.open(os_path).convert('RGB')
        except:
            os_ = Image.new('RGB', CONFIG['img_size'], 'black')
        od_t = self.transform(od)
        os_t = self.transform(os_)
        return torch.cat([od_t, os_t], dim=0)


class RegressionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        if x.ndim == 3:
            x = x[:, 0]
        return self.head(x)


def build_supervised_loaders(config, transform):
    assert RetinaScanLoader is not None, "RetinaScanLoader not available"
    df_feat = RetinaScanLoader().get_data().df.reset_index(drop=False)
    nan_frac = df_feat.isna().mean()
    cols_keep = nan_frac[nan_frac <= 0.7].index.tolist()
    df_feat = df_feat[cols_keep].dropna().reset_index(drop=True)
    man_df = pd.read_csv(config['manifest_csv'])
    if 'date' in df_feat.columns:
        df_feat['date'] = pd.to_datetime(df_feat['date']).dt.strftime('%Y-%m-%d')
    if 'date' in man_df.columns:
        man_df['date'] = pd.to_datetime(man_df['date']).dt.strftime('%Y-%m-%d')
    merge_keys = ['RegistrationCode']
    if 'date' in df_feat.columns and 'date' in man_df.columns:
        merge_keys.append('date')
    merged = pd.merge(df_feat, man_df, on=merge_keys, how='inner')
    feature_cols = [c for c in df_feat.columns if c.startswith('automorph_')]
    labeled = merged.dropna(subset=['od_path','os_path'] + feature_cols).reset_index(drop=True)
    paths = labeled[['od_path','os_path']].values
    feats = labeled[feature_cols].values.astype(np.float32)
    idx = np.arange(len(paths))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=config['seed'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=config['seed'])
    paths_train, paths_val, paths_test = paths[train_idx], paths[val_idx], paths[test_idx]
    feats_train, feats_val, feats_test = feats[train_idx], feats[val_idx], feats[test_idx]
    # Reduce training samples by 40% (keep 60%)
    if len(paths_train) > 0:
        print(len(paths_train))
        rng = np.random.default_rng(config['seed'])
        n_sub = int(len(paths_train) * 0.3)
        sub_idx = rng.choice(len(paths_train), n_sub, replace=False)
        paths_train = paths_train[sub_idx]
        feats_train = feats_train[sub_idx]
    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_val   = scaler.transform(feats_val)
    feats_test  = scaler.transform(feats_test)
    class SupDS(Dataset):
        def __init__(self, paths, feats):
            self.paths = paths
            self.feats = feats
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            od, os_ = self.paths[idx]
            od = os.path.expanduser(od)
            os_img_path = os.path.expanduser(os_)
            try:
                img_od = Image.open(od).convert('RGB')
            except:
                img_od = Image.new('RGB', CONFIG['img_size'], 'black')
            try:
                img_os = Image.open(os_img_path).convert('RGB')
            except:
                img_os = Image.new('RGB', CONFIG['img_size'], 'black')
            t_od = transform(img_od)
            t_os = transform(img_os)
            img = torch.cat([t_od, t_os], dim=0)
            target = torch.tensor(self.feats[idx], dtype=torch.float32)
            return img, target
    sup_train = DataLoader(SupDS(paths_train, feats_train), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    sup_val   = DataLoader(SupDS(paths_val,   feats_val),   batch_size=config['batch_size'], shuffle=False,num_workers=config['num_workers'], pin_memory=True)
    sup_test  = DataLoader(SupDS(paths_test,  feats_test),  batch_size=config['batch_size'], shuffle=False,num_workers=config['num_workers'], pin_memory=True)
    sup_loaders = {'train': sup_train, 'val': sup_val, 'test': sup_test}
    n_features = len(feature_cols)
    return sup_loaders, n_features


# ---------- Checkpoint Utilities ----------
CHECKPOINT_PATH = 'checkpoint.pth'
MAX_TRAIN_SECONDS = 11.5 * 3600  # 11.5 hours in seconds

# Robust GradScaler instantiation for PyTorch 1.x and 2.x
def get_grad_scaler():
    amp_grad_scaler = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if amp_grad_scaler is not None:
        sig = inspect.signature(amp_grad_scaler)
        if 'device_type' in sig.parameters:
            return amp_grad_scaler(device_type='cuda')
        else:
            return amp_grad_scaler()
    else:
        return torch.cuda.amp.GradScaler()

# Helper to plot and save loss curves with image size in the title
def save_loss_plot(train_losses, val_losses, epochs, filename, config, stage):
    plt.figure()
    plt.plot(epochs, train_losses, label=f'{stage.capitalize()} Train MSE')
    plt.plot(epochs, val_losses, label=f'{stage.capitalize()} Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{stage.capitalize()} Loss (Image size: {config["img_size"]})')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_checkpoint(state, train_losses, val_losses, epoch_list, config, stage, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")
    # Save loss plot at checkpoint
    save_loss_plot(train_losses, val_losses, epoch_list, 'checkpoint_loss.png', config, stage)
    logger.info("Checkpoint loss plot saved to checkpoint_loss.png")


def load_checkpoint(filename=CHECKPOINT_PATH):
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from {filename}")
        return torch.load(filename, map_location=DEVICE)
    return None


def run_masked_pretrain(enc, pred, loader_train, loader_val, config, start_epoch=1, elapsed_time=0):
    freeze_lora_params(enc)
    freeze_lora_params(pred)
    optimizer = AdamW(
        [{'params': [p for n,p in list(enc.named_parameters())+list(pred.named_parameters()) if p.requires_grad],
          'weight_decay': config['weight_decay']}],
        lr=config['lr']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = get_grad_scaler()
    best_val = float('inf')
    masked_train_losses, masked_val_losses = [], []
    start_time = time.time()
    accum_steps = 2
    for epoch in range(start_epoch, config['epochs']+1):
        enc.train(); pred.train()
        train_losses = []
        optimizer.zero_grad()
        for i, (imgs, m_enc, m_pred) in enumerate(tqdm(loader_train, desc=f"Pretrain E{epoch}")):
            imgs = imgs.to(DEVICE)
            m_enc = [m.to(DEVICE) for m in m_enc]
            m_pred= [m.to(DEVICE) for m in m_pred]
            with torch.amp.autocast(device_type='cuda'):
                z    = enc(imgs, m_enc)
                out  = pred(z, m_enc, m_pred)
                with torch.no_grad():
                    h = F.layer_norm(enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc),1,1)
                loss = F.mse_loss(out, t) / accum_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * accum_steps)  # store unscaled loss
            # Gradient accumulation
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # Check for time limit
            total_elapsed = elapsed_time + (time.time() - start_time)
            if total_elapsed >= MAX_TRAIN_SECONDS:
                epoch_list = list(range(start_epoch, epoch+1))
                save_checkpoint({
                    'enc': enc.state_dict(),
                    'pred': pred.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'elapsed_time': total_elapsed,
                    'stage': 'pretrain',
                },
                masked_train_losses + [np.mean(train_losses)],
                masked_val_losses,
                epoch_list,
                config,
                'pretrain')
                logger.info("Reached max training time. Exiting.")
                sys.exit(0)
        # Final step if leftover gradients
        if len(loader_train) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        val_losses = []
        enc.eval(); pred.eval()
        with torch.no_grad():
            for imgs, m_enc, m_pred in loader_val:
                imgs = imgs.to(DEVICE)
                m_enc = [m.to(DEVICE) for m in m_enc]
                m_pred= [m.to(DEVICE) for m in m_pred]
                with torch.amp.autocast(device_type='cuda'):
                    z   = enc(imgs, m_enc)
                    out = pred(z, m_enc, m_pred)
                    h   = F.layer_norm(enc(imgs), (z.size(-1),))
                    t   = apply_masks(h, m_pred).repeat(len(m_enc),1,1)
                    l   = F.mse_loss(out, t)
                val_losses.append(l.item())
        mt = np.mean(train_losses); mv = np.mean(val_losses)
        masked_train_losses.append(mt)
        masked_val_losses.append(mv)
        logger.info(f"Pretrain E{epoch} Train MSE={mt:.4f} Val MSE={mv:.4f}")
        scheduler.step()
        if mv < best_val:
            best_val = mv
            #torch.save(enc.state_dict(), 'best_enc.pth')
            #torch.save(pred.state_dict(), 'best_pred.pth')
    epochs = list(range(start_epoch, config['epochs']+1))
    save_loss_plot(masked_train_losses, masked_val_losses, epochs, 'masked_pretrain_loss.png', config, 'masked pretrain')
    logger.info("Saved masked pretrain loss curve to masked_pretrain_loss.png")
    logger.info("Masked pretraining complete.")


def run_supervised_finetune(enc, head, sup_loaders, config, n_features, start_epoch=1, elapsed_time=0):
    freeze_lora_params(enc)
    head.to(DEVICE)
    optimizer = AdamW([
        {'params': [p for n,p in enc.named_parameters() if p.requires_grad], 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.0}
    ], lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = get_grad_scaler()
    sup_train_losses, sup_val_losses = [], []
    start_time = time.time()
    accum_steps = 2
    for epoch in range(start_epoch, config['epochs']+1):
        enc.eval(); head.train()
        train_losses = []
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(tqdm(sup_loaders['train'], desc=f"Sup E{epoch} Train")):
            imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                f = enc(imgs); preds = head(f); loss = F.mse_loss(preds, targets) / accum_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * accum_steps)
            # Gradient accumulation
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # Check for time limit
            total_elapsed = elapsed_time + (time.time() - start_time)
            if total_elapsed >= MAX_TRAIN_SECONDS:
                epoch_list = list(range(start_epoch, epoch+1))
                save_checkpoint({
                    'enc': enc.state_dict(),
                    'head': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'elapsed_time': total_elapsed,
                    'stage': 'finetune',
                },
                sup_train_losses + [np.mean(train_losses)],
                sup_val_losses,
                epoch_list,
                config,
                'finetune')
                logger.info("Reached max training time. Exiting.")
                sys.exit(0)
        # Final step if leftover gradients
        if len(sup_loaders['train']) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # validation
        enc.eval(); head.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, targets in sup_loaders['val']:
                imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
                with torch.amp.autocast(device_type='cuda'):
                    f = enc(imgs); preds = head(f); loss = F.mse_loss(preds, targets)
                val_losses.append(loss.item())
        mt = np.mean(train_losses); mv = np.mean(val_losses)
        sup_train_losses.append(mt)
        sup_val_losses.append(mv)
        logger.info(f"Sup E{epoch} Train MSE={mt:.4f} Val MSE={mv:.4f}")
        scheduler.step()
    epochs = list(range(start_epoch, config['epochs']+1))
    save_loss_plot(sup_train_losses, sup_val_losses, epochs, 'supervised_finetune_loss.png', config, 'supervised finetune')
    logger.info("Saved supervised fine-tuning loss curve to supervised_finetune_loss.png")
    logger.info("Supervised fine-tuning complete.")


def main():
    # Load manifest and compute transforms (used by both stages)
    man_df = pd.read_csv(CONFIG['manifest_csv'])
    paths_all = man_df[['od_path','os_path']].values
    stats_file = 'image_stats.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        mean, std = stats['mean'], stats['std']
        logger.info(f"Loaded cached image stats from {stats_file}")
    else:
        mean, std = compute_image_stats(paths_all, CONFIG)
        with open(stats_file, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f)
        logger.info(f"Computed and saved image stats to {stats_file}")
    trsf = build_transforms(mean, std)

    # Check for checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        stage = checkpoint.get('stage', None)
        epoch = checkpoint.get('epoch', 1)
        elapsed_time = checkpoint.get('elapsed_time', 0)
        logger.info(f"Resuming from checkpoint at epoch {epoch}, elapsed_time {elapsed_time/3600:.2f} hrs, stage {stage}")
    else:
        stage = None
        epoch = 1
        elapsed_time = 0

    # Masked pretrain stage
    if RUN_MASKED_PRETRAIN and (stage is None or stage == 'pretrain'):
        tr_paths, val_paths = train_test_split(
            paths_all, test_size=0.2, random_state=CONFIG['seed']
        )
        paths = {'train': tr_paths, 'val': val_paths}
        loader_train, loader_val = build_masked_dataloaders(paths, trsf, CONFIG)
        enc, pred = build_encoder_and_predictor(CONFIG)
        load_pretrained_encoder(enc, CONFIG)
        if checkpoint is not None and stage == 'pretrain':
            enc.load_state_dict(checkpoint['enc'])
            pred.load_state_dict(checkpoint['pred'])
        run_masked_pretrain(enc, pred, loader_train, loader_val, CONFIG, start_epoch=epoch, elapsed_time=elapsed_time)

    # Supervised fine-tune stage
    if RUN_SUPERVISED_FINETUNE and (stage is None or stage == 'finetune'):
        try:
            enc
        except NameError:
            enc, _ = build_encoder_and_predictor(CONFIG)
            load_pretrained_encoder(enc, CONFIG)
        sup_loaders, n_features = build_supervised_loaders(CONFIG, trsf)
        head = RegressionHead(CONFIG['embed_dim'], n_features)
        if checkpoint is not None and stage == 'finetune':
            enc.load_state_dict(checkpoint['enc'])
            head.load_state_dict(checkpoint['head'])
        run_supervised_finetune(enc, head, sup_loaders, CONFIG, n_features, start_epoch=epoch, elapsed_time=elapsed_time)

if __name__ == '__main__':
    main()
