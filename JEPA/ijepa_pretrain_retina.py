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
import matplotlib.pyplot as plt
import time
import inspect

# Import IJepa and optional loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from ijepa.src.masks.multiblock import MaskCollator
from ijepa.src.masks.utils import apply_masks

# ---------- Configuration & Reproducibility ----------
SEED = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load existing config
with open('config.json', 'r') as f:
    CONFIG = json.load(f)
CONFIG['img_size'] = tuple(CONFIG['img_size'])

CONFIG['epochs'] = 100
CONFIG['use_lora'] = False  # Ensure LoRA is not used during pretraining
CONFIG['lr'] = 1e-4  # Lower learning rate for stability
print("Batch size:", CONFIG['batch_size'])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

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
    sqs = torch.zeros(3)
    cnt = 0
    for batch in tqdm(loader, desc="Compute stats"):
        b, c, h, w = batch.shape
        batch = batch.view(b, c, -1)
        sums += batch.mean(dim=2).sum(dim=0)
        sqs += (batch ** 2).mean(dim=2).sum(dim=0)
        cnt += b
    mean = (sums / cnt).tolist()
    std = ((sqs / cnt - (sums / cnt) ** 2).sqrt()).tolist()
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
        num_patches=(config['img_size'][0] // config['patch_size']) ** 2,
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
                    or k in ('cls_token', 'pos_embed')}
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
    class RetinaImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

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

    ds = lambda p: RetinaImageDataset(p, transform)
    mask_coll = MaskCollator(
        input_size=config['img_size'],
        patch_size=config['patch_size'],
        enc_mask_scale=config['mask_scales']['enc'],
        pred_mask_scale=config['mask_scales']['pred'],
        aspect_ratio=(0.2, 5.0),
        nenc=config['nenc'],
        npred=config['npred'],
        min_keep=2,
        allow_overlap=False
    )
    train_ds, val_ds = ds(paths['train']), ds(paths['val'])
    return (
        DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                   num_workers=config['num_workers'], pin_memory=True, collate_fn=mask_coll),
        DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                   num_workers=config['num_workers'], pin_memory=True, collate_fn=mask_coll)
    )


# ---------- Checkpoint Utilities ----------
CHECKPOINT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_pretrain.pth'
MAX_TRAIN_SECONDS = 11.5 * 3600  # 11.5 hours in seconds


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


def save_loss_plot(train_losses, val_losses, epochs, filename, config, stage):
    min_len = min(len(epochs), len(train_losses), len(val_losses))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_losses = val_losses[:min_len]
    plt.figure()
    plt.plot(epochs, train_losses, label=f'{stage.capitalize()} Train MSE')
    plt.plot(epochs, val_losses, label=f'{stage.capitalize()} Val MSE')
    plt.xlabel('Epoch');
    plt.ylabel('MSE Loss')
    plt.title(f'{stage.capitalize()} Loss (Image size: {config["img_size"]})')
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_checkpoint(state, train_losses, val_losses, epoch_list, config, stage, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")
    save_loss_plot(train_losses, val_losses, epoch_list, 'checkpoint_pretrain_loss.png', config, stage)
    logger.info("Checkpoint loss plot saved to checkpoint_pretrain_loss.png")


def load_checkpoint(filename=CHECKPOINT_PATH):
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from {filename}")
        return torch.load(filename, map_location=DEVICE)
    return None


def run_masked_pretrain(enc, pred, loader_train, loader_val, config, start_epoch=1, elapsed_time=0):
    # Debug: print which parameters do not require grad
    for name, param in enc.named_parameters():
        if not param.requires_grad:
            print(f"Encoder param {name} does not require grad")
    for name, param in pred.named_parameters():
        if not param.requires_grad:
            print(f"Predictor param {name} does not require grad")
    optimizer = AdamW(
        [{'params': [p for n, p in list(enc.named_parameters()) + list(pred.named_parameters()) if p.requires_grad],
          'weight_decay': config['weight_decay']}],
        lr=config['lr']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = get_grad_scaler()
    best_val = float('inf')
    masked_train_losses, masked_val_losses = [], []
    start_time = time.time()
    accum_steps = 2
    for epoch in range(start_epoch, config['epochs'] + 1):
        enc.train();
        pred.train()
        train_losses = []
        optimizer.zero_grad()
        for i, (imgs, m_enc, m_pred) in enumerate(tqdm(loader_train, desc=f"Pretrain E{epoch}")):
            imgs = imgs.to(DEVICE)
            m_enc = [m.to(DEVICE) for m in m_enc]
            m_pred = [m.to(DEVICE) for m in m_pred]
            if torch.isnan(imgs).any():
                print(f"NaN detected in input images at batch {i}!")
            with torch.amp.autocast('cuda'):
                z = enc(imgs, m_enc)
                out = pred(z, m_enc, m_pred)
                if torch.isnan(out).any():
                    print(f"NaN detected in model output at batch {i}!")
                with torch.no_grad():
                    h = F.layer_norm(enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                loss = F.mse_loss(out, t) / accum_steps
                if torch.isnan(loss):
                    print(f"NaN detected in loss at batch {i}!")
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
                epoch_list = list(range(start_epoch, epoch + 1))
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
        enc.eval();
        pred.eval()
        with torch.no_grad():
            for i, (imgs, m_enc, m_pred) in enumerate(loader_val):
                imgs = imgs.to(DEVICE)
                m_enc = [m.to(DEVICE) for m in m_enc]
                m_pred = [m.to(DEVICE) for m in m_pred]
                if torch.isnan(imgs).any():
                    print(f"NaN detected in validation input images at batch {i}!")
                with torch.amp.autocast('cuda'):
                    z = enc(imgs, m_enc)
                    out = pred(z, m_enc, m_pred)
                    if torch.isnan(out).any():
                        print(f"NaN detected in validation model output at batch {i}!")
                    h = F.layer_norm(enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                    l = F.mse_loss(out, t)
                    if torch.isnan(l):
                        print(f"NaN detected in validation loss at batch {i}!")
                val_losses.append(l.item())
        mt = np.mean(train_losses);
        mv = np.mean(val_losses)
        masked_train_losses.append(mt)
        masked_val_losses.append(mv)
        logger.info(f"Pretrain E{epoch} Train MSE={mt:.4f} Val MSE={mv:.4f}")
        scheduler.step()
        if mv < best_val:
            best_val = mv
    epochs = list(range(start_epoch, config['epochs'] + 1))
    save_loss_plot(masked_train_losses, masked_val_losses, epochs, 'masked_pretrain_loss.png', config,
                   'masked pretrain')
    logger.info("Saved masked pretrain loss curve to masked_pretrain_loss.png")
    logger.info("Masked pretraining complete.")


def main():
    man_df = pd.read_csv(CONFIG['manifest_csv'])
    paths_all = man_df[['od_path', 'os_path']].values
    stats_file = 'image_stats_pretrain.json'
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

    checkpoint = load_checkpoint()
    # If no checkpoint, check for ImageNet pretrain
    if checkpoint is None:
        imagenet_ckpt = os.path.expanduser(
            '~/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar')
        if os.path.isfile(imagenet_ckpt):
            CONFIG['pretrained_ckpt'] = imagenet_ckpt
            logger.info(f"No training checkpoint found, using ImageNet pretrain: {imagenet_ckpt}")
        else:
            CONFIG['pretrained_ckpt'] = None
            logger.info("No checkpoint found, starting from scratch.")
        stage = None
        epoch = 1
        elapsed_time = 0
    else:
        stage = checkpoint.get('stage', None)
        epoch = checkpoint.get('epoch', 1)
        elapsed_time = 0  # Reset elapsed time on resume
        logger.info(f"Resuming from checkpoint at epoch {epoch}, elapsed_time reset to 0, stage {stage}")

    # Always run masked pretraining in this script
    if stage is None or stage == 'pretrain':
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
            elapsed_time = 0

        run_masked_pretrain(enc, pred, loader_train, loader_val, CONFIG, start_epoch=epoch, elapsed_time=elapsed_time)


if __name__ == '__main__':
    main()