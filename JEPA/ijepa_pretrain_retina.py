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
import copy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Import IJepa and optional loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from ijepa.src.masks.multiblock import MaskCollator
from ijepa.src.masks.utils import apply_masks


def match_encoder_input_channels(encoder: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Adjust batch tensor channels to match encoder's expected in_chans.
    If encoder expects 6 and input is 3, duplicate (OD as OS). If encoder expects 3 and input is 6, average halves.
    """
    try:
        target_c = encoder.patch_embed.proj.weight.shape[1]
    except Exception:
        return imgs
    c = imgs.shape[1]
    if c == target_c:
        return imgs
    if target_c == 6 and c == 3:
        return torch.cat([imgs, imgs], dim=1)
    if target_c == 3 and c == 6:
        return 0.5 * (imgs[:, 0:3, ...] + imgs[:, 3:6, ...])
    # Fallbacks
    if c > target_c:
        return imgs[:, :target_c, ...]
    if c < target_c:
        reps = target_c // c
        return imgs.repeat(1, reps, 1, 1)[:, :target_c, ...]
    return imgs


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
CONFIG['img_size'] = (602, 602)

CONFIG['epochs'] = 100
CONFIG['num_workers'] = 0
CONFIG['batch_size'] = 1
CONFIG['use_lora'] = False  # Ensure LoRA is not used during pretraining
CONFIG['lr'] = 5.0e-5  # Lower learning rate for stability
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


def sanity_check_black_images(paths, sample_size=200, threshold=10.0):
    """Sample up to sample_size pairs and check mean brightness.
    threshold is in [0,255] scale; below means "very dark/black".
    """
    total_checked = 0
    black_count = 0
    examples = []
    for i, (od_path, os_path) in enumerate(paths[:sample_size]):
        try:
            od = Image.open(od_path).convert('RGB')
            os_ = Image.open(os_path).convert('RGB')
            od_mean = np.array(od).mean()
            os_mean = np.array(os_).mean()
            total_checked += 2
            if od_mean < threshold:
                black_count += 1
                if len(examples) < 3:
                    examples.append(od_path)
            if os_mean < threshold:
                black_count += 1
                if len(examples) < 3:
                    examples.append(os_path)
        except Exception:
            # Treat unreadable images as black
            black_count += 2
            total_checked += 2
    pct = (black_count / max(1, total_checked)) * 100.0
    logger.info(f"[SANITY] Checked {total_checked} eyes; black/dark: {black_count} ({pct:.1f}%).")
    if examples:
        logger.info(f"[SANITY] Example dark files: {examples}")
    if pct > 50.0:
        logger.warning(
            "[SANITY] More than 50% of sampled images are very dark; please verify dataset paths and preprocessing.")
    return pct


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
CHECKPOINT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_pretrain_newrun.pth'
MAX_TRAIN_SECONDS = 11.5 * 3600  # 11.5 hours in seconds
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

OUTPUT_ROOT = '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/outputs'
IMAGES_DIR = os.path.join(OUTPUT_ROOT, 'images')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def get_grad_scaler():
    amp_grad_scaler = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    # If bfloat16 is supported, we won't use GradScaler (not needed for bf16)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return None
    # Otherwise (fp32 or fp16), return a GradScaler if available
    if amp_grad_scaler is not None:
        sig = inspect.signature(amp_grad_scaler)
        if 'device_type' in sig.parameters:
            return amp_grad_scaler(device_type='cuda')
        else:
            return amp_grad_scaler()
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
    save_loss_plot(train_losses, val_losses, epoch_list, os.path.join(IMAGES_DIR, 'checkpoint_pretrain_loss.png'),
                   config, stage)
    logger.info("Checkpoint loss plot saved to checkpoint_pretrain_loss.png")


def load_checkpoint(filename=CHECKPOINT_PATH):
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from {filename}")
        return torch.load(filename, map_location=DEVICE)
    return None


def run_masked_pretrain(enc, pred, loader_train, loader_val, config, start_epoch=1, elapsed_time=0, demo_eval=None):
    import signal
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
    # Create EMA target encoder
    target_enc = copy.deepcopy(enc).to(DEVICE)
    for p in target_enc.parameters():
        p.requires_grad = False
    ema_momentum = 0.99
    # Periodic time-based checkpointing (every 2 hours)
    time_ckpt_interval = 2 * 3600
    last_time_ckpt = time.time()

    def write_time_checkpoint(epoch_completed):
        torch.save({
            'enc': enc.state_dict(),
            'pred': pred.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch_completed,
            'stage': 'pretrain',
        }, CHECKPOINT_PATH)
        logger.info(f"Checkpoint overwritten at {CHECKPOINT_PATH}")

    # Save on SIGTERM
    def _handle_sigterm(signum, frame):
        logger.info("SIGTERM received: saving last checkpoint and exiting")
        write_time_checkpoint(max(start_epoch, 1) - 1)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        pass
    best_val = float('inf')
    masked_train_losses, masked_val_losses = [], []
    start_time = time.time()
    accum_steps = 2
    # Unpack demo eval tuple
    eval_loader = eval_maps = metrics_path = None
    if isinstance(demo_eval, tuple) and len(demo_eval) == 3:
        eval_loader, eval_maps, metrics_path = demo_eval
    for epoch in range(start_epoch, config['epochs'] + 1):
        enc.train();
        pred.train()
        train_losses = []
        optimizer.zero_grad()
        for i, (imgs, m_enc, m_pred) in enumerate(tqdm(loader_train, desc=f"Pretrain E{epoch}")):
            imgs = imgs.to(DEVICE)
            m_enc = [m.to(DEVICE) for m in m_enc]
            m_pred = [m.to(DEVICE) for m in m_pred]
            # Print image stats for the first batch of each epoch
            if i == 0:
                print(
                    f"Batch {i} image stats: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, mean={imgs.mean().item():.4f}")
            # 4. Check for Infs in input images
            if torch.isinf(imgs).any():
                print(f"Inf detected in input images at batch {i}!")
            if torch.isnan(imgs).any():
                print(f"NaN detected in input images at batch {i}!")
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(torch.cuda.is_available() and torch.cuda.is_bf16_supported())):
                z = enc(imgs, m_enc)
                out = pred(z, m_enc, m_pred)
                if torch.isnan(out).any():
                    print(f"NaN detected in model output at batch {i}!")
                with torch.no_grad():
                    h = F.layer_norm(target_enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                loss = F.smooth_l1_loss(out, t) / accum_steps
                if torch.isnan(loss):
                    print(f"NaN detected in loss at batch {i}!")
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # 5. Check for NaNs in gradients
            for name, param in enc.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in encoder gradient: {name}")
            for name, param in pred.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in predictor gradient: {name}")
            train_losses.append(loss.item() * accum_steps)  # store unscaled loss
            # Gradient accumulation
            if (i + 1) % accum_steps == 0:
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(pred.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # EMA update for target encoder
                with torch.no_grad():
                    for p_online, p_target in zip(enc.parameters(), target_enc.parameters()):
                        p_target.data.mul_(ema_momentum).add_((1 - ema_momentum) * p_online.data)
            # Check for time limit
            total_elapsed = elapsed_time + (time.time() - start_time)
            if total_elapsed >= MAX_TRAIN_SECONDS:
                # Save only up to the last completed epoch
                epoch_list = list(range(start_epoch, epoch))  # up to epoch-1
                save_checkpoint({
                    'enc': enc.state_dict(),
                    'pred': pred.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': None if scaler is None else scaler.state_dict(),
                    'epoch': epoch - 1,  # last completed epoch
                    'elapsed_time': total_elapsed,
                    'stage': 'pretrain',
                },
                    masked_train_losses,  # up to last completed epoch
                    masked_val_losses,
                    epoch_list,
                    config,
                    'pretrain')
                # PCA plots AFTER training (timeout)
                try:
                    if eval_loader is not None and eval_maps is not None:
                        def _pca_after():
                            enc.eval()
                            feats = []
                            regs = []
                            with torch.no_grad():
                                for imgs, regcode in eval_loader:
                                    imgs = imgs.to(DEVICE)
                                    imgs = match_encoder_input_channels(enc, imgs)
                                    z = enc(imgs)
                                    if z.ndim == 3:
                                        z = z.mean(dim=1)
                                    feats.append(z.cpu().numpy())
                                    regs.extend(list(regcode))
                            if feats:
                                X = np.concatenate(feats, axis=0)
                                Xs = StandardScaler().fit_transform(X)
                                pca = PCA(n_components=2, random_state=42)
                                X2 = pca.fit_transform(Xs)
                                evr = pca.explained_variance_ratio_
                                xlab = f'PC1 ({evr[0] * 100:.1f}%)'
                                ylab = f'PC2 ({evr[1] * 100:.1f}%)'

                                # reuse plotting helper by re-importing create_pca_plots via closure not available here; inline minimal
                                def cont_plot(vals, title, fname):
                                    v = np.array(vals, dtype=float)
                                    m = ~np.isnan(v)
                                    if not m.any():
                                        return
                                    norm = mcolors.Normalize(vmin=np.nanpercentile(v[m], 5),
                                                             vmax=np.nanpercentile(v[m], 95))
                                    colors = cm.viridis(norm(v[m]))
                                    fig, ax = plt.subplots(figsize=(6, 5))
                                    ax.scatter(X2[m, 0], X2[m, 1], s=6, alpha=0.8, c=colors)
                                    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
                                    sm.set_array([])
                                    fig.colorbar(sm, ax=ax).set_label(title)
                                    ax.set_title(f'PCA (after) - {title}')
                                    ax.set_xlabel(xlab);
                                    ax.set_ylabel(ylab)
                                    fig.tight_layout()
                                    fig.savefig(os.path.join(IMAGES_DIR, fname))
                                    plt.close(fig)

                                # Sex
                                sex_vals = [eval_maps['sex_map'].get(rc, None) for rc in regs]
                                try:
                                    mask = np.array([v is not None for v in sex_vals])
                                    if mask.any():
                                        s = np.array([int(v) for v in np.array(sex_vals, dtype=object)[mask]])
                                        plt.figure(figsize=(6, 5))
                                        for label, color, name in [(0, 'tab:blue', 'Class 0'),
                                                                   (1, 'tab:orange', 'Class 1')]:
                                            sel_idx = np.where(mask)[0][s == label]
                                            plt.scatter(X2[sel_idx, 0], X2[sel_idx, 1], s=6, alpha=0.7, c=color,
                                                        label=name)
                                        plt.legend(title='Sex')
                                        plt.title(f'PCA (after) - Sex')
                                        plt.xlabel(xlab);
                                        plt.ylabel(ylab)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(IMAGES_DIR, 'pca_after_sex.png'))
                                        plt.close()
                                except Exception:
                                    pass
                                cont_plot([eval_maps['age_map'].get(rc, None) for rc in regs], 'Age',
                                          'pca_after_age.png')
                                cont_plot([eval_maps['bmi_map'].get(rc, None) for rc in regs], 'BMI',
                                          'pca_after_bmi.png')

                        _pca_after()
                except Exception as e:
                    logger.info(f"PCA (after-timeout) failed: {e}")
                logger.info("Reached max training time. Exiting.")
                sys.exit(0)
            # Periodic time-based checkpoint
            now = time.time()
            if now - last_time_ckpt >= time_ckpt_interval:
                write_time_checkpoint(epoch_completed=max(epoch - 1, start_epoch - 1))
                last_time_ckpt = now
        # Final step if leftover gradients
        if len(loader_train) % accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                for p_online, p_target in zip(enc.parameters(), target_enc.parameters()):
                    p_target.data.mul_(ema_momentum).add_((1 - ema_momentum) * p_online.data)
        val_losses = []
        enc.eval();
        pred.eval()
        with torch.no_grad():
            for i, (imgs, m_enc, m_pred) in enumerate(loader_val):
                imgs = imgs.to(DEVICE)
                m_enc = [m.to(DEVICE) for m in m_enc]
                m_pred = [m.to(DEVICE) for m in m_pred]
                # 4. Check for Infs in input images during validation
                if torch.isinf(imgs).any():
                    print(f"Inf detected in validation input images at batch {i}!")
                if torch.isnan(imgs).any():
                    print(f"NaN detected in validation input images at batch {i}!")
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                        enabled=(torch.cuda.is_available() and torch.cuda.is_bf16_supported())):
                    z = enc(imgs, m_enc)
                    out = pred(z, m_enc, m_pred)
                    if torch.isnan(out).any():
                        print(f"NaN detected in validation model output at batch {i}!")
                    h = F.layer_norm(target_enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                    l = F.smooth_l1_loss(out, t)
                    if torch.isnan(l):
                        print(f"NaN detected in validation loss at batch {i}!")
                val_losses.append(l.item())
        mt = np.mean(train_losses);
        mv = np.mean(val_losses)
        masked_train_losses.append(mt)
        masked_val_losses.append(mv)
        # Per-epoch demographic evaluation
        epoch_metrics = {
            'epoch': epoch,
            'train_mse': float(mt),
            'val_mse': float(mv),
        }
        if eval_loader is not None and eval_maps is not None:
            demo_res = None
            try:
                # Use online encoder features (no masks) for demographics
                def _extractor(x):
                    return enc(x)

                demo_res = evaluate_demographics(enc, eval_loader, eval_maps)
            except Exception as e:
                logger.info(f"Demographic eval failed: {e}")
            if demo_res:
                epoch_metrics.update(demo_res)
        # Append to metrics JSON
        if metrics_path:
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        hist = json.load(f)
                else:
                    hist = []
                hist.append(epoch_metrics)
                with open(metrics_path, 'w') as f:
                    json.dump(hist, f, indent=2)
                logger.info(f"Saved epoch metrics to {metrics_path}")
            except Exception as e:
                logger.info(f"Failed to write metrics JSON: {e}")
        # Compute and print relative MSE (MSE / variance of target)
        # Use a batch of targets from validation set for variance estimate
        try:
            all_targets = []
            for i, (imgs, m_enc, m_pred) in enumerate(loader_val):
                with torch.no_grad():
                    imgs = imgs.to(DEVICE)
                    m_enc = [m.to(DEVICE) for m in m_enc]
                    m_pred = [m.to(DEVICE) for m in m_pred]
                    h = F.layer_norm(target_enc(imgs), (enc(imgs).size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                    all_targets.append(t.cpu().numpy().ravel())  # flatten to 1D
                if i > 10:  # limit to 10 batches for speed
                    break
            all_targets = np.concatenate(all_targets, axis=0)  # now 1D
            target_var = np.var(all_targets)
            rel_train_mse = mt / target_var if target_var > 0 else float('nan')
            rel_val_mse = mv / target_var if target_var > 0 else float('nan')
            logger.info(f"Pretrain E{epoch} Relative Train MSE={rel_train_mse:.4f} Relative Val MSE={rel_val_mse:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute relative MSE: {e}")
        logger.info(f"Pretrain E{epoch} Train MSE={mt:.4f} Val MSE={mv:.4f}")
        scheduler.step()
        if mv < best_val:
            best_val = mv


def build_demo_maps():
    try:
        from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
    except Exception as e:
        logger.warning(f"BodyMeasuresLoader not available ({e}); demographic evaluation disabled")
        return None
    try:
        study_ids = [10, 1001, 1002]
        bm = BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first')
        df_meta = bm.df.join(bm.df_metadata)
        df_meta = df_meta.reset_index().rename(columns={'index': 'RegistrationCode'})
        df_meta['RegistrationCode'] = df_meta['RegistrationCode'].astype(str)
        sex_col = next((c for c in df_meta.columns if 'sex' in c.lower() or 'gender' in c.lower()), None)
        bmi_col = next((c for c in df_meta.columns if 'bmi' in c.lower()), None)
        if sex_col:
            df_meta = df_meta.dropna(subset=[sex_col])
            sex_map = (
                df_meta.set_index('RegistrationCode')[sex_col]
                .astype('category').cat.codes.to_dict()
            )
        else:
            sex_map = {}
        if bmi_col:
            bmi_series = df_meta.set_index('RegistrationCode')[bmi_col].astype(float)
            bmi_median = bmi_series.median(skipna=True)
            bmi_series = bmi_series.fillna(bmi_median)
            bmi_map = bmi_series.to_dict()
        else:
            bmi_map = {}
        age_col = next((c for c in df_meta.columns if c.lower() == 'age'), None)
        if age_col is None and 'yob' in df_meta.columns:
            current_year = time.localtime().tm_year
            age_series = (current_year - pd.to_numeric(df_meta['yob'], errors='coerce')).astype(float)
            df_meta = df_meta.assign(_age=age_series)
            age_map = df_meta.set_index('RegistrationCode')['_age'].dropna().to_dict()
        elif age_col is not None:
            age_series = df_meta.set_index('RegistrationCode')[age_col].astype(float)
            age_map = age_series.dropna().to_dict()
        else:
            age_map = {}
        sex_map = {str(k): v for k, v in sex_map.items()}
        bmi_map = {str(k): v for k, v in bmi_map.items()}
        age_map = {str(k): v for k, v in age_map.items()}
        return {'sex_map': sex_map, 'bmi_map': bmi_map, 'age_map': age_map}
    except Exception as e:
        logger.warning(f"Failed to build demographic maps: {e}; demographic evaluation disabled")
        return None


class EvalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        od_path = row['od_path']
        os_path = row.get('os_path', None)
        rc = str(row['RegistrationCode']) if 'RegistrationCode' in row else None
        try:
            od = Image.open(od_path).convert('RGB')
        except Exception:
            od = Image.new('RGB', CONFIG['img_size'], 'black')
        if os_path is not None:
            try:
                os_ = Image.open(os_path).convert('RGB')
            except Exception:
                os_ = Image.new('RGB', CONFIG['img_size'], 'black')
        else:
            os_ = od.copy()
        od_t = self.transform(od)
        os_t = self.transform(os_)
        img6 = torch.cat([od_t, os_t], dim=0)
        return img6, rc


def main():
    man_df = pd.read_csv(CONFIG['manifest_csv'])
    paths_all = man_df[['od_path', 'os_path']].values
    # Sanity check for dark images
    sanity_check_black_images(paths_all, sample_size=200, threshold=10.0)

    # Prefer cached stats; compute once if missing
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

    # ---------------- Demographic metadata maps ----------------
    demo_maps = build_demo_maps()

    # ---------------- Evaluation dataset for demographics ----------------
    eval_loader = None
    try:
        if demo_maps is not None and 'RegistrationCode' in man_df.columns:
            # include os_path if available to build 6-ch inputs
            cols = [c for c in ['od_path', 'os_path', 'RegistrationCode'] if c in man_df.columns]
            eval_df = man_df[cols].copy()
            # Drop rows without RegistrationCode
            eval_df = eval_df.dropna(subset=['RegistrationCode']).reset_index(drop=True)
            # Sample up to N examples
            max_eval = 1000
            if len(eval_df) > max_eval:
                eval_df = eval_df.sample(n=max_eval, random_state=42).reset_index(drop=True)
            eval_loader = DataLoader(EvalDataset(eval_df, trsf), batch_size=CONFIG['batch_size'], shuffle=False,
                                     num_workers=0)
    except Exception as e:
        logger.info(f"Eval setup failed: {e}")

    # Function to extract features and compute AUCs
    def evaluate_demographics(encoder, loader, maps):
        if loader is None or maps is None:
            return None
        encoder.eval()
        feats = []
        labels = {'sex': [], 'bmi_high': [], 'age_high': []}
        regs_all = []
        with torch.no_grad():
            for imgs, regcode in loader:
                imgs = imgs.to(DEVICE)
                imgs = match_encoder_input_channels(encoder, imgs)
                z = encoder(imgs)
                if z.ndim == 3:
                    z = z.mean(dim=1)
                feats.append(z.cpu().numpy())
                regs_all.extend(list(regcode))
                for rc in regcode:
                    rc_val = rc
                    labels['sex'].append(maps.get('sex_map', {}).get(rc_val, None))
                    labels['bmi_high'].append(maps.get('bmi_map', {}).get(rc_val, None))
                    labels['age_high'].append(maps.get('age_map', {}).get(rc_val, None))
        X = np.concatenate(feats, axis=0) if feats else np.empty((0, getattr(encoder, 'embed_dim', 0)))
        if X.shape[0] == 0:
            return None

        def fit_auc(y_cont_or_bin):
            y = np.array(y_cont_or_bin)
            mask = ~pd.isna(y)
            Xm = X[mask]
            ym = y[mask]
            if Xm.shape[0] < 10:
                return None
            unique_vals = np.unique(ym)
            if unique_vals.size > 2:
                thr = np.median(ym)
                ym = (ym >= thr).astype(int)
            else:
                ym = (ym == unique_vals.max()).astype(int)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xm)
            try:
                clf = LogisticRegression(max_iter=200)
                clf.fit(Xs, ym)
                y_prob = clf.predict_proba(Xs)[:, 1]
                auc = roc_auc_score(ym, y_prob)
                return float(auc)
            except Exception:
                return None

        return {
            'sex_auc': fit_auc(labels['sex']),
            'bmi_auc': fit_auc(labels['bmi_high']),
            'age_auc': fit_auc(labels['age_high'])
        }


def create_pca_plots(encoder, loader, maps, prefix):
    try:
        if loader is None or maps is None:
            return
        encoder.eval()
        feats = []
        regs = []
        with torch.no_grad():
            for imgs, regcode in loader:
                imgs = imgs.to(DEVICE)
                imgs = match_encoder_input_channels(encoder, imgs)
                z = encoder(imgs)
                if z.ndim == 3:
                    z = z.mean(dim=1)
                feats.append(z.cpu().numpy())
                regs.extend(list(regcode))
        if not feats:
            return
        X = np.concatenate(feats, axis=0)
        if X.shape[0] < 10:
            return
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_
        xlab = f'PC1 ({evr[0] * 100:.1f}%)'
        ylab = f'PC2 ({evr[1] * 100:.1f}%)'
        # Sex plot
        sex_vals = [maps.get('sex_map', {}).get(rc, None) for rc in regs]
        try:
            mask = np.array([v is not None for v in sex_vals])
            if mask.any():
                s = np.array([int(v) for v in np.array(sex_vals, dtype=object)[mask]])
                plt.figure(figsize=(6, 5))
                for label, color, name in [(0, 'tab:blue', 'Class 0'), (1, 'tab:orange', 'Class 1')]:
                    sel_idx = np.where(mask)[0][s == label]
                    plt.scatter(X2[sel_idx, 0], X2[sel_idx, 1], s=6, alpha=0.7, c=color, label=name)
                plt.legend(title='Sex')
                plt.title(f'PCA ({prefix}) - Sex')
                plt.xlabel(xlab);
                plt.ylabel(ylab)
                plt.tight_layout()
                plt.savefig(os.path.join(IMAGES_DIR, f'pca_{prefix}_sex.png'))
                plt.close()
        except Exception:
            pass

        # Continuous helper
        def cont_plot(vals, title, fname):
            v = np.array(vals, dtype=float)
            m = ~np.isnan(v)
            if not m.any():
                return
            norm = mcolors.Normalize(vmin=np.nanpercentile(v[m], 5), vmax=np.nanpercentile(v[m], 95))
            colors = cm.viridis(norm(v[m]))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(X2[m, 0], X2[m, 1], s=6, alpha=0.8, c=colors)
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax).set_label(title)
            ax.set_title(f'PCA ({prefix}) - {title}')
            ax.set_xlabel(xlab);
            ax.set_ylabel(ylab)
            fig.tight_layout()
            fig.savefig(os.path.join(IMAGES_DIR, fname))
            plt.close(fig)

        cont_plot([maps.get('age_map', {}).get(rc, np.nan) for rc in regs], 'Age', f'pca_{prefix}_age.png')
        cont_plot([maps.get('bmi_map', {}).get(rc, np.nan) for rc in regs], 'BMI', f'pca_{prefix}_bmi.png')
        logger.info(f'PCA ({prefix}) plots saved')
    except Exception as e:
        logger.info(f'PCA plotting failed: {e}')


# Always define stage, epoch, elapsed_time before use
stage = None
epoch = 1
elapsed_time = 0
checkpoint = load_checkpoint()
# Always prefer resuming from retina pretrain checkpoint if available
if checkpoint is not None:
    logger.info("Found retina pretrain checkpoint, resuming training from it.")
    stage = checkpoint.get('stage', None)
    epoch = checkpoint.get('epoch', 1)
    elapsed_time = 0  # Reset elapsed time on resume
    enc, pred = build_encoder_and_predictor(CONFIG)
    enc.load_state_dict(checkpoint['enc'])
    pred.load_state_dict(checkpoint['pred'])
    if CONFIG['use_lora']:
        freeze_lora_params(enc)
        logger.info("Encoder LoRA-only trainable params enabled.")
    else:
        for p in enc.parameters():
            p.requires_grad = True
        logger.info("Encoder set to full-train mode (no LoRA).")
else:
    # No retina pretrain checkpoint, start from ImageNet checkpoint
    imagenet_ckpt = os.path.expanduser('~/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar')
    if os.path.isfile(imagenet_ckpt):
        logger.info(f"No retina pretrain checkpoint found, initializing from ImageNet checkpoint: {imagenet_ckpt}")
    else:
        logger.info("No retina pretrain checkpoint or ImageNet checkpoint found, starting from scratch.")
    stage = None
    epoch = 1
    elapsed_time = 0
    enc, pred = build_encoder_and_predictor(CONFIG)
    # Only load ImageNet checkpoint if it exists
    if os.path.isfile(imagenet_ckpt):
        # Use the same logic as load_pretrained_encoder but with explicit path
        state = torch.load(imagenet_ckpt, map_location=DEVICE)
        filtered = {k: v for k, v in state.items()
                    if k.startswith('patch_embed.') or k.startswith('blocks.')
                    or k in ('cls_token', 'pos_embed')}
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 3 and enc.patch_embed.proj.weight.shape[1] == 6:
            filtered['patch_embed.proj.weight'] = w.repeat(1, 2, 1, 1)[:, :6]
        enc.load_state_dict(filtered, strict=False)
        logger.info("Pretrained ImageNet weights loaded into encoder")
    if CONFIG['use_lora']:
        freeze_lora_params(enc)
        logger.info("Encoder initialized with LoRA-only trainable params.")
    else:
        for p in enc.parameters():
            p.requires_grad = True
        logger.info("Encoder initialized in full-train mode (no LoRA).")
# Print data pipeline summary
print(
    f"Data pipeline: batch_size={CONFIG['batch_size']}, img_size={CONFIG['img_size']}, normalization=mean/std {CONFIG.get('mean', '[default]')}/{CONFIG.get('std', '[default]')}")
# Always run masked pretraining in this script
if stage is None or stage == 'pretrain':
    # Define manifest-derived paths and transforms before splitting
    man_df = pd.read_csv(CONFIG['manifest_csv'])
    paths_all = man_df[['od_path', 'os_path']].values
    stats_file = 'image_stats_pretrain.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        mean, std = stats['mean'], stats['std']
    else:
        mean, std = compute_image_stats(paths_all, CONFIG)
        with open(stats_file, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f)
    trsf = build_transforms(mean, std)
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
        # Build demographic maps and evaluation loader for PCA/AUC
    demo_maps = build_demo_maps()
    if demo_maps is not None and 'RegistrationCode' in man_df.columns:
        cols = [c for c in ['od_path', 'os_path', 'RegistrationCode'] if c in man_df.columns]
        eval_df = man_df[cols].dropna(subset=['RegistrationCode']).reset_index(drop=True)
        # Sample up to 1000 for speed
        if len(eval_df) > 1000:
            eval_df = eval_df.sample(n=1000, random_state=42).reset_index(drop=True)
        eval_loader = DataLoader(EvalDataset(eval_df, trsf), batch_size=CONFIG['batch_size'], shuffle=False,
                                 num_workers=0)
    else:
        eval_loader = None
    # Path for accumulating results
    metrics_path = os.path.join(OUTPUT_ROOT, 'retina_pretrain_epoch_metrics.json')
    try:
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                json.dump([], f)
    except Exception:
        pass
    # Diagnostics: overlap
    try:
        if eval_loader is not None and demo_maps is not None and 'sex_map' in demo_maps:
            regs = set(eval_loader.dataset.df.get('RegistrationCode', pd.Series([], dtype=str)))
            meta_regs = set(demo_maps['sex_map'].keys())
            logger.info(f"[DEMO] RegistrationCode overlap: {len(regs & meta_regs)} / {len(regs)}")
    except Exception:
        pass
    # BEFORE PCA and baseline AUC only if NOT resuming from retina checkpoint
    do_before = not (checkpoint is not None and stage == 'pretrain')
    if do_before:
        create_pca_plots(enc, eval_loader, demo_maps, prefix='before')
        # Baseline epoch-0 AUCs
        if eval_loader is not None and demo_maps is not None:
            try:
                base = evaluate_demographics(enc, eval_loader, demo_maps)
                if base:
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            hist = json.load(f)
                    else:
                        hist = []
                    base_entry = {'epoch': 0, 'train_mse': None, 'val_mse': None}
                    base_entry.update(base)
                    hist.append(base_entry)
                    with open(metrics_path, 'w') as f:
                        json.dump(hist, f, indent=2)
                    logger.info("Saved baseline (epoch 0) demographic AUCs")
            except Exception as e:
                logger.info(f"Baseline demographics failed: {e}")
    run_masked_pretrain(enc, pred, loader_train, loader_val, CONFIG, start_epoch=epoch, elapsed_time=elapsed_time,
                        demo_eval=(eval_loader, demo_maps, metrics_path))

if __name__ == '__main__':
    main() 