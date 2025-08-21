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
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
import matplotlib.pyplot as plt
import time
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    'lr': 2e-4,
    'weight_decay': 1e-2,
    'epochs': 50,
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

# Outputs (plots/json) directory
OUTPUT_ROOT = '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/outputs'
IMAGES_DIR = os.path.join(OUTPUT_ROOT, 'images')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

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
    # training/eval for supervised task (kept as-is)
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_eval_transforms(mean, std):
    # deterministic probe transform: no augmentation
    return transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
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
    for p in model.parameters():
        p.requires_grad = False
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


def match_encoder_input_channels(encoder: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
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
    if c > target_c:
        return imgs[:, :target_c, ...]
    reps = max(1, target_c // max(1, c))
    return imgs.repeat(1, reps, 1, 1)[:, :target_c, ...]


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
    # 80/20 split
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(idx)
    n_train = int(0.8 * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    paths_train, paths_val = paths[train_idx], paths[val_idx]
    feats_train, feats_val = feats[train_idx], feats[val_idx]
    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_val   = scaler.transform(feats_val) if len(feats_val) > 0 else feats_val
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
    sup_val   = DataLoader(SupDS(paths_val,   feats_val),   batch_size=config['batch_size'], shuffle=False,num_workers=config['num_workers'], pin_memory=True) if len(paths_val) > 0 else None
    sup_loaders = {'train': sup_train, 'val': sup_val}
    n_features = len(feature_cols)
    return sup_loaders, n_features, feature_cols


# ---------- Checkpoint Utilities ----------
CHECKPOINT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth'
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
    min_len = min(len(epochs), len(train_losses), len(val_losses))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_losses = val_losses[:min_len]
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
    state = dict(state)
    state['epoch'] = state.get('epoch', 1) + 1
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")
    save_loss_plot(train_losses, val_losses, epoch_list, 'checkpoint_loss.png', config, stage)
    logger.info("Checkpoint loss plot saved to checkpoint_loss.png")


def load_checkpoint(filename=CHECKPOINT_PATH):
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from {filename}")
        return torch.load(filename, map_location=DEVICE)
    return None

# ---------- NEW: Demographic helpers (global) ----------
def build_demo_maps():
    """Build maps RegistrationCode -> sex (coded), bmi (float), age (float)."""
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


def build_eval_loader(config):
    """Light eval set from manifest for demographic probing (deterministic transforms)."""
    try:
        man_df = pd.read_csv(config['manifest_csv'])
    except Exception as e:
        logger.info(f"Failed reading manifest for eval loader: {e}")
        return None
    cols = [c for c in ['od_path', 'os_path', 'RegistrationCode'] if c in man_df.columns]
    if 'RegistrationCode' not in cols:
        logger.info("Manifest lacks RegistrationCode; demographic baseline disabled")
        return None
    eval_df = man_df[cols].dropna(subset=['RegistrationCode']).reset_index(drop=True)
    if len(eval_df) == 0:
        logger.info("Empty eval_df; demographic baseline disabled")
        return None
    if len(eval_df) > 1000:
        eval_df = eval_df.sample(n=1000, random_state=42).reset_index(drop=True)

    class EvalDS(Dataset):
        def __init__(self, df, transform, img_size):
            self.df = df.reset_index(drop=True)
            self.t = transform
            self.img_size = img_size
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            od_path = row['od_path']; os_path = row.get('os_path', None)
            rc = str(row['RegistrationCode'])
            try:
                od = Image.open(od_path).convert('RGB')
            except Exception:
                od = Image.new('RGB', self.img_size, 'black')
            if os_path is not None:
                try:
                    os_ = Image.open(os_path).convert('RGB')
                except Exception:
                    os_ = Image.new('RGB', self.img_size, 'black')
            else:
                os_ = od.copy()
            od_t = self.t(od); os_t = self.t(os_)
            img6 = torch.cat([od_t, os_t], dim=0)
            return img6, rc

    # Deterministic, ImageNet-like normalization
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    eval_trsf = build_eval_transforms(mean, std)
    return DataLoader(EvalDS(eval_df, eval_trsf, config['img_size']),
                      batch_size=config['batch_size'], shuffle=False, num_workers=0)


def evaluate_demographics_cv(encoder, loader, maps, n_splits=5, random_state=42):
    """
    5-fold CV linear probe AUCs for sex / BMI(high) / Age(high).
    Returns mean AUC in legacy keys (sex_auc, bmi_auc, age_auc) plus *_auc_std, *_aucs (per-fold),
    and diagnostics (n, pos_rate).
    """
    if loader is None or maps is None:
        return None

    # ---- Extract embeddings once ----
    encoder.eval()
    feats, regs = [], []
    with torch.no_grad():
        for imgs, regcode in loader:
            imgs = imgs.to(DEVICE)
            imgs = match_encoder_input_channels(encoder, imgs)
            z = encoder(imgs)
            if z.ndim == 3:  # (B,T,D)
                z = z.mean(dim=1)
            feats.append(z.cpu().numpy())
            regs.extend(list(regcode))
    if not feats:
        return None
    X = np.concatenate(feats, axis=0)

    # ---- Build raw targets ----
    sex_raw = np.array([maps.get('sex_map', {}).get(r, np.nan) for r in regs], dtype=float)
    bmi_raw = np.array([maps.get('bmi_map', {}).get(r, np.nan) for r in regs], dtype=float)
    age_raw = np.array([maps.get('age_map', {}).get(r, np.nan) for r in regs], dtype=float)

    def cv_auc(kind, y_raw, continuous=False):
        # mask valid rows
        m = ~np.isnan(y_raw)
        Xv = X[m]
        yv = y_raw[m]

        res = {f'{kind}_auc': None, f'{kind}_auc_std': None, f'{kind}_aucs': [],
               f'{kind}_n': int(Xv.shape[0]), f'{kind}_pos_rate': None}

        if Xv.shape[0] < 20:  # need enough for 5 folds
            return res

        # labels
        if continuous:
            # compute split labels via global median; per-fold we re-threshold on train median
            y_for_split = (yv >= np.median(yv)).astype(int)
        else:
            u = np.unique(yv)
            if u.size < 2:
                return res
            yv = (yv == u.max()).astype(int)
            y_for_split = yv.copy()

        if len(np.unique(y_for_split)) < 2:
            return res

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        aucs, pos_rates = [], []

        for train_idx, val_idx in skf.split(Xv, y_for_split):
            Xtr, Xte = Xv[train_idx], Xv[val_idx]
            if continuous:
                thr = np.median(yv[train_idx])
                ytr = (yv[train_idx] >= thr).astype(int)
                yte = (yv[val_idx]   >= thr).astype(int)
            else:
                ytr = yv[train_idx]
                yte = yv[val_idx]

            # guard against single-class folds
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue

            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xte = scaler.transform(Xte)

            clf = LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                C=1.0,
                class_weight="balanced"
            )
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, prob))
            pos_rates.append(float(np.mean(yte)))

        if len(aucs) == 0:
            return res

        res[f'{kind}_auc'] = float(np.mean(aucs))
        res[f'{kind}_auc_std'] = float(np.std(aucs))
        res[f'{kind}_aucs'] = [float(a) for a in aucs]
        res[f'{kind}_pos_rate'] = float(np.mean(pos_rates)) if pos_rates else None
        return res

    out = {}
    out.update(cv_auc('sex', sex_raw, continuous=False))
    out.update(cv_auc('bmi', bmi_raw,  continuous=True))
    out.update(cv_auc('age', age_raw,  continuous=True))
    return out


def append_epoch_metrics(entry, output_root):
    """Append an entry (including baseline epoch 0) to the finetune metrics JSON."""
    os.makedirs(output_root, exist_ok=True)
    metrics_path = os.path.join(output_root, 'retina_finetune_epoch_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                hist = json.load(f)
        except Exception:
            hist = []
    else:
        hist = []
    hist.append(entry)
    with open(metrics_path, 'w') as f:
        json.dump(hist, f, indent=2)
    logger.info(f"Appended metrics to {metrics_path}")
# ---------- End NEW helpers ----------



def run_supervised_finetune(enc, head, sup_loaders, config, n_features, feature_cols, start_epoch=1, elapsed_time=0):
    import signal

    freeze_lora_params(enc)
    head.to(DEVICE)

    optimizer = AdamW([
        {'params': [p for n,p in enc.named_parameters() if p.requires_grad], 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.0}
    ], lr=config['lr'])

    # LR warmup + cosine
    from torch.optim.lr_scheduler import SequentialLR, LinearLR
    warmup_epochs = 5
    main_epochs = max(1, config['epochs'] - warmup_epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = get_grad_scaler()

    # ---- Demographics & eval dataset (same as your original local helpers) ----
    def build_demo_maps_local():
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

    demo_maps = build_demo_maps_local()

    # Build eval loader (same as your original)
    eval_loader = None
    try:
        man_df = pd.read_csv(config['manifest_csv'])
        cols = [c for c in ['od_path', 'os_path', 'RegistrationCode'] if c in man_df.columns]
        eval_df = man_df[cols].dropna(subset=['RegistrationCode']).reset_index(drop=True)

        class EvalDS(Dataset):
            def __init__(self, df, transform):
                self.df = df.reset_index(drop=True)
                self.t = transform
            def __len__(self): return len(self.df)
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                od_path = row['od_path']; os_path = row.get('os_path', None)
                rc = str(row['RegistrationCode']) if 'RegistrationCode' in row else None
                try:
                    od = Image.open(od_path).convert('RGB')
                except Exception:
                    od = Image.new('RGB', config['img_size'], 'black')
                if os_path is not None:
                    try:
                        os_ = Image.open(os_path).convert('RGB')
                    except Exception:
                        os_ = Image.new('RGB', config['img_size'], 'black')
                else:
                    os_ = od.copy()
                od_t = self.t(od); os_t = self.t(os_)
                img6 = torch.cat([od_t, os_t], dim=0)
                return img6, rc

        if len(eval_df) > 1000:
            eval_df = eval_df.sample(n=1000, random_state=42).reset_index(drop=True)
        mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
        eval_trsf = build_transforms(mean, std)
        eval_loader = DataLoader(EvalDS(eval_df, eval_trsf), batch_size=config['batch_size'], shuffle=False, num_workers=0)
    except Exception as e:
        logger.info(f"Eval loader not built: {e}")

    def evaluate_demographics_local(encoder, loader, maps):
        if loader is None or maps is None:
            return None
        encoder.eval()
        feats, regs = [], []
        with torch.no_grad():
            for imgs, rc in loader:
                imgs = imgs.to(DEVICE)
                imgs = match_encoder_input_channels(encoder, imgs)
                z = encoder(imgs)
                if z.ndim == 3:
                    z = z.mean(dim=1)
                feats.append(z.cpu().numpy())
                regs.extend(list(rc))
        if not feats:
            return None
        X = np.concatenate(feats, axis=0)

        def fit_auc(which):
            y = np.array([maps.get('sex_map',{}).get(r, np.nan) if which=='sex' else
                          maps.get('bmi_map',{}).get(r, np.nan) if which=='bmi' else
                          maps.get('age_map',{}).get(r, np.nan) for r in regs], dtype=float)
            m = ~np.isnan(y)
            if m.sum() < 10:
                return None
            if which in ('bmi','age'):
                thr = np.median(y[m])
                yb = (y[m] >= thr).astype(int)
            else:
                u = np.unique(y[m]); yb = (y[m] == u.max()).astype(int)
            Xs = StandardScaler().fit_transform(X[m])
            try:
                clf = LogisticRegression(max_iter=1000, solver='liblinear')
                clf.fit(Xs, yb)
                prob = clf.predict_proba(Xs)[:,1]
                return float(roc_auc_score(yb, prob))
            except Exception:
                return None

        return {'sex_auc': fit_auc('sex'), 'bmi_auc': fit_auc('bmi'), 'age_auc': fit_auc('age')}

    def create_pca_plots(encoder, loader, maps, prefix):
        try:
            if loader is None or maps is None:
                return
            encoder.eval()
            feats, regs = [], []
            with torch.no_grad():
                for imgs, rc in loader:
                    imgs = imgs.to(DEVICE)
                    imgs = match_encoder_input_channels(encoder, imgs)
                    z = encoder(imgs)
                    if z.ndim == 3:
                        z = z.mean(dim=1)
                    feats.append(z.cpu().numpy()); regs.extend(list(rc))
            if not feats:
                return
            X = np.concatenate(feats, axis=0)
            if X.shape[0] < 10:
                return
            Xs = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(Xs)
            evr = pca.explained_variance_ratio_
            xlab = f'PC1 ({evr[0]*100:.1f}%)'
            ylab = f'PC2 ({evr[1]*100:.1f}%)'
            # Sex
            sex_vals = [maps['sex_map'].get(str(r), None) for r in regs]
            try:
                mask = np.array([v is not None for v in sex_vals])
                if mask.any():
                    s = np.array([int(v) for v in np.array(sex_vals, dtype=object)[mask]])
                    plt.figure(figsize=(6,5))
                    for label, color, name in [(0,'tab:blue','Class 0'), (1,'tab:orange','Class 1')]:
                        sel = np.where(mask)[0][s==label]
                        plt.scatter(X2[sel,0], X2[sel,1], s=6, alpha=0.7, c=color, label=name)
                    plt.legend(title='Sex'); plt.title(f'PCA ({prefix}) - Sex')
                    plt.xlabel(xlab); plt.ylabel(ylab)
                    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, f'pca_{prefix}_sex_ft.png')); plt.close()
            except Exception:
                pass
            # Continuous
            def cont_plot(vals, title, fname):
                v = np.array(vals, dtype=float); m = ~np.isnan(v)
                if not m.any(): return
                norm = mcolors.Normalize(vmin=np.nanpercentile(v[m],5), vmax=np.nanpercentile(v[m],95))
                colors = cm.viridis(norm(v[m]))
                fig, ax = plt.subplots(figsize=(6,5))
                ax.scatter(X2[m,0], X2[m,1], s=6, alpha=0.8, c=colors)
                sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm); sm.set_array([])
                fig.colorbar(sm, ax=ax).set_label(title)
                ax.set_title(f'PCA ({prefix}) - {title}')
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                fig.tight_layout(); fig.savefig(os.path.join(IMAGES_DIR, fname)); plt.close(fig)
            cont_plot([maps['age_map'].get(str(r), np.nan) for r in regs], 'Age', f'pca_{prefix}_age_ft.png')
            cont_plot([maps['bmi_map'].get(str(r), np.nan) for r in regs], 'BMI', f'pca_{prefix}_bmi_ft.png')
            logger.info(f'PCA ({prefix}) plots saved (finetune)')
        except Exception as e:
            logger.info(f'PCA plotting failed (finetune {prefix}): {e}')

    # ------------- graceful shutdown plumbing -------------
    stop_now = False
    did_shutdown = False

    def _request_stop(signum, frame):
        nonlocal stop_now
        stop_now = True
        logger.info(f"Signal {signum} received; will stop after current step.")

    for sig in (getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGINT", None),
                getattr(signal, "SIGQUIT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _request_stop)
            except Exception:
                pass

    def _save_ckpt(last_completed_epoch, sup_train_losses, sup_val_losses, start_time_local):
        """Save checkpoint (up to last completed epoch) and a loss plot."""
        state = {
            'enc': enc.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'epoch': last_completed_epoch,  # last fully completed epoch
            'elapsed_time': elapsed_time + (time.time() - start_time_local),
            'stage': 'finetune',
        }
        torch.save(state, CHECKPOINT_PATH)
        logger.info(f"Checkpoint saved to {CHECKPOINT_PATH}")
        epoch_list = list(range(start_epoch, last_completed_epoch + 1)) if last_completed_epoch >= start_epoch else []
        save_loss_plot(sup_train_losses, sup_val_losses, epoch_list, 'checkpoint_loss.png', config, 'finetune')

    def _graceful_shutdown(last_completed_epoch, sup_train_losses, sup_val_losses, start_time_local):
        nonlocal did_shutdown
        if did_shutdown:
            return
        did_shutdown = True
        _save_ckpt(last_completed_epoch, sup_train_losses, sup_val_losses, start_time_local)
        try:
            create_pca_plots(enc, eval_loader, demo_maps, prefix='after')
        except Exception as e:
            logger.warning(f"PCA (after) failed during shutdown: {e}")

    # ---------------- training ----------------
    sup_train_losses, sup_val_losses = [], []
    sup_train_r2s, sup_val_r2s = [], []
    start_time = time.time()
    accum_steps = 2

    # PCA BEFORE finetune
    create_pca_plots(enc, eval_loader, demo_maps, prefix='before')

    for epoch in range(start_epoch, config['epochs'] + 1):
        enc.train(); head.train()
        train_losses = []
        train_targets = []
        train_preds = []
        optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(tqdm(sup_loaders['train'], desc=f"Sup E{epoch} Train")):
            imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
            with torch.cuda.amp.autocast():
                f = enc(imgs); preds = head(f); loss = F.mse_loss(preds, targets) / accum_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * accum_steps)
            train_targets.append(targets.detach().cpu().numpy())
            train_preds.append(preds.detach().cpu().numpy())

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()

            # time/signal stop mid-epoch => shutdown up to last *completed* epoch
            total_elapsed = elapsed_time + (time.time() - start_time)
            if stop_now or total_elapsed >= MAX_TRAIN_SECONDS:
                # don't append epoch-level metrics; epoch not complete
                _graceful_shutdown(epoch - 1, sup_train_losses, sup_val_losses, start_time)
                return

        # leftover step
        if len(sup_loaders['train']) % accum_steps != 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

        # --- compute train metrics
        train_targets_np = np.concatenate(train_targets, axis=0) if train_targets else np.array([])
        train_preds_np = np.concatenate(train_preds, axis=0) if train_preds else np.array([])
        train_mse = np.mean((train_preds_np - train_targets_np) ** 2, axis=0) if train_targets_np.size > 0 else np.array([])
        train_r2 = [r2_score(train_targets_np[:, i], train_preds_np[:, i]) if train_targets_np.shape[0] > 0 else float('nan')
                    for i in range(n_features)]
        sup_train_losses.append(float(np.mean(train_mse)) if train_mse.size else float('nan'))
        sup_train_r2s.append(train_r2)

        # --- validation
        enc.eval(); head.eval()
        val_losses = []; val_targets = []; val_preds = []
        if sup_loaders['val'] is not None:
            with torch.no_grad():
                for imgs, targets in sup_loaders['val']:
                    imgs = imgs.to(DEVICE); targets = targets.to(DEVICE)
                    with torch.cuda.amp.autocast():
                        f = enc(imgs); preds = head(f); loss = F.mse_loss(preds, targets)
                    val_losses.append(loss.item())
                    val_targets.append(targets.detach().cpu().numpy())
                    val_preds.append(preds.detach().cpu().numpy())

        val_targets_np = np.concatenate(val_targets, axis=0) if val_targets else np.array([])
        val_preds_np   = np.concatenate(val_preds, axis=0)   if val_preds   else np.array([])
        val_mse = np.mean((val_preds_np - val_targets_np) ** 2, axis=0) if val_targets_np.size > 0 else np.array([])
        val_r2  = [r2_score(val_targets_np[:, i], val_preds_np[:, i]) if val_targets_np.shape[0] > 0 else float('nan')
                   for i in range(n_features)]
        sup_val_losses.append(float(np.mean(val_mse)) if val_mse.size else float('nan'))
        sup_val_r2s.append(val_r2)

        # Logging
        logger.info(f"Sup E{epoch} Train MSE={np.mean(train_mse):.4f} Val MSE={np.mean(val_mse):.4f}")
        for i, feat in enumerate(feature_cols):
            logger.info(f"  Feature {feat}: Train MSE={train_mse[i]:.4f} Val MSE={val_mse[i]:.4f} "
                        f"Train R2={train_r2[i]:.4f} Val R2={val_r2[i]:.4f}")

        # Demographic AUCs per epoch (same local probe as before)
        demo = evaluate_demographics_local(enc, eval_loader, demo_maps)
        if demo:
            logger.info(f"  Demographics AUCs: sex={demo.get('sex_auc')}, "
                        f"bmi={demo.get('bmi_auc')}, age={demo.get('age_auc')}")
            # Append to JSON
            try:
                metrics_path = os.path.join(OUTPUT_ROOT, 'retina_finetune_epoch_metrics.json')
                top5 = []
                if val_targets_np.size > 0 and len(val_mse) > 0:
                    top_idx = np.argsort(val_mse)[:5]
                    for idx in top_idx:
                        top5.append({
                            'feature': feature_cols[idx],
                            'val_mse': float(val_mse[idx]),
                            'val_r2': float(val_r2[idx])
                        })
                entry = {
                    'epoch': epoch,
                    'train_mse_mean': float(np.mean(train_mse)) if train_mse.size else None,
                    'val_mse_mean': float(np.mean(val_mse)) if val_mse.size else None,
                    'top5_features_by_val_mse': top5,
                    **demo,
                }
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        hist = json.load(f)
                else:
                    hist = []
                hist.append(entry)
                with open(metrics_path, 'w') as f:
                    json.dump(hist, f, indent=2)
                logger.info(f"Saved finetune epoch metrics to {metrics_path}")
            except Exception as e:
                logger.info(f"Failed to write finetune metrics JSON: {e}")

        scheduler.step()

        # after finishing the epoch: if time/signal, shutdown with this epoch persisted + PCA
        total_elapsed = elapsed_time + (time.time() - start_time)
        if stop_now or total_elapsed >= MAX_TRAIN_SECONDS:
            _graceful_shutdown(epoch, sup_train_losses, sup_val_losses, start_time)
            return

    # normal completion ? also make the PCA
    try:
        create_pca_plots(enc, eval_loader, demo_maps, prefix='after')
    except Exception as e:
        logger.warning(f"PCA (after) failed at normal completion: {e}")

    # R2 & MSE plots (unchanged)
    epochs = list(range(start_epoch, config['epochs']+1))
    sup_train_r2s_np = np.array(sup_train_r2s)
    sup_val_r2s_np = np.array(sup_val_r2s)
    for i, feat in enumerate(feature_cols):
        plt.figure()
        plt.plot(epochs, sup_train_r2s_np[:, i], label='Train R2')
        plt.plot(epochs, sup_val_r2s_np[:, i], label='Val R2')
        plt.xlabel('Epoch'); plt.ylabel('R2'); plt.title(f'R2 for {feat}')
        plt.legend(); plt.tight_layout(); plt.savefig(f'r2_{feat}.png'); plt.close()
    plt.figure()
    plt.plot(epochs, sup_train_losses, label='Train MSE')
    plt.plot(epochs, sup_val_losses, label='Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('MSE per Epoch')
    plt.legend(); plt.tight_layout(); plt.savefig('mse_per_epoch.png'); plt.close()
    logger.info("Saved R2 and MSE plots.")
    logger.info("Supervised fine-tuning complete.")


IMAGENET_CKPT = os.path.expanduser('~/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar')

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
        elapsed_time = 0
        logger.info(f"Resuming from checkpoint at epoch {epoch}, elapsed_time {elapsed_time:.2f}s, stage {stage}")
        if 'enc' in checkpoint:
            enc, _ = build_encoder_and_predictor(CONFIG)
            enc.load_state_dict(checkpoint['enc'])
            logger.info("Encoder weights loaded from checkpoint.")
            freeze_lora_params(enc)
            logger.info("Encoder LoRA parameters unfrozen.")
        else:
            logger.info("No encoder checkpoint found. Building new encoder.")
            enc, _ = build_encoder_and_predictor(CONFIG)
            if os.path.isfile(IMAGENET_CKPT):
                CONFIG['pretrained_ckpt'] = IMAGENET_CKPT
                logger.info(f"Using ImageNet checkpoint: {IMAGENET_CKPT}")
            else:
                CONFIG['pretrained_ckpt'] = None
                logger.info("No ImageNet checkpoint found. Starting from scratch.")
            load_pretrained_encoder(enc, CONFIG)
            freeze_lora_params(enc)
            logger.info("Encoder LoRA parameters frozen.")
    else:
        stage = None; epoch = 1; elapsed_time = 0
        logger.info("No checkpoint found. Building new encoder.")
        enc, _ = build_encoder_and_predictor(CONFIG)
        if os.path.isfile(IMAGENET_CKPT):
            CONFIG['pretrained_ckpt'] = IMAGENET_CKPT
            logger.info(f"Using ImageNet checkpoint: {IMAGENET_CKPT}")
        else:
            CONFIG['pretrained_ckpt'] = None
            logger.info("No ImageNet checkpoint found. Starting from scratch.")
        load_pretrained_encoder(enc, CONFIG)
        freeze_lora_params(enc)
        logger.info("Encoder LoRA parameters frozen.")

    # Supervised fine-tune stage
    if RUN_SUPERVISED_FINETUNE and (stage is None or stage == 'finetune'):
        sup_loaders, n_features, feature_cols = build_supervised_loaders(CONFIG, trsf)
        head = RegressionHead(CONFIG['embed_dim'], n_features)
        if checkpoint is not None and stage == 'finetune':
            enc.load_state_dict(checkpoint['enc'])
            head.load_state_dict(checkpoint['head'])

        # ---------- Baseline demographics AUCs at epoch 0 (5-fold CV) ----------
        do_before = not (checkpoint is not None and stage == 'finetune')
        if do_before:
            try:
                demo_maps = build_demo_maps()
                eval_loader = build_eval_loader(CONFIG)
                baseline = evaluate_demographics_cv(enc, eval_loader, demo_maps)
                if baseline:
                    def fmt(d, k):
                        m, s = d.get(f"{k}_auc"), d.get(f"{k}_auc_std")
                        return "NA" if m is None else (f"{m:.3f}±{(s or 0):.3f}")
                    logger.info(
                        f"Baseline Demographics 5-fold AUCs (epoch 0): "
                        f"sex={fmt(baseline,'sex')}, bmi={fmt(baseline,'bmi')}, age={fmt(baseline,'age')}"
                    )
                    entry = {
                        'epoch': 0,
                        'train_mse_mean': None,
                        'val_mse_mean': None,
                        'top5_features_by_val_mse': [],
                        **baseline,
                    }
                    append_epoch_metrics(entry, OUTPUT_ROOT)
                else:
                    logger.info("Baseline demographics AUCs not computed (missing maps or eval loader).")
            except Exception as e:
                logger.info(f"Baseline demographics evaluation failed: {e}")
        # ----------------------------------------------------------------------

        run_supervised_finetune(enc, head, sup_loaders, CONFIG, n_features, feature_cols, start_epoch=epoch, elapsed_time=elapsed_time)

if __name__ == '__main__':
    main()
