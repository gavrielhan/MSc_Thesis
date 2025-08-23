#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import atexit
import argparse
import random
import json
import logging
import numpy as np
import pandas as pd
import time
import inspect
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# --- IJepa imports (your local repo layout) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from ijepa.src.masks.multiblock import MaskCollator
from ijepa.src.masks.utils import apply_masks


# ---------------------- Repro ----------------------
SEED = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()


# ---------------------- CLI ----------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='Fast dev run: 1 epoch, tiny timeout, limit train/val batches.')
    ap.add_argument('--max-steps', type=int, default=None,
                    help='Stop after this many optimizer steps (across all epochs).')
    ap.add_argument('--limit-train-batches', type=int, default=None,
                    help='Use at most N train batches per epoch.')
    ap.add_argument('--limit-val-batches', type=int, default=None,
                    help='Use at most N val batches per epoch.')
    ap.add_argument('--save-every-steps', type=int, default=0,
                    help='If >0, write an interim checkpoint every N optimizer steps.')
    ap.add_argument('--override-epochs', type=int, default=None,
                    help='Override CONFIG["epochs"].')
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='Override checkpoint path (e.g., ./ckpt_smoke.pth).')
    return ap.parse_args()


# ---------------------- Checkpoints ----------------------
CHECKPOINT_PATH = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_pretrain_newrun.pth'
MAX_TRAIN_SECONDS =11 * 3600  # 11.5 hours

OUTPUT_ROOT = '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/outputs'
IMAGES_DIR = os.path.join(OUTPUT_ROOT, 'images')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)


def _ensure_writable_or_fallback(path_like: str) -> str:
    """If target dir is not writable, fall back to CWD with same filename."""
    if not path_like:
        return os.path.join(os.getcwd(), 'checkpoint_pretrain.pth')
    target_dir = os.path.dirname(path_like) or '.'
    test_path = os.path.join(target_dir, f'.write_test_{os.getpid()}')
    try:
        with open(test_path, 'w') as f:
            f.write('ok'); f.flush(); os.fsync(f.fileno())
        os.remove(test_path)
        return path_like
    except Exception as e:
        fb = os.path.join(os.getcwd(), os.path.basename(path_like))
        logger.warning(f"[CKPT] Dir not writable ({e}). Falling back to {fb}")
        return fb


def _atomic_save_torch(state_obj, target_path):
    d = os.path.dirname(target_path) or '.'
    base = os.path.basename(target_path)
    tmp = os.path.join(d, f'.{base}.tmp-{os.getpid()}')
    try:
        with open(tmp, 'wb') as f:
            torch.save(state_obj, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, target_path)
        return True, target_path
    except Exception as e:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        return False, str(e)


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
    filename = _ensure_writable_or_fallback(filename)
    ok, info = _atomic_save_torch(state, filename)
    if ok: logger.info(f"[CKPT] Checkpoint saved to {info}")
    else:  logger.error(f"[CKPT] FAILED to save checkpoint: {info}")
    try:
        save_loss_plot(train_losses, val_losses, epoch_list,
                       os.path.join(IMAGES_DIR, 'checkpoint_pretrain_loss.png'),
                       config, stage)
        logger.info("[CKPT] Loss plot saved")
    except Exception as e:
        logger.warning(f"[CKPT] Loss plot failed: {e}")


def load_checkpoint(filename=CHECKPOINT_PATH):
    filename = _ensure_writable_or_fallback(filename)
    if os.path.isfile(filename):
        logger.info(f"[CKPT] Loading checkpoint from {filename}")
        return torch.load(filename, map_location=DEVICE)
    return None


# ---------------------- AMP scaler ----------------------
def get_grad_scaler():
    # Prefer bf16 autocast w/o scaler
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return None
    amp_grad_scaler = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if amp_grad_scaler is not None:
        sig = inspect.signature(amp_grad_scaler)
        return amp_grad_scaler(device_type='cuda') if 'device_type' in sig.parameters else amp_grad_scaler()
    # Fallback (older torch)
    return getattr(torch.cuda, 'amp', None).GradScaler() if hasattr(getattr(torch.cuda, 'amp', None), 'GradScaler') else None


# ---------------------- Data helpers ----------------------
def compute_image_stats(paths, config):
    class SimpleDS(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            od, _ = self.paths[idx]
            img = Image.open(od).convert('RGB')
            img = transforms.Resize(config['img_size'])(img)
            return transforms.ToTensor()(img)

    loader = DataLoader(SimpleDS(paths),
                        batch_size=config['batch_size'],
                        num_workers=config['num_workers'],
                        pin_memory=True)
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
                if len(examples) < 3: examples.append(od_path)
            if os_mean < threshold:
                black_count += 1
                if len(examples) < 3: examples.append(os_path)
        except Exception:
            black_count += 2
            total_checked += 2
    pct = (black_count / max(1, total_checked)) * 100.0
    logger.info(f"[SANITY] Checked {total_checked} eyes; black/dark: {black_count} ({pct:.1f}%).")
    if examples:
        logger.info(f"[SANITY] Example dark files: {examples}")
    if pct > 50.0:
        logger.warning("[SANITY] >50% of sampled images are very dark; verify dataset paths/preprocessing.")
    return pct


def match_encoder_input_channels(encoder: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Adjust batch tensor channels to match encoder's expected in_chans."""
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
    reps = target_c // c if c else 1
    reps = max(1, reps)
    return imgs.repeat(1, reps, 1, 1)[:, :target_c, ...]


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
        if 'lora_' in n: p.requires_grad = True


def build_masked_dataloaders(paths, transform, config):
    class RetinaImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            od_path, os_path = self.paths[idx]
            try:     od = Image.open(od_path).convert('RGB')
            except:  od = Image.new('RGB', CONFIG['img_size'], 'black')
            try:     os_ = Image.open(os_path).convert('RGB')
            except:  os_ = Image.new('RGB', CONFIG['img_size'], 'black')
            od_t = self.transform(od); os_t = self.transform(os_)
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


# ---------------------- Demo metrics (5-fold CV) ----------------------
def evaluate_demographics_cv(encoder, loader, maps, n_splits=5, random_state=42):
    if loader is None or maps is None:
        return None

    encoder.eval()
    feats, regs = [], []
    with torch.no_grad():
        for imgs, regcode in loader:
            imgs = imgs.to(DEVICE)
            # match chans
            try:
                target_c = encoder.patch_embed.proj.weight.shape[1]
                c = imgs.shape[1]
                if target_c == 6 and c == 3:
                    imgs = torch.cat([imgs, imgs], dim=1)
                elif target_c == 3 and c == 6:
                    imgs = 0.5 * (imgs[:, 0:3] + imgs[:, 3:6])
                elif c > target_c:
                    imgs = imgs[:, :target_c]
                elif c < target_c:
                    imgs = imgs.repeat(1, target_c // c + 1, 1, 1)[:, :target_c]
            except Exception:
                pass
            z = encoder(imgs)
            if z.ndim == 3:
                z = z.mean(dim=1)
            feats.append(z.cpu().numpy())
            regs.extend(list(regcode))
    if not feats:
        return None
    X = np.concatenate(feats, axis=0)

    sex_raw = np.array([maps.get('sex_map', {}).get(r, np.nan) for r in regs], dtype=float)
    bmi_raw = np.array([maps.get('bmi_map', {}).get(r, np.nan) for r in regs], dtype=float)
    age_raw = np.array([maps.get('age_map', {}).get(r, np.nan) for r in regs], dtype=float)

    def cv_auc(kind, y_raw, continuous=False):
        m = ~np.isnan(y_raw)
        Xv = X[m]; yv = y_raw[m]
        res = {f'{kind}_auc': None, f'{kind}_auc_std': None, f'{kind}_aucs': [],
               f'{kind}_n': int(Xv.shape[0]), f'{kind}_pos_rate': None}
        if Xv.shape[0] < 20:
            return res

        if continuous:
            y_for_split = (yv >= np.median(yv)).astype(int)
        else:
            u = np.unique(yv)
            if u.size < 2: return res
            yv = (yv == u.max()).astype(int)
            y_for_split = yv.copy()

        if len(np.unique(y_for_split)) < 2:
            return res

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        aucs = []; pos_rates = []
        for tr_idx, te_idx in skf.split(Xv, y_for_split):
            Xtr, Xte = Xv[tr_idx], Xv[te_idx]
            if continuous:
                thr = np.median(yv[tr_idx])
                ytr = (yv[tr_idx] >= thr).astype(int)
                yte = (yv[te_idx] >= thr).astype(int)
            else:
                ytr = yv[tr_idx]; yte = yv[te_idx]

            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue

            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
            clf = LogisticRegression(solver="liblinear", max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, prob))
            pos_rates.append(float(np.mean(yte)))

        if len(aucs) == 0: return res
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


def _fmt_demo_cv(prefix, d):
    def one(kind):
        m = d.get(f'{kind}_auc')
        s = d.get(f'{kind}_auc_std')
        n = d.get(f'{kind}_n')
        pr = d.get(f'{kind}_pos_rate')
        folds = d.get(f'{kind}_aucs')
        if m is None:
            return f"{kind}: AUC=NA, n={n}, pos={'NA' if pr is None else f'{pr:.1%}'}"
        folds_str = "NA" if not folds else "[" + ", ".join(f"{x:.3f}" for x in folds) + "]"
        pr_str = "NA" if pr is None else f"{pr:.1%}"
        return f"{kind}: AUC={m:.3f}±{(s or 0):.3f} folds={folds_str}, n={n}, pos={pr_str}"
    prefix_str = f"{prefix} " if prefix else ""
    return f"[DEMO] {prefix_str}" + " | ".join([one('sex'), one('bmi'), one('age')])


# ---------------------- PCA helper ----------------------
def create_pca_plots(encoder, loader, maps, prefix):
    try:
        if loader is None or maps is None:
            return
        encoder.eval()
        feats, regs = [], []
        with torch.no_grad():
            for imgs, regcode in loader:
                imgs = imgs.to(DEVICE)
                imgs = match_encoder_input_channels(encoder, imgs)
                z = encoder(imgs)
                if z.ndim == 3:
                    z = z.mean(dim=1)
                feats.append(z.cpu().numpy())
                regs.extend(list(regcode))
        if not feats: return
        X = np.concatenate(feats, axis=0)
        if X.shape[0] < 10: return
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_
        xlab = f'PC1 ({evr[0]*100:.1f}%)'
        ylab = f'PC2 ({evr[1]*100:.1f}%)'
        # Sex
        sex_vals = [maps.get('sex_map', {}).get(rc, None) for rc in regs]
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
                plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, f'pca_{prefix}_sex.png')); plt.close()
        except Exception:
            pass

        def cont_plot(vals, title, fname):
            v = np.array(vals, dtype=float); m = ~np.isnan(v)
            if not m.any(): return
            norm = mcolors.Normalize(vmin=np.nanpercentile(v[m], 5), vmax=np.nanpercentile(v[m], 95))
            colors = cm.viridis(norm(v[m]))
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(X2[m,0], X2[m,1], s=6, alpha=0.8, c=colors)
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm); sm.set_array([])
            fig.colorbar(sm, ax=ax).set_label(title)
            ax.set_title(f'PCA ({prefix}) - {title}')
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
            fig.tight_layout(); fig.savefig(os.path.join(IMAGES_DIR, fname)); plt.close(fig)

        cont_plot([maps.get('age_map', {}).get(rc, np.nan) for rc in regs], 'Age', f'pca_{prefix}_age.png')
        cont_plot([maps.get('bmi_map', {}).get(rc, np.nan) for rc in regs], 'BMI', f'pca_{prefix}_bmi.png')
        logger.info(f'PCA ({prefix}) plots saved')
    except Exception as e:
        logger.info(f'PCA plotting failed: {e}')


# ---------------------- Demo metadata maps ----------------------
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
            sex_map = (df_meta.set_index('RegistrationCode')[sex_col]
                       .astype('category').cat.codes.to_dict())
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


# ---------------------- Train loop ----------------------
def run_masked_pretrain(enc, pred, loader_train, loader_val, config,
                        start_epoch=1, elapsed_time=0, demo_eval=None):
    import signal
    import atexit
    import copy

    # --- Optimizer/scheduler/AMP
    params = [p for _, p in list(enc.named_parameters()) + list(pred.named_parameters()) if p.requires_grad]
    optimizer = AdamW([{'params': params, 'weight_decay': config['weight_decay']}], lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = get_grad_scaler()

    # --- EMA target (teacher)
    target_enc = copy.deepcopy(enc).to(DEVICE)
    for p in target_enc.parameters():
        p.requires_grad = False
    ema_momentum = 0.99

    # --- optional demographics tuple
    eval_loader = eval_maps = metrics_path = None
    if isinstance(demo_eval, tuple) and len(demo_eval) == 3:
        eval_loader, eval_maps, metrics_path = demo_eval

    # --- time bookkeeping & buffers
    masked_train_losses, masked_val_losses = [], []
    start_time = time.time()
    accum_steps = 2
    last_completed_epoch = start_epoch - 1

    # --- graceful shutdown helpers (TIME/SIGNAL ONLY)
    stop_now = False
    did_shutdown = False

    def _request_stop(signum, frame):
        nonlocal stop_now
        stop_now = True
        logger.info(f"[CKPT] Signal {signum} received; will stop after current step.")

    for sig in (getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGINT",  None),
                getattr(signal, "SIGQUIT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _request_stop)
            except Exception:
                pass

    def _pca_after():
        if eval_loader is None or eval_maps is None:
            return
        try:
            enc.eval()
            feats, regs = [], []
            with torch.no_grad():
                for imgs, regcode in eval_loader:
                    imgs = imgs.to(DEVICE)
                    imgs = match_encoder_input_channels(enc, imgs)
                    z = enc(imgs)
                    if z.ndim == 3:
                        z = z.mean(dim=1)
                    feats.append(z.cpu().numpy())
                    regs.extend(list(regcode))
            if not feats:
                return
            X = np.concatenate(feats, axis=0)
            Xs = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(Xs)
            evr = pca.explained_variance_ratio_
            xlab = f'PC1 ({evr[0]*100:.1f}%)'; ylab = f'PC2 ({evr[1]*100:.1f}%)'
            # Sex (discrete)
            try:
                sex_vals = [eval_maps.get('sex_map', {}).get(str(r), None) for r in regs]
                mask = np.array([v is not None for v in sex_vals])
                if mask.any():
                    s = np.array([int(v) for v in np.array(sex_vals, dtype=object)[mask]])
                    plt.figure(figsize=(6,5))
                    for label, color, name in [(0,'tab:blue','Class 0'), (1,'tab:orange','Class 1')]:
                        sel = np.where(mask)[0][s==label]
                        plt.scatter(X2[sel,0], X2[sel,1], s=6, alpha=0.7, c=color, label=name)
                    plt.legend(title='Sex'); plt.title('PCA (after) - Sex')
                    plt.xlabel(xlab); plt.ylabel(ylab)
                    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, 'pca_after_sex.png')); plt.close()
            except Exception:
                pass
            # Continuous overlays
            def _cont(vals, title, fname):
                v = np.array(vals, dtype=float); m = ~np.isnan(v)
                if not m.any(): return
                norm = mcolors.Normalize(vmin=np.nanpercentile(v[m],5), vmax=np.nanpercentile(v[m],95))
                colors = cm.viridis(norm(v[m]))
                fig, ax = plt.subplots(figsize=(6,5))
                ax.scatter(X2[m,0], X2[m,1], s=6, alpha=0.8, c=colors)
                sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm); sm.set_array([])
                fig.colorbar(sm, ax=ax).set_label(title)
                ax.set_title(f'PCA (after) - {title}')
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                fig.tight_layout(); fig.savefig(os.path.join(IMAGES_DIR, fname)); plt.close(fig)
            _cont([eval_maps.get('age_map', {}).get(str(r), np.nan) for r in regs], 'Age', 'pca_after_age.png')
            _cont([eval_maps.get('bmi_map', {}).get(str(r), np.nan) for r in regs], 'BMI', 'pca_after_bmi.png')
            logger.info("Saved PCA (after) plots.")
        except Exception as e:
            logger.warning(f"PCA (after) failed: {e}")

    def _graceful_shutdown(epoch_done, tr_losses, va_losses, started_at):
        """TIME/SIGNAL-based save: persists last fully-completed epoch and exits cleanly."""
        nonlocal did_shutdown
        if did_shutdown:
            return
        did_shutdown = True
        total_elapsed = elapsed_time + (time.time() - started_at)
        epoch_list = list(range(start_epoch, max(start_epoch, epoch_done) + 1))
        state = {
            'enc': enc.state_dict(),
            'pred': pred.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch_done,
            'elapsed_time': total_elapsed,
            'stage': 'pretrain',
        }
        save_checkpoint(state, tr_losses, va_losses, epoch_list, config, 'pretrain')
        _pca_after()

    # also save if the interpreter exits unexpectedly
    atexit.register(lambda: _graceful_shutdown(last_completed_epoch, masked_train_losses, masked_val_losses, start_time))

    # --- training
    best_val = float('inf')
    for epoch in range(start_epoch, config['epochs'] + 1):
        enc.train(); pred.train()
        train_losses = []
        optimizer.zero_grad()

        for i, (imgs, m_enc, m_pred) in enumerate(tqdm(loader_train, desc=f"Pretrain E{epoch}")):
            imgs  = imgs.to(DEVICE)
            m_enc = [m.to(DEVICE) for m in m_enc]
            m_pred= [m.to(DEVICE) for m in m_pred]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(torch.cuda.is_available() and
                                             hasattr(torch.cuda, "is_bf16_supported") and
                                             torch.cuda.is_bf16_supported())):
                z   = enc(imgs, m_enc)
                out = pred(z, m_enc, m_pred)
                with torch.no_grad():
                    h = F.layer_norm(target_enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                loss = F.smooth_l1_loss(out, t) / accum_steps

            if scaler is not None: scaler.scale(loss).backward()
            else:                   loss.backward()

            train_losses.append(loss.item() * accum_steps)

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # EMA update
                with torch.no_grad():
                    for p_online, p_target in zip(enc.parameters(), target_enc.parameters()):
                        p_target.data.mul_(ema_momentum).add_((1 - ema_momentum) * p_online.data)

            # --- TIME-BASED STOP (mid-epoch)
            total_elapsed = elapsed_time + (time.time() - start_time)
            if stop_now or total_elapsed >= MAX_TRAIN_SECONDS:
                # save up to the LAST COMPLETED epoch
                _graceful_shutdown(epoch - 1, masked_train_losses, masked_val_losses, start_time)
                return

        # leftover optimizer step if needed
        if len(loader_train) % accum_steps != 0:
            if scaler is not None: scaler.step(optimizer); scaler.update()
            else:                  optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                for p_online, p_target in zip(enc.parameters(), target_enc.parameters()):
                    p_target.data.mul_(ema_momentum).add_((1 - ema_momentum) * p_online.data)

        # validation
        enc.eval(); pred.eval()
        val_losses = []
        with torch.no_grad():
            for vi, (imgs, m_enc, m_pred) in enumerate(loader_val):
                imgs  = imgs.to(DEVICE)
                m_enc = [m.to(DEVICE) for m in m_enc]
                m_pred= [m.to(DEVICE) for m in m_pred]
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                        enabled=(torch.cuda.is_available() and
                                                 hasattr(torch.cuda, "is_bf16_supported") and
                                                 torch.cuda.is_bf16_supported())):
                    z = enc(imgs, m_enc)
                    out = pred(z, m_enc, m_pred)
                    h = F.layer_norm(target_enc(imgs), (z.size(-1),))
                    t = apply_masks(h, m_pred).repeat(len(m_enc), 1, 1)
                    l = F.smooth_l1_loss(out, t)
                val_losses.append(l.item())

        mt = float(np.mean(train_losses))
        mv = float(np.mean(val_losses))
        masked_train_losses.append(mt)
        masked_val_losses.append(mv)
        logger.info(f"Pretrain E{epoch} Train MSE={mt:.4f} Val MSE={mv:.4f}")
        scheduler.step()
        best_val = min(best_val, mv)
        last_completed_epoch = epoch

        # per-epoch demo CV (optional)
        if eval_loader is not None and eval_maps is not None:
            try:
                demo_res = evaluate_demographics_cv(enc, eval_loader, eval_maps)
                if demo_res:
                    logger.info(_fmt_demo_cv(f"E{epoch}", demo_res))
                    if metrics_path:
                        try:
                            hist = []
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r') as f: hist = json.load(f)
                            entry = {'epoch': epoch, 'train_mse': mt, 'val_mse': mv, **demo_res}
                            hist.append(entry)
                            with open(metrics_path, 'w') as f: json.dump(hist, f, indent=2)
                        except Exception as e:
                            logger.info(f"Failed to write metrics JSON: {e}")
            except Exception as e:
                logger.info(f"Demographic eval failed: {e}")

        # --- TIME-BASED STOP (right after epoch completes)
        total_elapsed = elapsed_time + (time.time() - start_time)
        if stop_now or total_elapsed >= MAX_TRAIN_SECONDS:
            _graceful_shutdown(epoch, masked_train_losses, masked_val_losses, start_time)
            return

    # normal completion
    logger.info("Pretraining completed.")
    if eval_loader is not None and eval_maps is not None:
        try:
            create_pca_plots(enc, eval_loader, eval_maps, prefix='after')
        except Exception as e:
            logger.info(f"PCA (after) failed: {e}")



# ---------------------- Main ----------------------
# Load JSON config first (then patch it)
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# set/override some defaults for pretrain
CONFIG['img_size'] = (602, 602)
CONFIG['epochs'] = 100 if 'epochs' not in CONFIG else CONFIG['epochs']
CONFIG['num_workers'] = 0
CONFIG['batch_size'] = 1
CONFIG['use_lora'] = False
CONFIG['lr'] = 5.0e-5

def main():
    args = _parse_args()

    # CLI overrides
    if args.override_epochs is not None:
        CONFIG['epochs'] = int(args.override_epochs)
    CONFIG['DEV_SMOKE'] = bool(args.smoke)
    CONFIG['LIMIT_TRAIN_BATCHES'] = args.limit_train_batches
    CONFIG['LIMIT_VAL_BATCHES'] = args.limit_val_batches
    CONFIG['MAX_STEPS'] = args.max_steps
    CONFIG['SAVE_EVERY_STEPS'] = args.save_every_steps

    global CHECKPOINT_PATH, MAX_TRAIN_SECONDS
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint
        os.makedirs(os.path.dirname(CHECKPOINT_PATH) or '.', exist_ok=True)
    if CONFIG['DEV_SMOKE']:
        MAX_TRAIN_SECONDS = 60  # 1 minute
        CONFIG['epochs'] = min(CONFIG['epochs'], 1)
        logger.info("[DEV] Smoke mode: 1 epoch, tiny timeout, optional batch limits.")

    # --- data manifest & transforms
    man_df = pd.read_csv(CONFIG['manifest_csv'])
    paths_all = man_df[['od_path', 'os_path']].values
    sanity_check_black_images(paths_all, sample_size=200, threshold=10.0)

    stats_file = 'image_stats_pretrain.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f: stats = json.load(f)
        mean, std = stats['mean'], stats['std']
        logger.info(f"Loaded cached image stats from {stats_file}")
    else:
        mean, std = compute_image_stats(paths_all, CONFIG)
        with open(stats_file, 'w') as f: json.dump({'mean': mean, 'std': std}, f)
        logger.info(f"Computed and saved image stats to {stats_file}")
    trsf = build_transforms(mean, std)

    # --- demo maps & eval loader
    demo_maps = build_demo_maps()
    eval_loader = None
    try:
        if demo_maps is not None and 'RegistrationCode' in man_df.columns:
            cols = [c for c in ['od_path', 'os_path', 'RegistrationCode'] if c in man_df.columns]
            eval_df = man_df[cols].copy()
            eval_df = eval_df.dropna(subset=['RegistrationCode']).reset_index(drop=True)
            max_eval = 1000
            if len(eval_df) > max_eval:
                eval_df = eval_df.sample(n=max_eval, random_state=42).reset_index(drop=True)

            class EvalDataset(Dataset):
                def __init__(self, df, transform):
                    self.df = df.reset_index(drop=True)
                    self.t = transform
                def __len__(self): return len(self.df)
                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    od_path = row['od_path']; os_path = row.get('os_path', None)
                    rc = str(row['RegistrationCode'])
                    try: od = Image.open(od_path).convert('RGB')
                    except Exception: od = Image.new('RGB', CONFIG['img_size'], 'black')
                    if os_path is not None:
                        try: os_ = Image.open(os_path).convert('RGB')
                        except Exception: os_ = Image.new('RGB', CONFIG['img_size'], 'black')
                    else:
                        os_ = od.copy()
                    od_t = self.t(od); os_t = self.t(os_)
                    img6 = torch.cat([od_t, os_t], dim=0)
                    return img6, rc

            eval_loader = DataLoader(EvalDataset(eval_df, trsf),
                                     batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    except Exception as e:
        logger.info(f"Eval setup failed: {e}")

    # --- build/restore models
    stage = None
    epoch = 1
    elapsed_time = 0
    checkpoint = load_checkpoint()

    if checkpoint is not None:
        logger.info("Found checkpoint; resuming.")
        stage = checkpoint.get('stage', None)
        epoch = checkpoint.get('epoch', 1) + 1
        elapsed_time = 0
        enc, pred = build_encoder_and_predictor(CONFIG)
        enc.load_state_dict(checkpoint['enc'])
        pred.load_state_dict(checkpoint['pred'])
        if CONFIG.get('use_lora', False): freeze_lora_params(enc)
        else:
            for p in enc.parameters(): p.requires_grad = True
    else:
        imagenet_ckpt = os.path.expanduser('~/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar')
        enc, pred = build_encoder_and_predictor(CONFIG)
        if os.path.isfile(imagenet_ckpt):
            logger.info(f"Init encoder from ImageNet checkpoint: {imagenet_ckpt}")
            state = torch.load(imagenet_ckpt, map_location=DEVICE)
            filtered = {k: v for k, v in state.items()
                        if k.startswith('patch_embed.') or k.startswith('blocks.')
                        or k in ('cls_token', 'pos_embed')}
            w = filtered.get('patch_embed.proj.weight')
            if w is not None and w.shape[1] == 3 and enc.patch_embed.proj.weight.shape[1] == 6:
                filtered['patch_embed.proj.weight'] = w.repeat(1, 2, 1, 1)[:, :6]
            enc.load_state_dict(filtered, strict=False)
            logger.info("Loaded ImageNet weights into encoder")
        if CONFIG.get('use_lora', False): freeze_lora_params(enc)
        else:
            for p in enc.parameters(): p.requires_grad = True

    # Print data pipeline summary
    logger.info(f"Data pipeline: batch_size={CONFIG['batch_size']}, img_size={CONFIG['img_size']}")

    # --- loaders
    tr_paths, val_paths = train_test_split(
        paths_all, test_size=0.2,
        random_state=CONFIG.get('seed', SEED)
    )
    paths = {'train': tr_paths, 'val': val_paths}
    loader_train, loader_val = build_masked_dataloaders(paths, trsf, CONFIG)
    load_pretrained_encoder(enc, CONFIG)
    if checkpoint is not None and stage == 'pretrain':
        enc.load_state_dict(checkpoint['enc'])
        pred.load_state_dict(checkpoint['pred'])
        elapsed_time = 0

    # --- metrics JSON path
    metrics_path = os.path.join(OUTPUT_ROOT, 'retina_pretrain_epoch_metrics.json')
    try:
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f: json.dump([], f)
    except Exception:
        pass

    # Overlap diagnostic
    try:
        if eval_loader is not None and demo_maps is not None and 'sex_map' in demo_maps:
            regs = set(eval_loader.dataset.df.get('RegistrationCode', pd.Series([], dtype=str)))
            meta_regs = set(demo_maps['sex_map'].keys())
            logger.info(f"[DEMO] RegistrationCode overlap: {len(regs & meta_regs)} / {len(regs)}")
    except Exception:
        pass

    # BEFORE PCA + baseline AUCs (only if NOT resuming an active pretrain)
    do_before = not (checkpoint is not None and stage == 'pretrain')
    if do_before:
        try:
            create_pca_plots(enc, eval_loader, demo_maps, prefix='before')
        except Exception as e:
            logger.info(f"PCA(before) failed: {e}")
        if eval_loader is not None and demo_maps is not None:
            try:
                base = evaluate_demographics_cv(enc, eval_loader, demo_maps)
                if base:
                    logger.info(_fmt_demo_cv("Baseline (epoch 0)", base))
                    hist = []
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f: hist = json.load(f)
                    base_entry = {'epoch': 0, 'train_mse': None, 'val_mse': None}
                    base_entry.update(base)
                    hist.append(base_entry)
                    with open(metrics_path, 'w') as f: json.dump(hist, f, indent=2)
                    logger.info("Saved baseline (epoch 0) demographic AUCs")
            except Exception as e:
                logger.info(f"Baseline demographics failed: {e}")

    # --- Train
    run_masked_pretrain(
        enc, pred, loader_train, loader_val, CONFIG,
        start_epoch=epoch, elapsed_time=elapsed_time,
        demo_eval=(eval_loader, demo_maps, metrics_path)
    )


if __name__ == '__main__':
    main()
