# Fine-tune I-JEPA on retina images for disease classification (cardiovascular, hypertension, diabetes)
# This script should be run from the JEPA directory, not inside ijepa
# --- Add I-JEPA src to sys.path so 'src' imports work ---
print(">>>> Script running <<<<")
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
print("debug1")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from ijepa.src.datasets.retina import RetinaDataset
from ijepa.src.models.vision_transformer import vit_huge  # Use ViT-H/14 as backbone
# LoRA config (manual, not peft)
USE_LORA = False  # Set to False to disable LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
print("debug2")
import numpy as np
import pandas as pd
import collections
import shutil
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import math
import random
# Remove joblib.Parallel and delayed. Run the 10-trial evaluation loop sequentially.
# Remove the joblib import if no longer used.
from torch.cuda.amp import autocast
import torch
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# --- MAIN MODE FLAG ---
CHECK = True  # Set to True to run the embedding/knn check and exit

# Class-Balanced Loss based on Effective Number of Samples
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, samples_per_cls, loss_type='softmax', device='cpu'):
        super().__init__()
        self.beta = beta
        self.samples_per_cls = samples_per_cls
        self.loss_type = loss_type
        self.device = device
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(self.samples_per_cls)
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        print(f"ClassBalancedLoss: effective_num={effective_num}, weights={weights}")

    def forward(self, logits, labels):
        if self.loss_type == 'softmax':
            return nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            raise NotImplementedError("Only softmax (cross-entropy) supported.")

# Custom Dataset for fine-tuning
class RetinaFineTuneDataset(Dataset):
    def __init__(self, df, label_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        od_img = Image.open(row['od_path']).convert('RGB')
        os_img = Image.open(row['os_path']).convert('RGB')
        if self.transform:
            od_img = self.transform(od_img)
            os_img = self.transform(os_img)
        # Ensure both are tensors before concatenation
        if not isinstance(od_img, torch.Tensor):
            od_img = transforms.ToTensor()(od_img)
        if not isinstance(os_img, torch.Tensor):
            os_img = transforms.ToTensor()(os_img)
        img = torch.cat([od_img, os_img], dim=0)
        label = int(row[self.label_col])
        # Add future_flag for the current disease
        future_flag = int(row.get(f'future_{DISEASE_TO_TRAIN}', 0))
        return img, label, future_flag

# Set custom temp directory to avoid filling up system TMPDIR
os.environ['TMPDIR'] = '/home/gavrielh/temp'
os.environ['TEMP'] = '/home/gavrielh/temp'
os.environ['TMP'] = '/home/gavrielh/temp'

# --- CONFIG ---
# Set base directory for all file paths
BASE_DIR = os.path.expanduser("~/PycharmProjects/MSc_Thesis/JEPA")
MANIFEST = os.path.join(BASE_DIR, "retina_manifest.csv")
DIAGNOSIS = os.path.join(BASE_DIR, "retina_patient_diagnosis.csv")
DISEASES = [
    "Obesity",  # cardiovascular
    "Essential hypertension",    # hypertension
    "Diabetes mellitus, type unspecified"  # diabetes
]
# --- NEW: Disease and label type selection ---
DISEASE_TO_TRAIN = "Essential hypertension"  # Change as needed
LABEL_TYPE = "future"  # 'prevalent' or 'future'
LABEL_COL = f"{LABEL_TYPE}_{DISEASE_TO_TRAIN}"
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = (336, 336)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_PATCH_SIZE = 14
VIT_EMBED_DIM = 1280  # for vit_huge
PRETRAINED_CKPT = os.path.join(BASE_DIR, "pretrained_IN/IN22K-vit.h.14-900e.pth.tar")
# Make sure CHECK is disabled for fine-tuning



# Set the desired number of samples for each group in one place
N_POS_PREVALENT = 161
N_POS_FUTURE_TRAIN = 200
N_POS_FUTURE_TEST = 170
N_NEG = 800 - N_POS_PREVALENT - N_POS_FUTURE_TRAIN

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 3 channels per eye
])

# --- MODEL ---
# Instantiate ViT-H/14 backbone for 6-channel input (with new image size)
backbone = vit_huge(patch_size=VIT_PATCH_SIZE, img_size=IMG_SIZE, in_chans=6,
                   use_lora=USE_LORA, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

# Optionally load pretrained weights (if available)
if PRETRAINED_CKPT is not None and os.path.isfile(PRETRAINED_CKPT):
    state_dict = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
    # If in_chans != 3, adapt the first conv layer weights
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

# Sweep: robust CLS extraction everywhere
# In all places where feats or feats_np is used to extract the CLS token, use:
# if feats_np.ndim == 1:
#     cls_token = feats_np
# else:
#     cls_token = feats_np[:, 0]

# Add a trainable classification head using the [CLS] token
num_classes = 2
class RetinaClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(VIT_EMBED_DIM, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats_np = feats.cpu().numpy() if hasattr(feats, 'cpu') else feats
        feats_np = np.atleast_2d(feats_np)
        cls_token = feats_np[:, 0]
        return self.head(torch.from_numpy(cls_token).to(x.device) if isinstance(cls_token, np.ndarray) else cls_token)

# --- MAIN EXECUTION LOGIC ---
if CHECK:
    # --- CHECK MODE: Single block for all logic ---
    import sys
    # Ensure images directory exists
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Define image sizes (divisible by 14, from 224 to ~1512)
    sizes = [224, 336, 448, 560, 672, 784, 896, 1008, 1120, 1232, 1344, 1456]
    pr_aucs = []
    roc_aucs = []
    pr_aucs_all = []  # For error bars
    roc_aucs_all = []
    valid_sizes = []
    for IMG_SIZE_CUR in sizes:
        print(f"\n[CHECK MODE] Running for image size {IMG_SIZE_CUR}x{IMG_SIZE_CUR} ...")
        batch_size_cur = 1 if IMG_SIZE_CUR >= 560 else 32
        N_TRIALS = 10
        def run_trial(trial):
            import torch
            try:
                transform_cur = transforms.Compose([
                    transforms.Resize((IMG_SIZE_CUR, IMG_SIZE_CUR)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                backbone = vit_huge(patch_size=VIT_PATCH_SIZE, img_size=(IMG_SIZE_CUR, IMG_SIZE_CUR), in_chans=6)
                if PRETRAINED_CKPT is not None and os.path.isfile(PRETRAINED_CKPT):
                    state_dict = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
                    if hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'proj'):
                        w = state_dict.get('patch_embed.proj.weight', None)
                        if w is not None and w.shape[1] == 3 and backbone.patch_embed.proj.weight.shape[1] == 6:
                            w6 = w.repeat(1, 2, 1, 1)[:, :6]
                            state_dict['patch_embed.proj.weight'] = w6
                    try:
                        backbone.load_state_dict(state_dict, strict=False)
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
                backbone = backbone.to(DEVICE)
                backbone.eval()
                PREVALENT_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
                df = pd.read_csv(PREVALENT_CSV)
                np.random.seed(42 + trial)
                neg_indices = df.index[(df[f'prevalent_{DISEASE_TO_TRAIN}'] == 0) & (df[f'future_{DISEASE_TO_TRAIN}'] == 0)].tolist()
                pos_indices = df.index[df[f'future_{DISEASE_TO_TRAIN}'] == 1].tolist()
                val_neg_indices = np.random.choice(neg_indices, 800, replace=False)
                val_pos_indices = np.random.choice(pos_indices, 200, replace=False)
                val_indices = np.concatenate([val_neg_indices, val_pos_indices])
                check_df = df.loc[val_indices]
                check_labels = np.array([0]*800 + [1]*200)
                check_dataset = RetinaFineTuneDataset(check_df, LABEL_COL, transform=transform_cur)
                check_loader = DataLoader(check_dataset, batch_size=batch_size_cur, shuffle=False, num_workers=NUM_WORKERS)
                all_embeds = []
                with torch.no_grad():
                    for batch in check_loader:
                        imgs = batch[0]
                        imgs = imgs.to(DEVICE)
                        feats = backbone(imgs)
                        if isinstance(feats, tuple):
                            feats = feats[0]
                        feats_np = feats.cpu().numpy()
                        feats_np = np.atleast_2d(feats_np)
                        cls_token = feats_np[:, 0]
                        all_embeds.append(cls_token)
                all_embeds = np.concatenate(all_embeds, axis=0)
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(all_embeds, check_labels)
                pred_probs = knn.predict_proba(all_embeds)[:, 1]
                roc_auc = roc_auc_score(check_labels, pred_probs)
                precision, recall, _ = precision_recall_curve(check_labels, pred_probs)
                pr_auc = auc(recall, precision)
                result = (pr_auc, roc_auc)
                # Explicitly delete large variables and free memory
                del all_embeds, check_loader, check_dataset
                if 'backbone' in locals():
                    del backbone
                if 'feats' in locals():
                    del feats
                if 'feats_np' in locals():
                    del feats_np
                if 'cls_token' in locals():
                    del cls_token
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                return result
            except RuntimeError as e:
                import torch
                if 'out of memory' in str(e) or 'allocate memory' in str(e):
                    print(f'OOM at size {IMG_SIZE_CUR}, trial {trial}, skipping this trial...')
                    torch.cuda.empty_cache()
                    return None
                else:
                    raise
        results = [run_trial(trial) for trial in range(N_TRIALS)]
        # Filter out None results (OOM trials)
        results = [r for r in results if r is not None]
        if len(results) == 0:
            print(f"All trials OOM for image size {IMG_SIZE_CUR}, skipping this size.")
            continue
        pr_aucs_trials, roc_aucs_trials = zip(*results)
        pr_aucs.append(np.mean(pr_aucs_trials))
        roc_aucs.append(np.mean(roc_aucs_trials))
        pr_aucs_all.append(pr_aucs_trials)
        roc_aucs_all.append(roc_aucs_trials)
        valid_sizes.append(IMG_SIZE_CUR)
        print(f"KNN ROC AUC (800N/200P, mean±std): {np.mean(roc_aucs_trials):.3f} ± {np.std(roc_aucs_trials):.3f}")
        print(f"KNN PR AUC (800N/200P, mean±std): {np.mean(pr_aucs_trials):.3f} ± {np.std(pr_aucs_trials):.3f}")
    # After all sizes, plot PR AUC and ROC AUC vs. image size (only for valid_sizes)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    pr_aucs_arr = np.array(pr_aucs_all)
    roc_aucs_arr = np.array(roc_aucs_all)
    pr_means = np.array([np.mean(x) for x in pr_aucs_all])
    pr_stds = np.array([np.std(x) for x in pr_aucs_all])
    roc_means = np.array([np.mean(x) for x in roc_aucs_all])
    roc_stds = np.array([np.std(x) for x in roc_aucs_all])
    plt.errorbar(valid_sizes, pr_means, yerr=pr_stds, fmt='o-r', label='PR AUC')
    plt.errorbar(valid_sizes, roc_means, yerr=roc_stds, fmt='o-b', label='ROC AUC')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('AUC')
    plt.title(f'PR AUC and ROC AUC vs. Image Size ({LABEL_COL})')
    plt.legend()
    plt.grid(True)
    summary_plot_path = os.path.join(IMAGES_DIR, f'auc_vs_size_{LABEL_COL.replace(" ", "_")}.png')
    plt.savefig(summary_plot_path)
    print(f"Saved summary plot of AUC vs. image size as '{summary_plot_path}'")
    sys.exit(0)

# --- FINE-TUNING MODE (with LoRA) ---
if not CHECK:
    print("Loading data for fine-tuning (LoRA)...")
    PREVALENT_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
    df = pd.read_csv(PREVALENT_CSV)
    # Load evaluation indices and exclude from training
    neg_indices_all = df.index[(df[f'prevalent_{DISEASE_TO_TRAIN}'] == 0) & (df[f'future_{DISEASE_TO_TRAIN}'] == 0)].tolist()
    pos_indices_prevalent = df.index[df[f'prevalent_{DISEASE_TO_TRAIN}'] == 1].tolist()
    pos_indices_future = df.index[df[f'future_{DISEASE_TO_TRAIN}'] == 1].tolist()
    # Remove eval indices code and sample validation set on-the-fly
    np.random.shuffle(neg_indices_all)
    val_neg_indices = neg_indices_all[:800]
    train_neg_indices = neg_indices_all[800:]
    np.random.shuffle(pos_indices_future)
    val_pos_indices = pos_indices_future[:200]
    train_pos_indices_future = pos_indices_future[200:]
    # Select all prevalent positives for training
    np.random.shuffle(pos_indices_prevalent)
    # Build train/val indices
    train_indices = np.array(train_neg_indices + train_pos_indices_future + pos_indices_prevalent)
    val_indices = np.array(list(val_neg_indices) + list(val_pos_indices))
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    # For training, label both prevalent and future as 1
    train_df[LABEL_COL] = ((train_df[f'prevalent_{DISEASE_TO_TRAIN}'] == 1) | (train_df[f'future_{DISEASE_TO_TRAIN}'] == 1)).astype(int)
    print(f'Train set label distribution ({LABEL_COL}):', train_df[LABEL_COL].value_counts())
    print(f'Val set label distribution ({LABEL_COL}):', val_df[LABEL_COL].value_counts())

    # Create custom datasets
    train_dataset = RetinaFineTuneDataset(train_df, LABEL_COL, transform=transform)
    val_dataset = RetinaFineTuneDataset(val_df, LABEL_COL, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Compute class-balanced weights
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    samples_per_cls = [np.sum(np.array(train_labels) == i) for i in range(2)]
    CB_BETA = 0.999  # Typical value for large datasets
    criterion = ClassBalancedLoss(beta=CB_BETA, samples_per_cls=samples_per_cls, device=DEVICE)
    # Update optimizer to only use trainable parameters
    model = RetinaClassifier(backbone, num_classes).to(DEVICE)
    # Add DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    def print_trainable_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    print_trainable_parameters(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None

    # --- PRE-LoRA KNN PR AUC (calculate ONCE before training loop) ---
    # Instantiate a backbone with use_lora=False, load same weights
    from ijepa.src.models.vision_transformer import vit_huge
    pre_lora_backbone = vit_huge(patch_size=VIT_PATCH_SIZE, img_size=IMG_SIZE, in_chans=6, use_lora=False)
    if PRETRAINED_CKPT is not None and os.path.isfile(PRETRAINED_CKPT):
        state_dict = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
        if hasattr(pre_lora_backbone, 'patch_embed') and hasattr(pre_lora_backbone.patch_embed, 'proj'):
            w = state_dict.get('patch_embed.proj.weight', None)
            if w is not None and w.shape[1] == 3 and pre_lora_backbone.patch_embed.proj.weight.shape[1] == 6:
                w6 = w.repeat(1, 2, 1, 1)[:, :6]
                state_dict['patch_embed.proj.weight'] = w6
        try:
            pre_lora_backbone.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading checkpoint for pre-LoRA backbone: {e}")
    pre_lora_backbone = pre_lora_backbone.to(DEVICE)
    pre_lora_backbone.eval()
    # Extract embeddings for validation set
    all_pre_lora_embeds = []
    all_val_labels = []
    all_val_future_flags = []
    val_loader_for_pre_lora = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for imgs, labels, future_flags in val_loader_for_pre_lora:
            imgs = imgs.to(DEVICE)
            feats = pre_lora_backbone(imgs)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats_np = feats.cpu().numpy()
            feats_np = np.atleast_2d(feats_np)
            cls_token = feats_np[:, 0]
            all_pre_lora_embeds.append(cls_token)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_future_flags.extend(future_flags.cpu().numpy())
    all_pre_lora_embeds = np.concatenate(all_pre_lora_embeds, axis=0)
    all_val_labels = np.array(all_val_labels)
    all_val_future_flags = np.array(all_val_future_flags)
    future_mask = all_val_future_flags == 1
    future_mask_idx = np.where(future_mask)[0].tolist()
    # KNN PR AUC for future positives
    if np.sum(future_mask) > 0:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import precision_recall_curve, auc
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(all_pre_lora_embeds, all_val_labels)
        pred_probs = knn.predict_proba(all_pre_lora_embeds)[:, 1]
        future_probs_knn = pred_probs[future_mask_idx]
        future_labels_knn = all_val_labels[future_mask_idx]
        precision_knn, recall_knn, _ = precision_recall_curve(future_labels_knn, future_probs_knn)
        pr_auc_knn = auc(recall_knn, precision_knn)
        print(f"[VAL] PR AUC for Future Dx (PRE-LoRA KNN, computed ONCE): {pr_auc_knn:.3f}")
    else:
        print("[VAL] No Future Dx samples in validation set for KNN PR AUC.")

    # --- TRAINING AND EVALUATION CODE ---
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n--- TRAINING (Epoch {epoch+1}/{EPOCHS}) ---")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            imgs, labels, *_ = batch
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        print(f"\n--- VALIDATION (Epoch {epoch+1}/{EPOCHS}) ---")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []
        all_val_logits = []
        all_val_future_flags = []
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels, future_flags = batch
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_logits.extend(logits.cpu().numpy())
                all_val_future_flags.extend(future_flags.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # PR AUC for Future Dx class in validation set (AFTER LoRA)
        from sklearn.metrics import precision_recall_curve, auc
        logits_arr = np.array(all_val_logits)
        future_mask = np.array(all_val_future_flags) == 1
        if logits_arr.ndim == 1 or (logits_arr.ndim == 2 and logits_arr.shape[1] == 1):
            probs = torch.sigmoid(torch.tensor(logits_arr)).numpy().reshape(-1)
        else:
            probs = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
        if np.sum(future_mask) > 0:
            if logits_arr.ndim == 1 or (logits_arr.ndim == 2 and logits_arr.shape[1] == 1):
                future_probs = probs[future_mask]
            else:
                future_probs = probs[future_mask, 1]
            future_mask_idx = np.where(future_mask)[0].tolist()
            future_labels = np.array(all_val_labels)[future_mask_idx]
            precision, recall, _ = precision_recall_curve(future_labels, future_probs)
            pr_auc = auc(recall, precision)
            print(f"[VAL] PR AUC for Future Dx (AFTER LoRA): {pr_auc:.3f}")
        else:
            print("[VAL] No Future Dx samples in validation set for PR AUC.")

        if val_acc > best_acc:
            best_acc = val_acc
            #torch.save(model.state_dict(), "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/best_retina_classifier.pt")

    # PR AUC for Prevalent Hypertension (positive class) in validation set
    from sklearn.metrics import precision_recall_curve, auc
    probs = torch.softmax(torch.tensor(all_val_logits), dim=1).numpy()
    precision, recall, _ = precision_recall_curve(all_val_labels, probs[:, 1])
    pr_auc = auc(recall, precision)
    print(f"[VAL] PR AUC for Prevalent Hypertension: {pr_auc:.3f}")

    # PR AUC for Future Dx class in validation set
    future_mask = np.array(all_val_future_flags) == 1
    if np.sum(future_mask) > 0:
        future_labels = all_val_labels[future_mask]
        future_probs = probs[future_mask, 1]
        precision, recall, _ = precision_recall_curve(future_labels, future_probs)
        pr_auc = auc(recall, precision)
        print(f"[VAL] PR AUC for Future Dx: {pr_auc:.3f}")
    else:
        print("[VAL] No Future Dx samples in validation set for PR AUC.")

    print("Training complete. Best val acc:", best_acc)
    # Clean up temp directory after training
    try:
        shutil.rmtree('/home/gavrielh/temp')
        print('Cleaned up /home/gavrielh/temp')
    except Exception as e:
        print(f'Could not clean up /home/gavrielh/temp: {e}')

    # --- POST-FINETUNE LORA EVALUATION (AUC vs. image size, 10x trials, like CHECK mode) ---
    print("\n[POST-FINETUNE] Evaluating frozen LoRA model on hold-out sets at multiple image sizes...")
    # After fine-tuning, freeze all model params before post-finetune evaluation
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    sizes = [224, 336, 448, 560, 672, 784, 896, 1008, 1120, 1232, 1344, 1456]
    pr_aucs = []
    roc_aucs = []
    pr_aucs_all = []
    roc_aucs_all = []
    valid_sizes = []
    for IMG_SIZE_CUR in sizes:
        print(f"[POST-FINETUNE] Running for image size {IMG_SIZE_CUR}x{IMG_SIZE_CUR} ...")
        batch_size_cur = 1
        N_TRIALS = 5
        def run_trial(trial):
            import torch
            try:
                transform_cur = transforms.Compose([
                    transforms.Resize((IMG_SIZE_CUR, IMG_SIZE_CUR)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                PREVALENT_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
                df = pd.read_csv(PREVALENT_CSV)
                np.random.seed(42 + trial)
                neg_indices = df.index[(df[f'prevalent_{DISEASE_TO_TRAIN}'] == 0) & (df[f'future_{DISEASE_TO_TRAIN}'] == 0)].tolist()
                pos_indices = df.index[df[f'future_{DISEASE_TO_TRAIN}'] == 1].tolist()
                val_neg_indices = np.random.choice(neg_indices, 800, replace=False)
                val_pos_indices = np.random.choice(pos_indices, 200, replace=False)
                val_indices = np.concatenate([val_neg_indices, val_pos_indices])
                check_df = df.loc[val_indices]
                check_labels = np.array([0]*800 + [1]*200)
                check_dataset = RetinaFineTuneDataset(check_df, LABEL_COL, transform=transform_cur)
                check_loader = DataLoader(check_dataset, batch_size=batch_size_cur, shuffle=False, num_workers=NUM_WORKERS)
                all_embeds = []
                with torch.no_grad():
                    for batch in check_loader:
                        imgs = batch[0]  # Only use images, ignore label/future_flag
                        imgs = imgs.to(DEVICE)
                        feats = model.backbone(imgs)
                        if isinstance(feats, tuple):
                            feats = feats[0]
                        feats_np = feats.cpu().numpy()
                        feats_np = np.atleast_2d(feats_np)
                        cls_token = feats_np[:, 0]
                        all_embeds.append(cls_token)
                all_embeds = np.concatenate(all_embeds, axis=0)
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(all_embeds, check_labels)
                pred_probs = knn.predict_proba(all_embeds)[:, 1]
                roc_auc = roc_auc_score(check_labels, pred_probs)
                precision, recall, _ = precision_recall_curve(check_labels, pred_probs)
                pr_auc = auc(recall, precision)
                result = (pr_auc, roc_auc)
                # Explicitly delete large variables and free memory
                del all_embeds, check_loader, check_dataset
                if 'backbone' in locals():
                    del backbone
                if 'feats' in locals():
                    del feats
                if 'feats_np' in locals():
                    del feats_np
                if 'cls_token' in locals():
                    del cls_token
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                return result
            except RuntimeError as e:
                import torch
                if 'out of memory' in str(e) or 'allocate memory' in str(e):
                    print(f'OOM at size {IMG_SIZE_CUR}, trial {trial}, skipping this trial...')
                    torch.cuda.empty_cache()
                    return None
                else:
                    raise
        results = [run_trial(trial) for trial in range(N_TRIALS)]
        # Filter out None results (OOM trials)
        results = [r for r in results if r is not None]
        if len(results) == 0:
            print(f"All trials OOM for image size {IMG_SIZE_CUR}, skipping this size.")
            continue
        pr_aucs_trials, roc_aucs_trials = zip(*results)
        pr_aucs.append(np.mean(pr_aucs_trials))
        roc_aucs.append(np.mean(roc_aucs_trials))
        pr_aucs_all.append(pr_aucs_trials)
        roc_aucs_all.append(roc_aucs_trials)
        valid_sizes.append(IMG_SIZE_CUR)
        print(f"[POST-FINETUNE] KNN ROC AUC (800N/200P, mean±std): {np.mean(roc_aucs_trials):.3f} ± {np.std(roc_aucs_trials):.3f}")
        print(f"[POST-FINETUNE] KNN PR AUC (800N/200P, mean±std): {np.mean(pr_aucs_trials):.3f} ± {np.std(pr_aucs_trials):.3f}")
    # After all sizes, plot PR AUC and ROC AUC vs. image size (only for valid_sizes)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    pr_aucs_arr = np.array(pr_aucs_all)
    roc_aucs_arr = np.array(roc_aucs_all)
    pr_means = np.array([np.mean(x) for x in pr_aucs_all])
    pr_stds = np.array([np.std(x) for x in pr_aucs_all])
    roc_means = np.array([np.mean(x) for x in roc_aucs_all])
    roc_stds = np.array([np.std(x) for x in roc_aucs_all])
    plt.errorbar(valid_sizes, pr_means, yerr=pr_stds, fmt='o-r', label='PR AUC')
    plt.errorbar(valid_sizes, roc_means, yerr=roc_stds, fmt='o-b', label='ROC AUC')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('AUC')
    plt.title(f'PR AUC and ROC AUC vs. Image Size (LoRA, {LABEL_COL})')
    plt.legend()
    plt.grid(True)
    summary_plot_path = os.path.join(IMAGES_DIR, f'auc_vs_size_lora_{LABEL_COL.replace(" ", "_")}.png')
    plt.savefig(summary_plot_path)
    print(f"Saved summary plot of AUC vs. image size (LoRA) as '{summary_plot_path}'")