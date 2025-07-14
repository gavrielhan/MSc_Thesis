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
USE_LORA = True  # Set to False to disable LoRA
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
CHECK = False

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

# Manual LoRA freezing: freeze all except LoRA and head
if not CHECK and USE_LORA:
    lora_param_count = 0
    for name, param in backbone.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            lora_param_count += param.numel()
        else:
            param.requires_grad = False
    print(f"LoRA enabled: Only LoRA and head parameters will be trainable. LoRA params: {lora_param_count}")
elif not CHECK:
    print("LoRA disabled: All backbone parameters will be trainable.")

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
        cls_token = feats[:, 0]
        return self.head(cls_token)

# --- MAIN CHECK MODE ---
if CHECK:
    import os
    # Ensure images directory exists
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Define image sizes (divisible by 14, from 224 to ~1512)
    sizes = [224, 336, 448, 560, 672, 784, 896, 1008, 1120, 1232, 1344, 1456]
    pr_aucs = []
    roc_aucs = []
    for IMG_SIZE_CUR in sizes:
        print(f"\n[CHECK MODE] Running for image size {IMG_SIZE_CUR}x{IMG_SIZE_CUR} ...")
        # Use batch size 1 for large images
        batch_size_cur = 1 if IMG_SIZE_CUR >= 560 else 32
        try:
            # Update transform for this size
            transform_cur = transforms.Compose([
                transforms.Resize((IMG_SIZE_CUR, IMG_SIZE_CUR)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            # Re-instantiate backbone for this size
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
                    print(f"Loaded pretrained weights from {PRETRAINED_CKPT}")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
            else:
                print("No pretrained checkpoint found or specified.")
            backbone = backbone.to(DEVICE)
            backbone.eval()
            PREVALENT_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
            df = pd.read_csv(PREVALENT_CSV)
            # Select 800 true negatives (both prevalent and future == 0) and 200 positives for the selected label
            neg_indices = df.index[(df[f'prevalent_{DISEASE_TO_TRAIN}'] == 0) & (df[f'future_{DISEASE_TO_TRAIN}'] == 0)].tolist()
            pos_indices = df.index[df[LABEL_COL] == 1].tolist()
            np.random.shuffle(neg_indices)
            np.random.shuffle(pos_indices)
            neg_indices_800 = neg_indices[:800]
            pos_indices_200 = pos_indices[:200]
            check_indices = np.concatenate([neg_indices_800, pos_indices_200])
            check_df = df.loc[check_indices]
            check_labels = np.array([0]*800 + [1]*200)
            check_dataset = RetinaFineTuneDataset(check_df, LABEL_COL, transform=transform_cur)
            check_loader = DataLoader(check_dataset, batch_size=batch_size_cur, shuffle=False, num_workers=NUM_WORKERS)
            # Extract embeddings
            all_embeds = []
            with torch.no_grad():
                for imgs, _ in check_loader:
                    imgs = imgs.to(DEVICE)
                    feats = backbone(imgs)
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    cls_token = feats[:, 0].cpu().numpy()
                    all_embeds.append(cls_token)
            all_embeds = np.concatenate(all_embeds, axis=0)
            print(f"Length of embeddings: {len(all_embeds)} (shape: {all_embeds.shape})")
            # KNN classification (full set: 800 neg, 200 pos)
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(all_embeds, check_labels)
            pred_probs = knn.predict_proba(all_embeds)[:, 1]
            roc_auc = roc_auc_score(check_labels, pred_probs)
            precision, recall, _ = precision_recall_curve(check_labels, pred_probs)
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
            print(f"KNN ROC AUC (800N/200P): {roc_auc:.3f}")
            print(f"KNN PR AUC (800N/200P): {pr_auc:.3f}")
            # PCA plot: 200 positives vs 200 negatives (balanced)
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            neg_indices_200 = neg_indices[:200]
            pos_indices_200 = pos_indices[:200]
            pca_indices_balanced = np.concatenate([neg_indices_200, pos_indices_200])
            pca_labels_balanced = np.array([0]*200 + [1]*200)
            mask_balanced = np.isin(check_indices, pca_indices_balanced)
            embeds_balanced = all_embeds[mask_balanced]
            knn_bal = KNeighborsClassifier(n_neighbors=5)
            knn_bal.fit(embeds_balanced, pca_labels_balanced)
            pred_probs_bal = knn_bal.predict_proba(embeds_balanced)[:, 1]
            roc_auc_bal = roc_auc_score(pca_labels_balanced, pred_probs_bal)
            precision_bal, recall_bal, _ = precision_recall_curve(pca_labels_balanced, pred_probs_bal)
            pr_auc_bal = auc(recall_bal, precision_bal)
            pca = PCA(n_components=2)
            pca_embeds_bal = pca.fit_transform(embeds_balanced)
            plt.figure(figsize=(6, 5))
            for label, name, color in zip([0, 1], ['negative', 'positive'], ['blue', 'red']):
                plt.scatter(pca_embeds_bal[pca_labels_balanced == label, 0], pca_embeds_bal[pca_labels_balanced == label, 1], label=name, alpha=0.6, s=20, c=color)
            plt.title(f'PCA: 200 negative vs 200 positive ({LABEL_COL}, {IMG_SIZE_CUR}x{IMG_SIZE_CUR})\nKNN ROC AUC={roc_auc_bal:.2f}, PR AUC={pr_auc_bal:.2f}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.tight_layout()
            pca_200n_200p_path = os.path.join(IMAGES_DIR, f'pca_200n_200p_{LABEL_COL.replace(" ", "_")}_{IMG_SIZE_CUR}.png')
            plt.savefig(pca_200n_200p_path)
            print(f"Saved PCA plot (200N/200P) as '{pca_200n_200p_path}'")
            # PCA plot: 800 negatives vs 200 positives (full set)
            pca = PCA(n_components=2)
            pca_embeds_full = pca.fit_transform(all_embeds)
            plt.figure(figsize=(6, 5))
            for label, name, color in zip([0, 1], ['negative', 'positive'], ['blue', 'red']):
                plt.scatter(pca_embeds_full[check_labels == label, 0], pca_embeds_full[check_labels == label, 1], label=name, alpha=0.6, s=20, c=color)
            plt.title(f'PCA: 800 negative vs 200 positive ({LABEL_COL}, {IMG_SIZE_CUR}x{IMG_SIZE_CUR})\nKNN ROC AUC={roc_auc:.2f}, PR AUC={pr_auc:.2f}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.tight_layout()
            pca_800n_200p_path = os.path.join(IMAGES_DIR, f'pca_800n_200p_{LABEL_COL.replace(" ", "_")}_{IMG_SIZE_CUR}.png')
            plt.savefig(pca_800n_200p_path)
            print(f"Saved PCA plot (800N/200P) as '{pca_800n_200p_path}'")
            # Plot ROC and PR curves (full set)
            fpr, tpr, _ = roc_curve(check_labels, pred_probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'KNN ROC Curve (Embeddings) - {LABEL_COL}, {IMG_SIZE_CUR}x{IMG_SIZE_CUR}')
            plt.legend()
            roc_curve_path = os.path.join(IMAGES_DIR, f'knn_roc_curve_{LABEL_COL.replace(" ", "_")}_{IMG_SIZE_CUR}.png')
            plt.savefig(roc_curve_path)
            plt.figure()
            plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'KNN Precision-Recall Curve (Embeddings) - {LABEL_COL}, {IMG_SIZE_CUR}x{IMG_SIZE_CUR}')
            plt.legend()
            pr_curve_path = os.path.join(IMAGES_DIR, f'knn_pr_curve_{LABEL_COL.replace(" ", "_")}_{IMG_SIZE_CUR}.png')
            plt.savefig(pr_curve_path)
            print(f"Saved KNN ROC and PR curves as '{roc_curve_path}' and '{pr_curve_path}'")
        except RuntimeError as e:
            import torch
            if 'out of memory' in str(e):
                print(f'CUDA OOM at size {IMG_SIZE_CUR}, skipping...')
                torch.cuda.empty_cache()
                continue
            else:
                raise
    # After all sizes, plot PR AUC and ROC AUC vs. image size
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(sizes[:len(pr_aucs)], pr_aucs, 'o-r', label='PR AUC')
    plt.plot(sizes[:len(roc_aucs)], roc_aucs, 'o-b', label='ROC AUC')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('AUC')
    plt.title(f'PR AUC and ROC AUC vs. Image Size ({LABEL_COL})')
    plt.legend()
    plt.grid(True)
    summary_plot_path = os.path.join(IMAGES_DIR, f'auc_vs_size_{LABEL_COL.replace(" ", "_")}.png')
    plt.savefig(summary_plot_path)
    print(f"Saved summary plot of AUC vs. image size as '{summary_plot_path}'")
    import sys
    sys.exit(0)

if not CHECK:
    # --- DATASET & DATALOADER ---
    print("Loading data")
    PREVALENT_CSV = os.path.join(BASE_DIR, "retina_prevalent_future_diagnosis.csv")
    df = pd.read_csv(PREVALENT_CSV)
    print(f"Label distribution in CSV ({LABEL_COL}):")
    print(df[LABEL_COL].value_counts())
    # Select indices for positives and true negatives
    pos_indices = df.index[df[LABEL_COL] == 1].tolist()
    neg_indices = df.index[(df[f'prevalent_{DISEASE_TO_TRAIN}'] == 0) & (df[f'future_{DISEASE_TO_TRAIN}'] == 0)].tolist()
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    n_pos = min(N_POS_FUTURE_TRAIN, len(pos_indices))
    n_neg = min(N_NEG, len(neg_indices))
    pos_sample = np.random.choice(pos_indices, n_pos, replace=False) if n_pos > 0 else []
    neg_sample = np.random.choice(neg_indices, n_neg, replace=False) if n_neg > 0 else []
    train_indices = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(train_indices)
    # Validation and test splits (20% for val, rest for test)
    n_val_pos = int(n_pos * 0.2)
    n_val_neg = int(n_neg * 0.2)
    val_pos = pos_indices[n_pos:n_pos + n_val_pos]
    val_neg = neg_indices[n_neg:n_neg + n_val_neg]
    val_indices = np.concatenate([val_pos, val_neg])
    np.random.shuffle(val_indices)
    test_pos = pos_indices[n_pos + n_val_pos:]
    test_neg = neg_indices[n_neg + n_val_neg:]
    test_indices = np.concatenate([test_pos, test_neg])
    np.random.shuffle(test_indices)
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]
    print(f'Train set label distribution ({LABEL_COL}):', train_df[LABEL_COL].value_counts())
    print(f'Val set label distribution ({LABEL_COL}):', val_df[LABEL_COL].value_counts())
    print(f'Test set label distribution ({LABEL_COL}):', test_df[LABEL_COL].value_counts())
    # Create custom datasets
    train_dataset = RetinaFineTuneDataset(train_df, LABEL_COL, transform=transform)
    val_dataset = RetinaFineTuneDataset(val_df, LABEL_COL, transform=transform)
    test_dataset = RetinaFineTuneDataset(test_df, LABEL_COL, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    # Compute class-balanced weights
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    samples_per_cls = [np.sum(np.array(train_labels) == i) for i in range(2)]
    CB_BETA = 0.999  # Typical value for large datasets
    criterion = ClassBalancedLoss(beta=CB_BETA, samples_per_cls=samples_per_cls, device=DEVICE)
    # Update optimizer to only use trainable parameters
    model = RetinaClassifier(backbone, num_classes).to(DEVICE)
    def print_trainable_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    print_trainable_parameters(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None

    # --- TRAINING LOOP ---
    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels, *_ in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
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
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        return total_loss / total, correct / total

    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for imgs, labels, *_ in loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        from sklearn.metrics import classification_report
        print(classification_report(all_labels, all_preds, target_names=["No Future Dx", "Future Dx"], zero_division=0))
        return total_loss / total, correct / total

    def check_tempdir_size(tempdir, max_gb=160):
        total = 0
        for dirpath, dirnames, filenames in os.walk(tempdir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
        total_gb = total / (1024 ** 3)
        if total_gb > max_gb:
            print(f"ERROR: {tempdir} exceeds {max_gb}GB ({total_gb:.2f}GB used). Exiting for safety.")
            sys.exit(1)

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
                with torch.amp.autocast('cuda'):
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
                imgs, labels, future_flags in batch
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
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
                all_val_future_flags.extend(future_flags.cpu().numpy()) # Assuming batch[2] is future_flag
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Print confusion matrix and classification report
        print("Confusion Matrix:")
        print(confusion_matrix(all_val_labels, all_val_preds))
        print(classification_report(all_val_labels, all_val_preds, target_names=["No Future Dx", "Future Dx"]))

        # PR AUC for Prevalent Hypertension (positive class) in validation set
        from sklearn.metrics import precision_recall_curve, auc
        logits_arr = np.array(all_val_logits)
        print('all_val_logits shape:', logits_arr.shape)
        if logits_arr.ndim == 1 or (logits_arr.ndim == 2 and logits_arr.shape[1] == 1):
            # Single logit per sample (binary), use sigmoid
            probs = torch.sigmoid(torch.tensor(logits_arr)).numpy().reshape(-1)
            precision, recall, _ = precision_recall_curve(all_val_labels, probs)
            pr_auc = auc(recall, precision)
            print(f"[VAL] PR AUC for Prevalent Hypertension: {pr_auc:.3f}")
        else:
            # Two logits per sample, use softmax
            probs = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
            precision, recall, _ = precision_recall_curve(all_val_labels, probs[:, 1])
            pr_auc = auc(recall, precision)
            print(f"[VAL] PR AUC for Prevalent Hypertension: {pr_auc:.3f}")

        # PR AUC for Future Dx class in validation set
        future_mask = np.array(all_val_future_flags) == 1
        if np.sum(future_mask) > 0:
            if logits_arr.ndim == 1 or (logits_arr.ndim == 2 and logits_arr.shape[1] == 1):
                future_probs = probs[future_mask]
            else:
                future_probs = probs[future_mask, 1]
            future_labels = np.array(all_val_labels)[future_mask]
            precision, recall, _ = precision_recall_curve(future_labels, future_probs)
            pr_auc = auc(recall, precision)
            print(f"[VAL] PR AUC for Future Dx: {pr_auc:.3f}")
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

    # --- Evaluation loop ---
    all_test_logits = []
    all_test_labels = []
    for batch in test_loader:  # Use test_loader for future hypertension evaluation
        imgs, labels, future_flags = batch
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)  # prevalent_hypertension labels
        with torch.no_grad():
            logits = model(imgs)
            all_test_logits.append(logits.cpu().numpy())
            all_test_labels.append(labels.cpu().numpy())
    test_logits = np.concatenate(all_test_logits, axis=0)  # shape: (num_samples, 2)
    test_labels = np.concatenate(all_test_labels, axis=0)

    # --- After evaluation, compute PR AUC for future hypertension in test set ---
    import torch
    probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    future_mask = test_df[LABEL_COL] == 1
    future_labels = np.array(test_labels)[future_mask]
    future_probs = probs[future_mask, 1]
    precision, recall, _ = precision_recall_curve(future_labels, future_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve (Future Hypertension)')
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, 'pr_auc_future_hypertension.png'))
    print(f"Saved PR AUC curve for future hypertension to {os.path.join(BASE_DIR, 'pr_auc_future_hypertension.png')} (AUC={pr_auc:.3f})")