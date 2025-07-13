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
print("debug2")
import numpy as np
import pandas as pd
import collections
import shutil
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image

# Custom Dataset for fine-tuning
class RetinaFineTuneDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
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
        label = int(row['prevalent_hypertension'])
        future_flag = int(row['future_hypertension'])
        return img, label, future_flag

# Set custom temp directory to avoid filling up system TMPDIR
os.environ['TMPDIR'] = '/home/gavrielh/temp'
os.environ['TEMP'] = '/home/gavrielh/temp'
os.environ['TMP'] = '/home/gavrielh/temp'

# --- CONFIG ---
MANIFEST = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_manifest.csv"
DIAGNOSIS = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_patient_diagnosis.csv"
DISEASES = [
    "Obesity",  # cardiovascular
    "Essential hypertension",    # hypertension
    "Diabetes mellitus, type unspecified"  # diabetes
]
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_PATCH_SIZE = 14
VIT_EMBED_DIM = 1280  # for vit_huge
PRETRAINED_CKPT = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar"
TRIAL = True  # Set to True for a quick test run with 1000 images (at least 100 positives)

# Set the desired number of samples for each group in one place
N_POS_PREVALENT = 161
N_POS_FUTURE_TRAIN = 200
N_POS_FUTURE_TEST = 170
N_NEG = 800 - N_POS_PREVALENT - N_POS_FUTURE_TRAIN
TRIAL_PERCENT = 1.0  # Use 20% of each group in TRIAL mode

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 3 channels per eye
])

# --- DATASET & DATALOADER ---
print("Loading data")
PREVALENT_CSV = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_prevalent_future_diagnosis.csv"
df = pd.read_csv(PREVALENT_CSV)
print("Label distribution in CSV (prevalent_hypertension):")
print(df['prevalent_hypertension'].value_counts())
print("Label distribution in CSV (future_hypertension):")
print(df['future_hypertension'].value_counts())

# For training: positives = prevalent_hypertension==1, negatives = prevalent_hypertension==0
if TRIAL:
    # Reduce numbers by TRIAL_PERCENT
    n_pos_prevalent = max(1, int(N_POS_PREVALENT * TRIAL_PERCENT))
    n_pos_future_train = max(1, int(N_POS_FUTURE_TRAIN * TRIAL_PERCENT))
    n_pos_future_test = max(1, int(N_POS_FUTURE_TEST * TRIAL_PERCENT))
    n_neg = max(1, int(N_NEG * TRIAL_PERCENT))
else:
    n_pos_prevalent = N_POS_PREVALENT
    n_pos_future_train = N_POS_FUTURE_TRAIN
    n_pos_future_test = N_POS_FUTURE_TEST
    n_neg = N_NEG

future_indices = df.index[df['future_hypertension'] == 1].tolist()
np.random.seed(42)
np.random.shuffle(future_indices)
future_train_indices = future_indices[:n_pos_future_train]
future_test_indices = future_indices[n_pos_future_train:n_pos_future_train + n_pos_future_test]
# Remove these from the general pool
remaining_indices = list(set(df.index) - set(future_train_indices) - set(future_test_indices))
# Sample the rest as before
pos_indices = df.index[df['prevalent_hypertension'] == 1].tolist()
neg_indices = df.index[df['prevalent_hypertension'] == 0].tolist()
# Remove any overlap with future_train/test
pos_indices = list(set(pos_indices) - set(future_train_indices) - set(future_test_indices))
neg_indices = list(set(neg_indices) - set(future_train_indices) - set(future_test_indices))
# Sample
n_pos_prevalent = min(n_pos_prevalent, len(pos_indices))
n_neg = min(n_neg, len(neg_indices))
pos_sample = np.random.choice(pos_indices, n_pos_prevalent, replace=False) if n_pos_prevalent > 0 else []
neg_sample = np.random.choice(neg_indices, n_neg, replace=False) if n_neg > 0 else []
train_indices = np.concatenate([future_train_indices, pos_sample, neg_sample])
np.random.shuffle(train_indices)
test_indices = np.array(future_test_indices)
print(f"TRAIN: {len(train_indices)} (future hypertension: {len(future_train_indices)}, prevalent: {len(pos_sample)}, negatives: {len(neg_sample)})")
print(f"TEST: {len(test_indices)} (future hypertension)")
# Prepare datasets
train_df = df.loc[train_indices]
test_df = df.loc[test_indices]

# Create custom datasets
train_dataset = RetinaFineTuneDataset(train_df, transform=transform)
test_dataset = RetinaFineTuneDataset(test_df, transform=transform)

# Split train/val with stratification to maintain class proportions
from sklearn.model_selection import train_test_split
train_labels = train_df['prevalent_hypertension'].tolist()
train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, stratify=train_labels, random_state=42)
from torch.utils.data import Subset
train_set = Subset(train_dataset, train_idx)
val_set = Subset(train_dataset, val_idx)
print("Train label distribution:", np.bincount([train_labels[i] for i in train_idx]))
print("Val label distribution:", np.bincount([train_labels[i] for i in val_idx]))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Compute class weights for imbalance
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
class_weights = compute_class_weight('balanced', classes=np.arange(2), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

# --- AMP SCALER ---
scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

# --- MODEL ---
# Instantiate ViT-H/14 backbone for 6-channel input
backbone = vit_huge(patch_size=VIT_PATCH_SIZE, img_size=IMG_SIZE, in_chans=6)

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

# Add a trainable classification head using the [CLS] token
if TRIAL:
    num_classes = 2
else:
    num_classes = len(DISEASES)
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
model = RetinaClassifier(backbone, num_classes).to(DEVICE)

# --- LOSS & OPTIMIZER ---
# Use Focal Loss with class weights for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# Default: use FocalLoss with class weights
# criterion = FocalLoss(alpha=0.90, gamma=2, weight=class_weights)
# To use label smoothing instead, comment the above and uncomment below:
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.2, weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

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
                # Debug prints
                print(f"[TRAIN] Batch: Loss={loss.item()} | Logits={logits.detach().cpu().numpy()} | Labels={labels.detach().cpu().numpy()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # Debug prints
            print(f"[TRAIN] Batch: Loss={loss.item()} | Logits={logits.detach().cpu().numpy()} | Labels={labels.detach().cpu().numpy()}")
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
                    # Debug prints
                    print(f"[VAL] Batch: Loss={loss.item()} | Logits={logits.detach().cpu().numpy()} | Labels={labels.detach().cpu().numpy()}")
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                # Debug prints
                print(f"[VAL] Batch: Loss={loss.item()} | Logits={logits.detach().cpu().numpy()} | Labels={labels.detach().cpu().numpy()}")
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # Debug prints for first batch
            if len(all_labels) <= imgs.size(0):
                print("Sample logits (softmax):", logits[:10].softmax(1).cpu().detach().numpy())
                print("Sample predictions:", preds[:10].cpu().detach().numpy())
                print("Sample labels:", labels[:10].cpu().detach().numpy())
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

# --- MAIN ---
if __name__ == "__main__":
    best_acc = 0
    for epoch in range(EPOCHS):
        check_tempdir_size('/home/gavrielh/temp', max_gb=160)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            #torch.save(model.state_dict(), "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/best_retina_classifier.pt")
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
    imgs = batch[0].to(DEVICE)
    labels = batch[1].to(DEVICE)  # prevalent_hypertension labels
    with torch.no_grad():
        logits = model(imgs)
        all_test_logits.append(logits.cpu().numpy())
        all_test_labels.append(labels.cpu().numpy())
test_logits = np.concatenate(all_test_logits, axis=0)  # shape: (num_samples, 2)
test_labels = np.concatenate(all_test_labels, axis=0)

# --- After evaluation, compute PR AUC for future hypertension in test set ---
import torch
probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
future_mask = test_df['future_hypertension'] == 1
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
plt.savefig('pr_auc_future_hypertension.png')
print(f"Saved PR AUC curve for future hypertension to pr_auc_future_hypertension.png (AUC={pr_auc:.3f})")