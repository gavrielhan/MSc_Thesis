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
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from ijepa.src.datasets.retina import RetinaDataset
from ijepa.src.models.vision_transformer import vit_huge  # Use ViT-H/14 as backbone
print("debug2")
import numpy as np
import pandas as pd
import collections
import shutil

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

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 3 channels per eye
])

# --- DATASET & DATALOADER ---
print("Loading data")
FUTURE_CSV = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_future_diagnosis.csv"
dataset = RetinaDataset(FUTURE_CSV, transform=transform)

# Print label distribution in the CSV
df = pd.read_csv(FUTURE_CSV)
print("Label distribution in CSV:")
print(df['label'].value_counts())

# Print disease distribution for positive samples (label==1)
if 'disease' in df.columns:
    pos_diseases = df[df['label'] == 1]['disease']
    disease_counts = collections.Counter(pos_diseases)
    print("Disease distribution among positive samples (label==1):")
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")
else:
    print("No 'disease' column found in CSV; cannot print disease distribution for positives.")

# Compute class weights for imbalance (per disease)
labels = [label for _, label, *_ in dataset]
print("Unique labels in dataset after trial/sample:", set(labels))
print("Number of samples after trial/sample:", len(labels))

if TRIAL:
    # Sample exactly 150 positives (hypertension) and 550 negatives for a total of 700 samples
    if 'disease' in df.columns:
        pos_indices = [i for i, row in df.iterrows() if row['label'] == 1 and row['disease'] == 'Essential hypertension']
        neg_indices = [i for i, row in df.iterrows() if row['label'] == 0]
    else:
        pos_indices = [i for i, (_, label, *_ ) in enumerate(dataset) if label == 1]
        neg_indices = [i for i, (_, label, *_ ) in enumerate(dataset) if label == 0]
    print(f"Available positives (hypertension): {len(pos_indices)}")
    print(f"Available negatives: {len(neg_indices)}")
    n_pos = min(150, len(pos_indices))
    n_neg = min(700 - n_pos, len(neg_indices))
    np.random.seed(42)
    pos_sample = np.random.choice(pos_indices, n_pos, replace=False) if n_pos > 0 else []
    neg_sample = np.random.choice(neg_indices, n_neg, replace=False) if n_neg > 0 else []
    trial_indices = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(trial_indices)
    print(f"Sampled positives: {len(pos_sample)}")
    print(f"Sampled negatives: {len(neg_sample)}")
    print(f"Total samples: {len(trial_indices)}")
    if len(trial_indices) < 700:
        print(f"WARNING: Only {len(trial_indices)} samples available (requested 700). Using all available samples.")
    from torch.utils.data import Subset
    dataset = Subset(dataset, trial_indices)
    # For stratified split, get labels for the sampled indices
    if 'disease' in df.columns:
        trial_labels = [df.iloc[i]['label'] for i in trial_indices]
    else:
        trial_labels = [dataset[i][1] for i in range(len(dataset))]
    print(f"TRIAL MODE: Using {len(trial_indices)} samples ({len(pos_sample)} hypertension positives, {len(neg_sample)} negatives)")
else:
    trial_labels = labels

# Compute class weights for imbalance (per disease)
labels = [label for _, label, *_ in dataset]
print("Unique labels in dataset after trial/sample:", set(labels))
print("Number of samples after trial/sample:", len(labels))
import numpy as np
class_weights = compute_class_weight('balanced', classes=np.arange(2), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

# Split train/val with stratification to maintain class proportions
from sklearn.model_selection import train_test_split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=trial_labels, random_state=42)
from torch.utils.data import Subset
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
print("Train label distribution:", np.bincount([trial_labels[i] for i in train_idx]))
print("Val label distribution:", np.bincount([trial_labels[i] for i in val_idx]))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

# Calculate class weights based on inverse frequency
if TRIAL:
    # For trial mode, use the balanced dataset
    n_neg_trial = 1000  # from trial sampling
    n_pos_trial = 370  # from trial sampling
    pos_weight = n_neg_trial / n_pos_trial  # 550/150 = 3.67
else:
    # For full dataset, use actual frequencies
    pos_weight = len(neg_indices) / len(pos_indices) if len(pos_indices) > 0 else 1.0

class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=DEVICE)
print(f"Using class weights: [1.0, {pos_weight:.2f}]")

# Default: use FocalLoss with class weights
criterion = FocalLoss(alpha=0.90, gamma=2, weight=class_weights)
# To use label smoothing instead, comment the above and uncomment below:
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