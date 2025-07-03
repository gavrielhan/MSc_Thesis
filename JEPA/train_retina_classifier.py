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

# --- CONFIG ---
MANIFEST = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_manifest.csv"
DIAGNOSIS = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_patient_diagnosis.csv"
DISEASES = [
    "Obesity",  # cardiovascular
    "Essential hypertension",    # hypertension
    "Diabetes mellitus, type unspecified"  # diabetes
]
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_PATCH_SIZE = 14
VIT_EMBED_DIM = 1280  # for vit_huge
PRETRAINED_CKPT = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar"
TRIAL = True  # Set to True for a quick test run with 1000 images (at least 100 positives)

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

if TRIAL:
    # Sample 1000 images, at least 100 positives (label==1)
    pos_indices = [i for i, (_, label, *_ ) in enumerate(dataset) if label == 1]
    neg_indices = [i for i, (_, label, *_ ) in enumerate(dataset) if label == 0]
    n_pos = min(100, len(pos_indices))
    n_neg = 1000 - n_pos
    np.random.seed(42)
    pos_sample = np.random.choice(pos_indices, n_pos, replace=False) if n_pos > 0 else []
    neg_sample = np.random.choice(neg_indices, n_neg, replace=False) if n_neg > 0 else []
    trial_indices = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(trial_indices)
    from torch.utils.data import Subset
    dataset = Subset(dataset, trial_indices)
    print(f"TRIAL MODE: Using {len(dataset)} samples ({n_pos} positives, {n_neg} negatives)")

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
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
from torch.utils.data import Subset
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
print("Train label distribution:", np.bincount([labels[i] for i in train_idx]))
print("Val label distribution:", np.bincount([labels[i] for i in val_idx]))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- AMP SCALER ---
scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

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

backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

# Add a trainable classification head using the [CLS] token
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
model = RetinaClassifier(backbone, len(DISEASES)).to(DEVICE)

# --- LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- TRAINING LOOP ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, *_ in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
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
                with torch.cuda.amp.autocast():
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
    print(classification_report(all_labels, all_preds, target_names=["No Future Dx", "Future Dx"]))
    return total_loss / total, correct / total

# --- MAIN ---
if __name__ == "__main__":
    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/best_retina_classifier.pt")
    print("Training complete. Best val acc:", best_acc) 