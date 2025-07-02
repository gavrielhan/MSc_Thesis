# Fine-tune I-JEPA on retina images for disease classification (cardiovascular, hypertension, diabetes)
# This script should be run from the JEPA directory, not inside ijepa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from ijepa.src.datasets.retina import RetinaDataset
from ijepa.src.models.vision_transformer import vit_base  # Use ViT-Base as default backbone

# --- CONFIG ---
MANIFEST = "retina_manifest.csv"
DIAGNOSIS = "retina_patient_diagnosis.csv"
DISEASES = [
    "Coronary atherosclerosis",  # cardiovascular
    "Essential hypertension",    # hypertension
    "Diabetes mellitus, type unspecified"  # diabetes
]
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_PATCH_SIZE = 16
VIT_EMBED_DIM = 768  # for vit_base
PRETRAINED_CKPT = None  # Set path to pretrained checkpoint if available

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*6, [0.5]*6)  # 6 channels (OD+OS)
])

# --- DATASET & DATALOADER ---
dataset = RetinaDataset(MANIFEST, DIAGNOSIS, DISEASES, transform=transform)
# Compute class weights for imbalance
labels = [label for _, label, *_ in dataset]
class_weights = compute_class_weight('balanced', classes=list(range(len(DISEASES))), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

# Split train/val
from sklearn.model_selection import train_test_split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
from torch.utils.data import Subset
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- MODEL ---
# Instantiate ViT backbone for 6-channel input
backbone = vit_base(patch_size=VIT_PATCH_SIZE, img_size=IMG_SIZE, in_chans=6)

# Optionally load pretrained weights (if available)
if PRETRAINED_CKPT is not None and os.path.isfile(PRETRAINED_CKPT):
    state_dict = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
    # If in_chans != 3, you may need to adapt the first conv layer weights
    if backbone.patch_embed.proj.weight.shape[1] != 3:
        # Average or repeat weights for extra channels
        w = state_dict['patch_embed.proj.weight']
        if w.shape[1] == 3 and backbone.patch_embed.proj.weight.shape[1] == 6:
            # Repeat weights for 6 channels
            w6 = w.repeat(1, 2, 1, 1)[:, :6]
            state_dict['patch_embed.proj.weight'] = w6
    backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {PRETRAINED_CKPT}")

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
        # ViT returns (B, N+1, D); [CLS] token is at position 0
        feats = self.backbone(x)  # (B, N+1, D)
        if isinstance(feats, tuple):
            feats = feats[0]
        cls_token = feats[:, 0]  # (B, D)
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
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
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
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=DISEASES))
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
            torch.save(model.state_dict(), "best_retina_classifier.pt")
    print("Training complete. Best val acc:", best_acc) 