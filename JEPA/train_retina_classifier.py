# Fine-tune I-JEPA on retina images for disease classification (cardiovascular, hypertension, diabetes)
# This script should be run from the JEPA directory, not inside ijepa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from JEPA.ijepa.src.datasets.retina import RetinaDataset

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
from JEPA.ijepa.src.models.ijepa import ijeparch
# Load pretrained I-JEPA backbone (ViT-H/14 or similar)
backbone = ijeparch(pretrained=True)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
# Add a trainable classification head
class RetinaClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
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