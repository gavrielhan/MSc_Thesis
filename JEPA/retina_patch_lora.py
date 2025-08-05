#!/usr/bin/env python3
"""
Retina Patch-Based LoRA Fine-Tuning on Messidor Dataset
Uses overlapping patches with distance-based weighting for high-resolution retina images.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import glob
import logging
from typing import List, Tuple, Dict, Any
import math

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Vision Transformer model
from ijepa.src.models.vision_transformer import VisionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Configuration
CONFIG = {
    'patch_size': 616,  # Patch size for training
    'stride': 512,  # Stride for overlapping patches
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'patch_size_vit': 16,
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.2,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'epochs': 30,
    'batch_size': 1,  # Process one patch at a time
    'num_workers': 0,
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/external_datasets',
    'checkpoint_path': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/checkpoint_retina_finetune.pth',
    'output_dir': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/outputs/retina_patch_messidor'
}

# Create output directories
OUTPUT_DIRS = {
    'checkpoints': os.path.join(CONFIG['output_dir'], 'checkpoints'),
    'images': os.path.join(CONFIG['output_dir'], 'images'),
    'results': os.path.join(CONFIG['output_dir'], 'results')
}

for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


class PatchExtractor:
    """Extracts overlapping patches from retina images of any size."""

    def __init__(self, patch_size: int = 616, stride: int = 512):
        self.patch_size = patch_size
        self.stride = stride
        print(f"Patch extractor initialized with {patch_size}x{patch_size} patches and {stride} stride")

    def extract_patches(self, image: Image.Image) -> List[Tuple[Image.Image, float]]:
        """Extract patches from image with distance-based weights."""
        # Get actual image dimensions
        img_width, img_height = image.size

        # Calculate number of patches for this specific image
        num_patches_h = max(1, (img_height - self.patch_size) // self.stride + 1)
        num_patches_w = max(1, (img_width - self.patch_size) // self.stride + 1)

        # Ensure we have at least one patch
        if img_height < self.patch_size or img_width < self.patch_size:
            # If image is smaller than patch size, resize the image
            print(f"Image {img_width}x{img_height} is smaller than patch {self.patch_size}x{self.patch_size}, resizing")
            image = image.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
            img_width, img_height = image.size
            num_patches_h = 1
            num_patches_w = 1

        # Calculate patch centers for distance weighting
        patch_centers = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                center_y = i * self.stride + self.patch_size // 2
                center_x = j * self.stride + self.patch_size // 2
                patch_centers.append((center_y, center_x))

        # Sort patches by distance to image center (closest first)
        image_center = (img_height // 2, img_width // 2)
        patch_distances = []
        for center in patch_centers:
            distance = math.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)
            patch_distances.append(distance)

        # Sort patches by distance
        sorted_indices = np.argsort(patch_distances)
        patch_centers = [patch_centers[i] for i in sorted_indices]
        patch_distances = [patch_distances[i] for i in sorted_indices]

        # Extract patches
        patches = []
        for i, (center_y, center_x) in enumerate(patch_centers):
            # Calculate patch coordinates
            y1 = max(0, center_y - self.patch_size // 2)
            y2 = min(img_height, y1 + self.patch_size)
            x1 = max(0, center_x - self.patch_size // 2)
            x2 = min(img_width, x1 + self.patch_size)

            # Extract patch
            patch = image.crop((x1, y1, x2, y2))

            # If patch is smaller than expected, pad it
            if patch.size != (self.patch_size, self.patch_size):
                padded_patch = Image.new('RGB', (self.patch_size, self.patch_size), 'black')
                padded_patch.paste(patch, (0, 0))
                patch = padded_patch

            # Calculate distance-based weight
            distance = patch_distances[i]
            image_radius = math.sqrt((img_height / 2) ** 2 + (img_width / 2) ** 2)  # Half-diagonal
            weight = 1.0 / (1.0 + (distance / image_radius) ** 2)

            patches.append((patch, weight))

        return patches


class MessidorPatchDataset(Dataset):
    """Dataset for Messidor with patch-based loading."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform, patch_extractor: PatchExtractor):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.patch_extractor = patch_extractor
        self.black_image_count = 0
        self.total_images_checked = 0

        # Expand dataset to include all patches
        self.expanded_data = []
        for idx, row in self.df.iterrows():
            img_name = row['image_id']
            label = int(row['adjudicated_dr_grade'])

            # Get patches for this image
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                patches = self.patch_extractor.extract_patches(img)

                for patch, weight in patches:
                    self.expanded_data.append({
                        'patch': patch,
                        'label': label,
                        'weight': weight,
                        'original_idx': idx,
                        'img_name': img_name
                    })
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create a black patch as fallback
                black_patch = Image.new('RGB', (self.patch_extractor.patch_size, self.patch_extractor.patch_size),
                                        'black')
                self.expanded_data.append({
                    'patch': black_patch,
                    'label': label,
                    'weight': 1.0,
                    'original_idx': idx,
                    'img_name': img_name
                })
                self.black_image_count += 1

        print(f"Expanded dataset: {len(self.df)} images ? {len(self.expanded_data)} patches")
        print(f"Black images: {self.black_image_count}")

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):
        data = self.expanded_data[idx]
        patch = data['patch']
        label = data['label']
        weight = data['weight']

        # Apply transforms
        patch_tensor = self.transform(patch)

        return patch_tensor, label, weight


class ClassificationHead(nn.Module):
    """Classification head for binary classification."""

    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, 0]  # Take CLS token
        return self.head(x)


def build_encoder(config: Dict[str, Any]) -> VisionTransformer:
    """Build the Vision Transformer encoder."""
    enc = VisionTransformer(
        img_size=config['patch_size'],  # Use patch size for ViT
        patch_size=config['patch_size_vit'],
        in_chans=3,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    return enc


def load_pretrained_encoder(enc: VisionTransformer, ckpt_path: str):
    """Load pretrained weights into encoder."""
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        enc_state = state.get('enc', state)

        # Load ALL parameters including LoRA parameters
        filtered = {k: v for k, v in enc_state.items()
                    if (k.startswith('patch_embed.') or k.startswith('blocks.')
                        or k.startswith('lora_') or k in ('cls_token', 'pos_embed'))}

        # Handle input channel mismatch
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 6 and enc.patch_embed.proj.weight.shape[1] == 3:
            filtered['patch_embed.proj.weight'] = w[:, :3]

        # Load state dict and report what was loaded
        missing_keys, unexpected_keys = enc.load_state_dict(filtered, strict=False)
        logger.info("Pretrained weights loaded into encoder")
        logger.info(f"Loaded {len(filtered)} parameters")
        logger.info(f"Missing keys: {len(missing_keys)}")
        logger.info(f"Unexpected keys: {len(unexpected_keys)}")

        # Debug: Check if LoRA parameters were loaded
        lora_params_loaded = [k for k in filtered.keys() if k.startswith('lora_')]
        logger.info(f"LoRA parameters loaded: {len(lora_params_loaded)}")
        if lora_params_loaded:
            logger.info(f"Sample LoRA params: {lora_params_loaded[:5]}")
    else:
        logger.info("No pretrained checkpoint found; skipping load")


def freeze_lora_params(model: nn.Module):
    """Freeze all parameters except LoRA."""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True


def build_transforms():
    """Build image transforms for patches."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((CONFIG['patch_size'], CONFIG['patch_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    all_labels = []
    all_probs = []
    all_weights = []

    with torch.no_grad():
        for patches, labels, weights in dataloader:
            patches = patches.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            logits = model(patches)
            probs = torch.softmax(logits, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_weights.append(weights.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_weights = np.concatenate(all_weights)

    # Calculate weighted metrics
    aucs = []
    pr_aucs = []
    f1s = []

    for c in range(2):  # Binary classification
        y_true = (all_labels == c).astype(int)
        y_score = all_probs[:, c]

        try:
            auc = roc_auc_score(y_true, y_score, sample_weight=all_weights)
            pr_auc = average_precision_score(y_true, y_score, sample_weight=all_weights)
            f1 = f1_score(y_true, np.argmax(all_probs, axis=1) == c, sample_weight=all_weights)
        except ValueError:
            auc = float('nan')
            pr_auc = float('nan')
            f1 = float('nan')

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(f1)

    return {
        'aucs': aucs,
        'pr_aucs': pr_aucs,
        'f1s': f1s,
        'avg_auc': np.mean(aucs),
        'avg_pr_auc': np.mean(pr_aucs),
        'avg_f1': np.mean(f1s)
    }


def main():
    """Main training function."""
    print("? Retina Patch-Based LoRA Fine-Tuning on Messidor")
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # Load Messidor dataset
    root = os.path.join(CONFIG['external_root'], 'messidor')
    img_dir = os.path.join(root, 'IMAGES')
    data_csv = os.path.join(root, 'messidor_data.csv')

    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} Messidor samples")

    # Create train/test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(0.8 * len(df))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Create patch extractor
    patch_extractor = PatchExtractor(
        patch_size=CONFIG['patch_size'],
        stride=CONFIG['stride']
    )

    # Create transforms
    transform = build_transforms()

    # Create datasets
    train_dataset = MessidorPatchDataset(train_df, img_dir, transform, patch_extractor)
    test_dataset = MessidorPatchDataset(test_df, img_dir, transform, patch_extractor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=CONFIG['num_workers'])

    # Build model
    encoder = build_encoder(CONFIG)
    classifier = ClassificationHead(CONFIG['embed_dim'], n_classes=2)
    model = nn.Sequential(encoder, classifier).to(DEVICE)

    # Load pretrained weights
    load_pretrained_encoder(encoder, CONFIG['checkpoint_path'])

    # Freeze non-LoRA parameters
    freeze_lora_params(encoder)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA parameters: {lora_params:,}")

    # Setup optimizer and loss
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'lora_' in n], 'lr': CONFIG['lr']},
        {'params': [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad],
         'lr': CONFIG['lr'] * 3}
    ], weight_decay=CONFIG['weight_decay'])

    criterion = nn.CrossEntropyLoss(reduction='none')  # No reduction for weighted loss

    # Training loop
    best_auc = 0.0
    train_losses = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (patches, labels, weights) in enumerate(train_loader):
            patches = patches.to(DEVICE)
            labels = labels.to(DEVICE)
            weights = weights.to(DEVICE)

            optimizer.zero_grad()

            logits = model(patches)
            losses = criterion(logits, labels)

            # Apply patch weights
            weighted_loss = (losses * weights).mean()

            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{CONFIG['epochs']}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {weighted_loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, DEVICE)

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Test AUC: {test_metrics['avg_auc']:.4f}")
        print(f"  Test PR AUC: {test_metrics['avg_pr_auc']:.4f}")
        print(f"  Test F1: {test_metrics['avg_f1']:.4f}")

        # Save best model
        if test_metrics['avg_auc'] > best_auc:
            best_auc = test_metrics['avg_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': CONFIG
            }, os.path.join(OUTPUT_DIRS['checkpoints'], 'best_model.pth'))
            print(f"  New best model saved! AUC: {best_auc:.4f}")

    print(f"\nTraining completed!")
    print(f"Best test AUC: {best_auc:.4f}")

    # Save final results
    results = {
        'config': CONFIG,
        'best_auc': best_auc,
        'train_losses': train_losses,
        'final_test_metrics': test_metrics
    }

    with open(os.path.join(OUTPUT_DIRS['results'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(OUTPUT_DIRS['images'], 'training_loss.png'))
    plt.close()

    print(f"Results saved to {CONFIG['output_dir']}")


if __name__ == '__main__':
    main()