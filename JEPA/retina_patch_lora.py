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

# Add the ijepa directory to Python path
ijepa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ijepa')
sys.path.append(ijepa_path)

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
    'patch_size': 224,  # Patch size for training (larger to reduce memory)
    'stride': 200,  # Stride for overlapping patches (less overlap)
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'patch_size_vit': 14,  # ViT patch size (MUST match checkpoint)
    'img_size_vit': 224,  # Input size for ViT (224x224 = 16x16 patches with patch_size_vit=14)
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.2,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'epochs': 20,
    'batch_size': 1,  # Process one patch at a time
    'num_workers': 0,
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets',
    'checkpoint_path': '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth',
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

    def __init__(self, patch_size: int = 224, stride: int = 200):
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

            # Debug: Print first few image names to verify
            if idx < 5:
                print(f"Processing CSV row {idx}: image_id = '{img_name}'")

            # Handle NaN values in labels
            if pd.isna(row['adjudicated_dr_grade']):
                print(f"Skipping row {idx} with NaN label for image {img_name}")
                continue

            label = int(row['adjudicated_dr_grade'])

            # Binarize labels: 0 = no DR, 1+ = DR
            label = 1 if label > 0 else 0

            # Get patches for this image
            img_path = os.path.join(self.img_dir, img_name)

            # Debug: Print first few paths to verify
            if idx < 5:
                print(f"  Looking for: {img_path}")

            # Check if image file exists (case-insensitive)
            if not os.path.exists(img_path):
                # Try with different case extensions
                base_name = os.path.splitext(img_name)[0]
                possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

                found_file = None
                for ext in possible_extensions:
                    alt_path = os.path.join(self.img_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        found_file = alt_path
                        if idx < 5:
                            print(f"  Found with different case: {alt_path}")
                        break

                if found_file:
                    img_path = found_file
                else:
                    print(f"Image file not found: {img_path}")
                    continue

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

        # Check label distribution
        labels = [data['label'] for data in self.expanded_data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

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
        img_size=(config['img_size_vit'], config['img_size_vit']),  # Use fixed ViT input size
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
        # The encoder state dict contains all parameters including LoRA
        filtered = enc_state

        # Handle input channel mismatch
        w = filtered.get('patch_embed.proj.weight')
        if w is not None and w.shape[1] == 6 and enc.patch_embed.proj.weight.shape[1] == 3:
            filtered['patch_embed.proj.weight'] = w[:, :3]

        # Handle positional embedding size mismatch by interpolation
        if 'pos_embed' in filtered:
            old_pos_embed = filtered['pos_embed']
            new_pos_embed = enc.pos_embed

            if old_pos_embed.shape != new_pos_embed.shape:
                print(f"Interpolating pos_embed from {old_pos_embed.shape} to {new_pos_embed.shape}")

                # Remove cls token for interpolation
                old_pos_embed_no_cls = old_pos_embed[:, 1:, :]  # Remove cls token
                new_pos_embed_no_cls = new_pos_embed[:, 1:, :]  # Remove cls token

                # Calculate old and new grid sizes (handle non-square cases)
                old_num_patches = old_pos_embed_no_cls.shape[1]
                new_num_patches = new_pos_embed_no_cls.shape[1]

                # Find the closest square dimensions
                old_size = int(math.sqrt(old_num_patches))
                new_size = int(math.sqrt(new_num_patches))

                # If not perfect squares, use the next larger square and pad
                if old_size * old_size != old_num_patches:
                    old_size = int(math.ceil(math.sqrt(old_num_patches)))
                if new_size * new_size != new_num_patches:
                    new_size = int(math.ceil(math.sqrt(new_num_patches)))

                # Pad old embeddings to square
                old_pad_size = old_size * old_size - old_num_patches
                if old_pad_size > 0:
                    old_pad = torch.zeros(1, old_pad_size, old_pos_embed_no_cls.shape[2],
                                          device=old_pos_embed_no_cls.device)
                    old_pos_embed_no_cls = torch.cat([old_pos_embed_no_cls, old_pad], dim=1)

                # Pad new embeddings to square
                new_pad_size = new_size * new_size - new_num_patches
                if new_pad_size > 0:
                    new_pad = torch.zeros(1, new_pad_size, new_pos_embed_no_cls.shape[2],
                                          device=new_pos_embed_no_cls.device)
                    new_pos_embed_no_cls = torch.cat([new_pos_embed_no_cls, new_pad], dim=1)

                # Reshape to 2D grid
                old_pos_embed_2d = old_pos_embed_no_cls.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                new_pos_embed_2d = new_pos_embed_no_cls.reshape(1, new_size, new_size, -1).permute(0, 3, 1, 2)

                # Interpolate
                interpolated = torch.nn.functional.interpolate(
                    old_pos_embed_2d,
                    size=(new_size, new_size),
                    mode='bilinear',
                    align_corners=False
                )

                # Reshape back
                interpolated = interpolated.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

                # Remove padding from interpolated result
                if new_pad_size > 0:
                    interpolated = interpolated[:, :new_num_patches, :]

                # Add cls token back
                cls_token = new_pos_embed[:, 0:1, :]
                filtered['pos_embed'] = torch.cat([cls_token, interpolated], dim=1)

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
            logger.warning("No LoRA parameters found in checkpoint!")
            logger.info(f"Available parameter prefixes: {list(set([k.split('.')[0] for k in filtered.keys()]))}")
            logger.info(f"First 10 loaded parameters: {list(filtered.keys())[:10]}")
            logger.info(f"Total parameters in checkpoint: {len(filtered)}")
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
        transforms.Resize((CONFIG['img_size_vit'], CONFIG['img_size_vit'])),  # Resize to ViT input size
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

    # Debug: Check CSV structure
    print(f"CSV columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())

    # Check for NaN values
    nan_counts = df.isna().sum()
    print(f"NaN counts per column:")
    print(nan_counts)

    # Check image directory
    print(f"Image directory: {img_dir}")
    if os.path.exists(img_dir):
        image_files = os.listdir(img_dir)
        print(f"Found {len(image_files)} files in image directory")
        print(f"First 5 image files: {image_files[:5]}")

        # Check if CSV image names match actual files
        csv_image_names = df['image_id'].tolist()
        print(f"First 5 CSV image names: {csv_image_names[:5]}")

        # Check how many CSV images actually exist
        existing_count = 0
        for img_name in csv_image_names:
            if os.path.exists(os.path.join(img_dir, img_name)):
                existing_count += 1

        print(f"CSV images that exist: {existing_count}/{len(csv_image_names)}")

    else:
        print(f"Image directory does not exist: {img_dir}")

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
    epoch_logs = []

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

        avg_loss = epoch_loss / num_batches

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, DEVICE)

        # Log epoch performance
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'test_auc': test_metrics['avg_auc'],
            'test_pr_auc': test_metrics['avg_pr_auc'],
            'test_f1': test_metrics['avg_f1'],
            'test_auc_class_0': test_metrics['aucs'][0],
            'test_auc_class_1': test_metrics['aucs'][1],
            'test_pr_auc_class_0': test_metrics['pr_aucs'][0],
            'test_pr_auc_class_1': test_metrics['pr_aucs'][1],
            'test_f1_class_0': test_metrics['f1s'][0],
            'test_f1_class_1': test_metrics['f1s'][1]
        }
        epoch_logs.append(epoch_log)

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Test AUC: {test_metrics['avg_auc']:.4f}")
        print(f"  Test PR AUC: {test_metrics['avg_pr_auc']:.4f}")
        print(f"  Test F1: {test_metrics['avg_f1']:.4f}")
        print(f"  Test AUC (Class 0): {test_metrics['aucs'][0]:.4f}")
        print(f"  Test AUC (Class 1): {test_metrics['aucs'][1]:.4f}")
        print(f"  Test PR AUC (Class 0): {test_metrics['pr_aucs'][0]:.4f}")
        print(f"  Test PR AUC (Class 1): {test_metrics['pr_aucs'][1]:.4f}")
        print(f"  Test F1 (Class 0): {test_metrics['f1s'][0]:.4f}")
        print(f"  Test F1 (Class 1): {test_metrics['f1s'][1]:.4f}")

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
            print(f"  ? New best model! AUC: {best_auc:.4f}")
        print()

    print(f"\nTraining completed! Best test AUC: {best_auc:.4f}")

    # Save performance log
    performance_log = {
        'config': CONFIG,
        'best_auc': best_auc,
        'epochs': epoch_logs
    }

    with open(os.path.join(OUTPUT_DIRS['results'], 'performance_log.json'), 'w') as f:
        json.dump(performance_log, f, indent=2)

    print(f"Performance log saved to {os.path.join(OUTPUT_DIRS['results'], 'performance_log.json')}")


if __name__ == '__main__':
    main()