#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import argparse

# IJepa imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer

# ---------- Output Directories ----------
OUTPUT_DIRS = {
    'checkpoints': "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/",
    'results': "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/",
    'images': os.path.join('outputs', 'images'),
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ---------- Config ----------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CONFIG = {
    'img_size': (448, 448),  # Changed to match new patch size
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'use_lora': True,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.2,
    'batch_size': 1,
    'num_workers': 0,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'epochs': 20,
    'external_root': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets',
    'checkpoint_path': '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth',
    'patch_size_extract': 448,  # Increased from 224 to reduce number of patches
    'stride': 400,  # Increased from 200 to reduce overlap
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Memory cleanup function
def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PatchExtractor:
    """Extract patches from retinal images with weights based on patch location."""

    def __init__(self, patch_size: int = 224, stride: int = 200):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image: Image.Image) -> List[Tuple[Image.Image, float]]:
        """Extract patches from image with weights based on distance from center."""
        patches = []
        img_width, img_height = image.size

        # Calculate center of image
        center_x, center_y = img_width // 2, img_height // 2

        for y in range(0, img_height - self.patch_size + 1, self.stride):
            for x in range(0, img_width - self.patch_size + 1, self.stride):
                # Extract patch
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

                # Calculate distance from center
                patch_center_x = x + self.patch_size // 2
                patch_center_y = y + self.patch_size // 2
                distance = np.sqrt((patch_center_x - center_x) ** 2 + (patch_center_y - center_y) ** 2)

                # Weight based on distance (closer to center = higher weight)
                max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                weight = 1.0 - (distance / max_distance) * 0.5  # Weight between 0.5 and 1.0

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
        total_images = len(self.df)
        print(f"Initializing Messidor dataset: processing {total_images} images...")

        for idx, row in self.df.iterrows():
            img_name = row['image_id']

            # Show progress every 10 images
            if idx % 10 == 0:
                print(f"Processing image {idx + 1}/{total_images} ({((idx + 1) / total_images) * 100:.1f}%)")

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

        print(f"Dataset initialization completed!")
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
        patch = self.transform(patch)

        return patch, label, weight


class IDRIDPatchDataset(Dataset):
    """Dataset for IDRID with patch-based loading."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform, patch_extractor: PatchExtractor):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.patch_extractor = patch_extractor
        self.black_image_count = 0
        self.total_images_checked = 0

        # Expand dataset to include all patches
        self.expanded_data = []
        total_images = len(self.df)
        print(f"Initializing IDRID dataset: processing {total_images} images...")

        for idx, row in self.df.iterrows():
            img_name = row['Image name']

            # Show progress every 10 images
            if idx % 10 == 0:
                print(f"Processing image {idx + 1}/{total_images} ({((idx + 1) / total_images) * 100:.1f}%)")

            # Debug: Print first few image names to verify
            if idx < 5:
                print(f"Processing CSV row {idx}: image_id = '{img_name}'")

            # Handle NaN values in labels
            if pd.isna(row['Retinopathy grade']):
                print(f"Skipping row {idx} with NaN label for image {img_name}")
                continue

            label = int(row['Retinopathy grade'])

            # Binarize labels: 0 = no DR, 1+ = DR
            label = 1 if label > 0 else 0

            # Get patches for this image
            # Add .jpg extension if not present
            if not img_name.endswith('.jpg'):
                img_name = img_name + '.jpg'

            img_path = os.path.join(self.img_dir, img_name)

            # Debug: Print first few paths to verify
            if idx < 5:
                print(f"  Looking for: {img_path}")

            # Check if image file exists
            if not os.path.exists(img_path):
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

        print(f"Dataset initialization completed!")
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
        patch = self.transform(patch)

        return patch, label, weight


class ClassificationHead(nn.Module):
    """Classification head for the model."""

    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.mean(dim=1)  # Global average pooling
        return self.head(x)


def build_encoder(config: Dict[str, Any]) -> VisionTransformer:
    """Build the vision transformer encoder."""
    encoder = VisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )
    return encoder


def load_pretrained_encoder(enc: VisionTransformer, ckpt_path: str):
    """Load pretrained weights into the encoder."""
    print(f"Loading pretrained weights from: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Filter encoder weights (exclude classification head)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('head.'):  # Exclude classification head
                encoder_state_dict[key] = value

        # Load weights
        missing_keys, unexpected_keys = enc.load_state_dict(encoder_state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Pretrained weights loaded successfully")

    except Exception as e:
        print(f"Error loading pretrained weights: {e}")


def freeze_lora_params(encoder: nn.Module):
    """Freeze non-LoRA parameters in the encoder."""
    print("Freezing non-LoRA parameters...")

    for name, param in encoder.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"LoRA parameter {name} is trainable")


def build_transforms():
    """Build image transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for patches, labels, weights in dataloader:
            patches = patches.to(device)
            labels = labels.to(device)

            logits = model(patches)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics for each class
    aucs = []
    pr_aucs = []
    f1s = []

    for c in range(2):  # Binary classification
        y_true = (all_labels == c).astype(int)
        y_score = all_probs[:, c]

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')

        try:
            pr_auc = average_precision_score(y_true, y_score)
        except ValueError:
            pr_auc = float('nan')

        try:
            y_pred = (y_score > 0.5).astype(int)
            f1 = f1_score(y_true, y_pred)
        except ValueError:
            f1 = float('nan')

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(f1)

    # Calculate average metrics
    avg_auc = np.nanmean(aucs)
    avg_pr_auc = np.nanmean(pr_aucs)
    avg_f1 = np.nanmean(f1s)

    return {
        'aucs': aucs,
        'pr_aucs': pr_aucs,
        'f1s': f1s,
        'avg_auc': avg_auc,
        'avg_pr_auc': avg_pr_auc,
        'avg_f1': avg_f1
    }


def load_messidor_data():
    """Load Messidor dataset with train/test split."""
    print("Loading Messidor dataset...")

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

    # Binarize adjudicated_dr_grade
    df['adjudicated_dr_grade'] = (df['adjudicated_dr_grade'] != 0).astype(int)

    # Create train/test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(0.8 * len(df))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    print(f"Train label distribution: {train_df['adjudicated_dr_grade'].value_counts().sort_index().to_dict()}")
    print(f"Test label distribution: {test_df['adjudicated_dr_grade'].value_counts().sort_index().to_dict()}")

    return train_df, test_df, img_dir


def load_idrid_data():
    """Load IDRID dataset with predefined train/test split."""
    print("Loading IDRID dataset...")

    root = os.path.join(CONFIG['external_root'], 'IDRID', 'B. Disease Grading')
    img_dir = os.path.join(root, '1. Original Images', 'a. Training Set')
    train_csv = os.path.join(root, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
    test_csv = os.path.join(root, '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')

    # Load and clean training data
    df_train = pd.read_csv(train_csv)
    df_train.columns = df_train.columns.str.strip()
    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    df_train = df_train.dropna(axis=1, how='all')

    # Load and clean test data
    df_test = pd.read_csv(test_csv)
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    df_test = df_test.dropna(axis=1, how='all')

    # Binarize labels
    for df_split in [df_train, df_test]:
        for col in ['Retinopathy grade', 'Risk of macular edema']:
            if col in df_split.columns:
                df_split[col] = (df_split[col] > 0).astype(int)

    # Debug: Check label ranges
    label_col = 'Retinopathy grade'
    print(f"Training set label range: {df_train[label_col].min()} to {df_train[label_col].max()}")
    print(f"Test set label range: {df_test[label_col].min()} to {df_test[label_col].max()}")
    print(f"Training set label distribution: {df_train[label_col].value_counts().sort_index().to_dict()}")
    print(f"Test set label distribution: {df_test[label_col].value_counts().sort_index().to_dict()}")

    # Ensure labels are in correct range (0 to n_classes-1)
    for df_split, split_name in [(df_train, 'train'), (df_test, 'test')]:
        if df_split[label_col].min() < 0 or df_split[label_col].max() >= 2:
            print(f"Warning: {split_name} set has labels outside [0,1] range. Fixing...")
            df_split[label_col] = df_split[label_col].clip(0, 1).astype(int)
            print(
                f"After fixing: {split_name} set label range: {df_split[label_col].min()} to {df_split[label_col].max()}")

    print("Using IDRID's predefined train/test splits")

    return df_train, df_test, img_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Retina Patch-Based LoRA Fine-Tuning')
    parser.add_argument('--dataset', type=str, choices=['messidor', 'idrid'], required=True,
                        help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    args = parser.parse_args()

    # Update config with command line arguments
    CONFIG['epochs'] = args.epochs
    CONFIG['lr'] = args.lr
    CONFIG['batch_size'] = args.batch_size

    print(f"? Retina Patch-Based LoRA Fine-Tuning on {args.dataset.upper()}")
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # Load dataset based on selection
    if args.dataset == 'messidor':
        train_df, test_df, img_dir = load_messidor_data()
        DatasetClass = MessidorPatchDataset
    elif args.dataset == 'idrid':
        train_df, test_df, img_dir = load_idrid_data()
        DatasetClass = IDRIDPatchDataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create patch extractor
    patch_extractor = PatchExtractor(
        patch_size=CONFIG['patch_size_extract'],
        stride=CONFIG['stride']
    )

    # Create transforms
    transform = build_transforms()

    # Create datasets
    train_dataset = DatasetClass(train_df, img_dir, transform, patch_extractor)
    test_dataset = DatasetClass(test_df, img_dir, transform, patch_extractor)

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

    # Ensure classification head is trainable
    for name, param in classifier.named_parameters():
        param.requires_grad = True
        print(f"Classification head parameter {name} is trainable")

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

    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    print("=" * 80)

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Training phase
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

        avg_train_loss = epoch_loss / num_batches

        # Evaluate on train set (for monitoring)
        train_metrics = evaluate_model(model, train_loader, DEVICE)

        # Evaluate on test set (proper evaluation)
        test_metrics = evaluate_model(model, test_loader, DEVICE)

        # Log epoch performance
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_auc': train_metrics['avg_auc'],
            'train_pr_auc': train_metrics['avg_pr_auc'],
            'train_f1': train_metrics['avg_f1'],
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

        # Print comprehensive epoch results
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train AUC: {train_metrics['avg_auc']:.4f} | Test AUC: {test_metrics['avg_auc']:.4f}")
        print(f"  Train PR AUC: {train_metrics['avg_pr_auc']:.4f} | Test PR AUC: {test_metrics['avg_pr_auc']:.4f}")
        print(f"  Train F1: {train_metrics['avg_f1']:.4f} | Test F1: {test_metrics['avg_f1']:.4f}")
        print(f"  Test AUC (Class 0): {test_metrics['aucs'][0]:.4f}")
        print(f"  Test AUC (Class 1): {test_metrics['aucs'][1]:.4f}")
        print(f"  Test PR AUC (Class 0): {test_metrics['pr_aucs'][0]:.4f}")
        print(f"  Test PR AUC (Class 1): {test_metrics['pr_aucs'][1]:.4f}")
        print(f"  Test F1 (Class 0): {test_metrics['f1s'][0]:.4f}")
        print(f"  Test F1 (Class 1): {test_metrics['f1s'][1]:.4f}")

        # Save best model based on test AUC
        if test_metrics['avg_auc'] > best_auc:
            best_auc = test_metrics['avg_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': CONFIG
            }, os.path.join(OUTPUT_DIRS['checkpoints'], f'best_model_{args.dataset}.pth'))
            print(f"  ? New best model! Test AUC: {best_auc:.4f}")
        print("-" * 80)

    print(f"\nTraining completed! Best test AUC: {best_auc:.4f}")

    # Save performance log
    performance_log = {
        'dataset': args.dataset,
        'config': CONFIG,
        'best_auc': best_auc,
        'epochs': epoch_logs
    }

    with open(os.path.join(OUTPUT_DIRS['results'], f'performance_log_{args.dataset}.json'), 'w') as f:
        json.dump(performance_log, f, indent=2)

    print(f"Performance log saved to {os.path.join(OUTPUT_DIRS['results'], f'performance_log_{args.dataset}.json')}")


if __name__ == '__main__':
    main()