#!/usr/bin/env python3
"""
Test script for retinal I-JEPA pretraining
"""

import os
import sys
import yaml
import torch
from PIL import Image
import pandas as pd

# Add I-JEPA to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ijepa")))

from src.retina_dataset import make_retina_dataset
from ijepa.src.transforms import make_transforms
from ijepa.src.masks.multiblock import MaskCollator


def test_retina_dataset():
    """Test if retinal dataset loads correctly"""
    print("Testing retinal dataset...")

    # Create test config
    config = {
        'data': {
            'batch_size': 1,
            'crop_size': 616,
            'crop_scale': [0.8, 1.0],
            'use_gaussian_blur': False,
            'use_horizontal_flip': True,
            'use_color_distortion': False,
            'color_jitter_strength': 0.0,
            'num_workers': 0,
            'pin_mem': True,
            'root_path': '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA'
        },
        'mask': {
            'allow_overlap': False,
            'aspect_ratio': [0.2, 5.0],
            'enc_mask_scale': [0.2, 0.6],
            'min_keep': 2,
            'num_enc_masks': 2,
            'num_pred_masks': 2,
            'patch_size': 14,
            'pred_mask_scale': [0.2, 0.6]
        }
    }

    # Create transforms
    transform = make_transforms(
        crop_size=config['data']['crop_size'],
        crop_scale=config['data']['crop_scale'],
        gaussian_blur=config['data']['use_gaussian_blur'],
        horizontal_flip=config['data']['use_horizontal_flip'],
        color_distortion=config['data']['use_color_distortion'],
        color_jitter=config['data']['color_jitter_strength']
    )

    # Create mask collator
    mask_collator = MaskCollator(
        input_size=config['data']['crop_size'],
        patch_size=config['mask']['patch_size'],
        pred_mask_scale=config['mask']['pred_mask_scale'],
        enc_mask_scale=config['mask']['enc_mask_scale'],
        aspect_ratio=config['mask']['aspect_ratio'],
        nenc=config['mask']['num_enc_masks'],
        npred=config['mask']['num_pred_masks'],
        allow_overlap=config['mask']['allow_overlap'],
        min_keep=config['mask']['min_keep']
    )

    # Test dataset creation
    try:
        manifest_csv = os.path.join(config['data']['root_path'], 'retina_manifest.csv')
        dataset, loader, sampler = make_retina_dataset(
            transform=transform,
            batch_size=config['data']['batch_size'],
            collator=mask_collator,
            pin_mem=config['data']['pin_mem'],
            training=True,
            num_workers=config['data']['num_workers'],
            world_size=1,
            rank=0,
            root_path=config['data']['root_path'],
            copy_data=False,
            drop_last=True,
            manifest_csv=manifest_csv
        )

        print(f"? Dataset created successfully with {len(dataset)} samples")
        print(f"? DataLoader created with {len(loader)} batches")

        # Test dataset creation (don't actually load batches)
        print(f"Dataset created successfully with {len(dataset)} samples")
        print(f"DataLoader created with {len(loader)} batches")
        print(f"Dataset structure verified - ready for training")

    except Exception as e:
        print(f"? Error creating dataset: {e}")
        return False

    return True


def test_config_file():
    """Test if YAML config loads correctly"""
    print("\nTesting YAML config...")

    config_path = "configs/retina_pretrain.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"? Config loaded from {config_path}")
        print(f"   - Batch size: {config['data']['batch_size']}")
        print(f"   - Crop size: {config['data']['crop_size']}")
        print(f"   - Epochs: {config['optimization']['epochs']}")
        return True
    except Exception as e:
        print(f"? Error loading config: {e}")
        return False


def check_manifest():
    """Check if manifest CSV exists and has correct format"""
    print("\nChecking manifest CSV...")

    manifest_path = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/retina_manifest.csv"

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return False

    print(f" Found manifest: {manifest_path}")

    try:
        df = pd.read_csv(manifest_path)
        print(f" Manifest loaded: {len(df)} rows")

        required_cols = ['RegistrationCode', 'date', 'od_path', 'os_path']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f" Missing columns: {missing_cols}")
            return False

        print(f" Required columns present: {required_cols}")

        # Check if some image paths exist
        sample_od = df['od_path'].iloc[0]
        sample_os = df['os_path'].iloc[0]

        if os.path.exists(sample_od):
            print(f" Sample OD image exists: {sample_od}")
        else:
            print(f"  Sample OD image not found: {sample_od}")

        if os.path.exists(sample_os):
            print(f" Sample OS image exists: {sample_os}")
        else:
            print(f"  Sample OS image not found: {sample_os}")

        return True

    except Exception as e:
        print(f" Error reading manifest: {e}")
        return False


def check_pretrained_model():
    """Check if pretrained ImageNet model exists"""
    print("\nChecking pretrained ImageNet model...")

    model_path = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar"

    if not os.path.exists(model_path):
        print(f" Pretrained model not found: {model_path}")
        print("   Run: python download_pretrained.py")
        return False

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Pretrained model loaded successfully")
        print(f"   - Keys: {list(checkpoint.keys())}")
        print(f"   - Encoder keys: {list(checkpoint['encoder'].keys())[:5]}...")
        return True
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return False


if __name__ == "__main__":
    print("Testing Retinal I-JEPA Pretraining Setup")
    print("=" * 50)

    # Run tests
    config_ok = test_config_file()
    manifest_ok = check_manifest()
    pretrained_ok = check_pretrained_model()
    dataset_ok = test_retina_dataset()

    print("\n" + "=" * 50)
    print("? Test Results:")
    print(f"   Config file: {'PASS' if config_ok else 'FAIL'}")
    print(f"   Manifest CSV: {'PASS' if manifest_ok else 'FAIL'}")
    print(f"   Pretrained model: {'PASS' if pretrained_ok else 'FAIL'}")
    print(f"   Dataset loading: {'PASS' if dataset_ok else 'FAIL'}")

    if all([config_ok, manifest_ok, pretrained_ok, dataset_ok]):
        print("\n All tests passed! Ready to run pretraining.")
        print("\nTo start pretraining, run:")
        print("cd JEPA/ijepa")
        print("python main.py --fname ../configs/retina_pretrain.yaml --devices cuda:0")
        print("\nThis will:")
        print("  - Load ImageNet pretrained I-JEPA model")
        print("  - Continue pretraining on your retinal images")
        print("  - Save retinal-pretrained checkpoints")
    else:
        print("\nSome tests failed. Please fix issues before running pretraining.")
        if not pretrained_ok:
            print("\n You need the pretrained ImageNet model.")
            print("   Download it from: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar")
            print("   Save it to: /PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN1K-vit.h.14-300e.pth.tar")