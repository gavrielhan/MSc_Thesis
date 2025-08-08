# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def make_retina_dataset(
        transform,
        batch_size,
        collator=None,
        pin_mem=True,
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=None,
        training=True,
        copy_data=False,
        drop_last=True,
        manifest_csv=None,
        image_folder=None  # Added to match I-JEPA interface
):
    """
    Create retinal dataset following I-JEPA interface

    Args:
        manifest_csv: Path to CSV with columns ['od_path', 'os_path']
    """
    dataset = RetinaDataset(
        root=root_path,
        manifest_csv=manifest_csv,
        transform=transform,
        train=training
    )
    logger.info(f'Retina dataset created with {len(dataset)} samples')

    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    # Use the original collator (mask_collator) which handles both images and masks
    # For batch_size=1, we need to ensure proper batching
    if batch_size == 1:
        # Custom collate that ensures proper batch format for mask collator
        def single_batch_collate(batch):
            # batch should be a list with one item
            if len(batch) == 1:
                # Return the single item as a batch
                return [batch[0]]
            else:
                return batch

        data_loader = DataLoader(
            dataset,
            collate_fn=lambda x: collator(single_batch_collate(x)),
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=False
        )
    else:
        data_loader = DataLoader(
            dataset,
            collate_fn=collator,  # This is the mask_collator that creates masks
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=False
        )
    logger.info('Retina unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class RetinaDataset(Dataset):
    """
    Retinal dataset that loads OD+OS image pairs and concatenates them
    """

    def __init__(
            self,
            root,
            manifest_csv,
            transform=None,
            train=True,
            test_split=0.2
    ):
        """
        Args:
            root: Root directory containing images
            manifest_csv: CSV with columns ['od_path', 'os_path']
            transform: Image transforms
            train: Whether this is training set
            test_split: Fraction of data to use for test
        """
        self.root = root
        self.transform = transform
        self.train = train

        # Load manifest - look for it in the root directory
        if manifest_csv:
            self.manifest_path = manifest_csv
        else:
            self.manifest_path = os.path.join(root, 'retina_manifest.csv')

        try:
            self.df = pd.read_csv(self.manifest_path)
            logger.info(f"Loaded {len(self.df)} image pairs from {self.manifest_path}")
        except FileNotFoundError:
            logger.error(f"Manifest CSV not found: {self.manifest_path}")
            raise

        # Split into train/test
        if train:
            self.df = self.df.iloc[:int(len(self.df) * (1 - test_split))]
        else:
            self.df = self.df.iloc[int(len(self.df) * (1 - test_split)):]

        logger.info(f"Using {'train' if train else 'test'} split: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        od_path = row['od_path']
        os_path = row['os_path']
        registration_code = row['RegistrationCode']
        date = row['date']

        # Load images
        try:
            od_img = Image.open(od_path).convert('RGB')
        except:
            logger.warning(f"Could not load OD image: {od_path}")
            od_img = Image.new('RGB', (224, 224), 'black')

        try:
            os_img = Image.open(os_path).convert('RGB')
        except:
            logger.warning(f"Could not load OS image: {os_path}")
            os_img = Image.new('RGB', (224, 224), 'black')

        # Apply transforms
        if self.transform:
            od_tensor = self.transform(od_img)
            os_tensor = self.transform(os_img)
        else:
            od_tensor = torch.from_numpy(np.array(od_img)).permute(2, 0, 1).float() / 255.0
            os_tensor = torch.from_numpy(np.array(os_img)).permute(2, 0, 1).float() / 255.0

        # For I-JEPA pretraining, use only OD image (3 channels)
        # The model expects (C, H, W) format with 3 channels
        # Return the tensor in the format expected by the mask collator
        # The mask collator will handle batching and create masks
        return od_tensor  # (C, H, W)