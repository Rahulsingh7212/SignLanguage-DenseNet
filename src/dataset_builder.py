# ============================================================
# src/dataset_builder.py
# PyTorch Dataset + DataLoader + Train/Val/Test Split
# ============================================================

import os
import sys
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent.parent))
import config
from src.augmentation import get_train_transforms, get_val_transforms


# ════════════════════════════════════════════════════════════
# PYTORCH DATASET CLASS
# ════════════════════════════════════════════════════════════

class ASLDataset(Dataset):
    """
    Custom PyTorch Dataset for ASL Sign Language images.

    Usage:
        dataset = ASLDataset(root_dir=config.TRAIN_DIR, split='train')
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        root_dir    : Path,
        split       : str = 'train',
        transform   = None
    ):
        """
        Args:
            root_dir  : Path to train/ val/ or test/ directory
            split     : 'train', 'val', or 'test'
            transform : albumentations or torchvision transforms
        """
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.transform  = transform
        self.classes    = config.CLASSES
        self.class_to_idx = config.CLASS_TO_IDX

        # Collect all image paths + labels
        self.samples = self._load_samples()

        print(f"  📂 {split.upper()} Dataset: {len(self.samples)} images | "
              f"{len(self.classes)} classes")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Scan directory and collect (image_path, label_idx) pairs."""
        samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            label = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append((img_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Count images per class for imbalance checking."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            cls_name = config.IDX_TO_CLASS[label]
            counts[cls_name] += 1
        return counts


# ════════════════════════════════════════════════════════════
# DATASET SPLITTER
# ════════════════════════════════════════════════════════════

class DatasetSplitter:
    """
    Splits the processed dataset into train/val/test.
    Uses stratified split to maintain class balance.
    """

    def __init__(
        self,
        source_dir  : Path = config.HAND_CROPS_DIR,
        train_dir   : Path = config.TRAIN_DIR,
        val_dir     : Path = config.VAL_DIR,
        test_dir    : Path = config.TEST_DIR,
        train_ratio : float = config.TRAIN_RATIO,
        val_ratio   : float = config.VAL_RATIO,
        random_seed : int   = config.RANDOM_SEED
    ):
        self.source_dir  = source_dir
        self.train_dir   = train_dir
        self.val_dir     = val_dir
        self.test_dir    = test_dir
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = 1.0 - train_ratio - val_ratio
        self.random_seed = random_seed

    def split(self, verbose: bool = True) -> Dict:
        """
        Perform stratified split and copy files to train/val/test dirs.
        """
        if verbose:
            print("\n" + "="*60)
            print("  📦 DATASET SPLITTING")
            print(f"     Train: {self.train_ratio*100:.0f}% | "
                  f"Val: {self.val_ratio*100:.0f}% | "
                  f"Test: {self.test_ratio*100:.0f}%")
            print("="*60)

        all_paths  = []
        all_labels = []

        # Collect all files
        for class_name in config.CLASSES:
            class_dir = self.source_dir / class_name
            if not class_dir.exists():
                if verbose:
                    print(f"   ⚠️  Skipping missing class: {class_name}")
                continue

            images = list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png"))

            all_paths.extend(images)
            all_labels.extend([class_name] * len(images))

        # ── Stratified split ─────────────────────────────────
        # First: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels,
            test_size    = self.test_ratio,
            stratify     = all_labels,
            random_state = self.random_seed
        )

        # Then: separate val from remaining
        relative_val = self.val_ratio / (self.train_ratio + self.val_ratio)

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size    = relative_val,
            stratify     = train_val_labels,
            random_state = self.random_seed
        )

        if verbose:
            print(f"\n  📊 Split Results:")
            print(f"     Train: {len(train_paths):,} images")
            print(f"     Val:   {len(val_paths):,} images")
            print(f"     Test:  {len(test_paths):,} images")
            print(f"     Total: {len(all_paths):,} images")

        # ── Copy files ───────────────────────────────────────
        splits = [
            (train_paths, train_labels, self.train_dir, "TRAIN"),
            (val_paths,   val_labels,   self.val_dir,   "VAL"),
            (test_paths,  test_labels,  self.test_dir,  "TEST"),
        ]

        stats = {}

        for paths, labels, dest_dir, split_name in splits:
            if verbose:
                print(f"\n  📁 Copying {split_name} set...")

            dest_dir.mkdir(parents=True, exist_ok=True)
            copied = 0

            for src_path, class_name in tqdm(
                zip(paths, labels),
                total=len(paths),
                desc=f"  {split_name}",
                disable=not verbose
            ):
                dest_class_dir = dest_dir / class_name
                dest_class_dir.mkdir(exist_ok=True)
                dest_path = dest_class_dir / src_path.name

                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                    copied += 1

            stats[split_name] = {'total': len(paths), 'copied': copied}
            if verbose:
                print(f"     ✅ {copied} files copied")

        if verbose:
            print("\n  ✅ Dataset split complete!")

        return stats


# ════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ════════════════════════════════════════════════════════════

def create_dataloaders(
    batch_size  : int = config.BATCH_SIZE,
    num_workers : int = 0,             # 0 = safe for Windows
    pin_memory  : bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Datasets
    train_dataset = ASLDataset(
        root_dir  = config.TRAIN_DIR,
        split     = 'train',
        transform = get_train_transforms()
    )
    val_dataset = ASLDataset(
        root_dir  = config.VAL_DIR,
        split     = 'val',
        transform = get_val_transforms()
    )
    test_dataset = ASLDataset(
        root_dir  = config.TEST_DIR,
        split     = 'test',
        transform = get_val_transforms()
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,           # Shuffle ONLY training data
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory
    )

    print(f"\n  ✅ DataLoaders Created:")
    print(f"     Train batches: {len(train_loader)}")
    print(f"     Val batches:   {len(val_loader)}")
    print(f"     Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader