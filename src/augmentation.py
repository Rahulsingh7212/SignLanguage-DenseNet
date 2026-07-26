# ============================================================
# src/augmentation.py
# Data Augmentation using Albumentations
# ============================================================

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Training augmentation pipeline.
    Applied ONLY to training data — not val/test.

    Augmentations chosen specifically for hand sign images:
    - Horizontal flip: mirrors signs (some signs are asymmetric — careful!)
    - Rotation ±15°: natural hand angle variation
    - Zoom 0.9–1.1: different distances from camera
    - Brightness/Contrast: lighting variation
    - Blur: simulates camera focus variation
    """
    return A.Compose([

        # ── Geometric transforms ─────────────────────────────
        A.HorizontalFlip(
            p=config.AUG_HFLIP_PROB
        ),
        A.Rotate(
            limit=config.AUG_ROTATION_LIMIT,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.7
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,        # Zoom ±10%
            rotate_limit=0,         # Rotation handled separately
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),

        # ── Color transforms ─────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS,
            contrast_limit=config.AUG_CONTRAST,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.4
        ),

        # ── Noise & blur ─────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=config.AUG_BLUR_PROB),

        A.GaussNoise(var_limit=(5, 20), p=0.2),

        # ── Cutout (occlusion simulation) ────────────────────
        A.CoarseDropout(
            max_holes=4,
            max_height=20,
            max_width=20,
            fill_value=0,
            p=0.2
        ),

        # ── Final resize (ensure correct size) ───────────────
        A.Resize(image_size, image_size),

        # ── Normalize for DenseNet ───────────────────────────
        A.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),

        # ── Convert to PyTorch tensor ─────────────────────────
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Validation/Test transforms.
    NO augmentation — only resize + normalize.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),
        ToTensorV2()
    ])


def get_inference_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Real-time webcam inference transforms.
    Same as validation — no augmentation.
    """
    return get_val_transforms(image_size)


def visualize_augmentations(
    image       : np.ndarray,
    n_samples   : int = 8,
    save_path   : Path = None
) -> None:
    """
    Show multiple augmented versions of the same image.
    Useful for verifying augmentation looks realistic.
    """
    import matplotlib.pyplot as plt

    # Basic transform without tensor conversion for visualization
    viz_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    ])

    cols = 4
    rows = (n_samples + cols) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('Data Augmentation Samples', fontsize=18, fontweight='bold')

    # First image = original
    axes[0][0].imshow(cv2.resize(image, (224, 224)))
    axes[0][0].set_title('ORIGINAL', fontweight='bold', color='green')
    axes[0][0].axis('off')

    # Rest = augmented
    for i in range(1, n_samples + 1):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            augmented = viz_transform(image=image)['image']
            axes[row][col].imshow(augmented)
            axes[row][col].set_title(f'Augmented #{i}', fontsize=9)
            axes[row][col].axis('off')

    # Hide empty subplots
    for i in range(n_samples + 1, rows * cols):
        axes[i // cols][i % cols].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Augmentation visualization saved: {save_path}")

    plt.show()