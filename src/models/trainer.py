# ============================================================
# src/models/trainer.py
# Keras Data Generator + Two-Phase Training Logic
# ============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


# ════════════════════════════════════════════════════════════
# DATA GENERATORS (Keras ImageDataGenerator)
# ════════════════════════════════════════════════════════════

def get_train_generator(
    train_dir   : Path = config.TRAIN_DIR,
    batch_size  : int  = config.BATCH_SIZE,
    image_size  : int  = config.IMAGE_SIZE,
    seed        : int  = config.RANDOM_SEED
) -> keras.preprocessing.image.DirectoryIterator:
    """
    Training data generator with augmentation.
    Reads images from directory structure: train/A/, train/B/, etc.
    """

    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale             = 1./255,           # Normalize 0-1
        rotation_range      = 15,               # ±15°
        width_shift_range   = 0.05,             # Horizontal shift
        height_shift_range  = 0.05,             # Vertical shift
        zoom_range          = [0.9, 1.1],       # Zoom ±10%
        horizontal_flip     = True,             # Mirror signs
        brightness_range    = [0.8, 1.2],       # Brightness variation
        shear_range         = 5.0,              # Slight shear
        fill_mode           = 'constant',       # Fill with black
        cval                = 0
    )

    train_generator = train_datagen.flow_from_directory(
        directory   = str(train_dir),
        target_size = (image_size, image_size),
        color_mode  = 'rgb',
        classes     = config.CLASSES,
        class_mode  = 'categorical',            # One-hot encoded
        batch_size  = batch_size,
        shuffle     = True,
        seed        = seed
    )

    print(f"  ✅ Train Generator:")
    print(f"     Images:  {train_generator.n:,}")
    print(f"     Classes: {train_generator.num_classes}")
    print(f"     Batches: {len(train_generator)}")

    return train_generator


def get_val_generator(
    val_dir     : Path = config.VAL_DIR,
    batch_size  : int  = config.BATCH_SIZE,
    image_size  : int  = config.IMAGE_SIZE
) -> keras.preprocessing.image.DirectoryIterator:
    """
    Validation generator — NO augmentation, only rescaling.
    """

    val_datagen = ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(
        directory   = str(val_dir),
        target_size = (image_size, image_size),
        color_mode  = 'rgb',
        classes     = config.CLASSES,
        class_mode  = 'categorical',
        batch_size  = batch_size,
        shuffle     = False
    )

    print(f"  ✅ Val Generator:")
    print(f"     Images:  {val_generator.n:,}")
    print(f"     Batches: {len(val_generator)}")

    return val_generator


def get_test_generator(
    test_dir    : Path = config.TEST_DIR,
    batch_size  : int  = config.BATCH_SIZE,
    image_size  : int  = config.IMAGE_SIZE
) -> keras.preprocessing.image.DirectoryIterator:
    """
    Test generator — NO augmentation, NO shuffle.
    """

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory   = str(test_dir),
        target_size = (image_size, image_size),
        color_mode  = 'rgb',
        classes     = config.CLASSES,
        class_mode  = 'categorical',
        batch_size  = batch_size,
        shuffle     = False             # IMPORTANT: keep order for evaluation
    )

    print(f"  ✅ Test Generator:")
    print(f"     Images:  {test_generator.n:,}")
    print(f"     Batches: {len(test_generator)}")

    return test_generator


# ════════════════════════════════════════════════════════════
# HISTORY PLOTTER
# ════════════════════════════════════════════════════════════

def plot_training_history(
    history         : keras.callbacks.History,
    phase           : int,
    save_path       : Path = None
) -> None:
    """
    Plot loss and accuracy curves for one training phase.
    """

    hist = history.history

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f'Training History — Phase {phase} '
        f'({"Head Training" if phase == 1 else "Fine-Tuning"})',
        fontsize=16, fontweight='bold'
    )

    epochs = range(1, len(hist['loss']) + 1)

    # ── Plot 1: Loss ─────────────────────────────────────────
    axes[0].plot(epochs, hist['loss'],     'b-o', label='Train Loss',  linewidth=2)
    axes[0].plot(epochs, hist['val_loss'], 'r-o', label='Val Loss',    linewidth=2)
    axes[0].set_title('Loss Curve',   fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(hist['val_loss']) + 1
    axes[0].axvline(x=best_epoch, color='green', linestyle='--',
                    alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[0].legend()

    # ── Plot 2: Accuracy ─────────────────────────────────────
    axes[1].plot(epochs, hist['accuracy'],     'b-o', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, hist['val_accuracy'], 'r-o', label='Val Acc',   linewidth=2)
    axes[1].set_title('Accuracy Curve', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    best_val_acc = max(hist['val_accuracy'])
    axes[1].axhline(y=best_val_acc, color='green', linestyle='--',
                     alpha=0.7, label=f'Best: {best_val_acc:.4f}')
    axes[1].legend()

    # ── Plot 3: Learning Rate ─────────────────────────────────
    if 'lr' in hist:
        axes[2].plot(epochs, hist['lr'], 'g-o', linewidth=2)
        axes[2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'LR data not available',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Training plot saved: {save_path.name}")

    plt.show()


def save_history_csv(
    history : keras.callbacks.History,
    phase   : int
) -> None:
    """Save training history to CSV."""
    df = pd.DataFrame(history.history)
    df['epoch'] = range(1, len(df) + 1)

    csv_path = config.TRAINING_LOGS_DIR / f"phase{phase}_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ History saved: {csv_path.name}")


def combine_histories(
    history1 : keras.callbacks.History,
    history2 : keras.callbacks.History,
    save_path: Path = None
) -> None:
    """
    Plot combined training curve for both phases.
    Shows the full training story in one graph.
    """
    h1 = history1.history
    h2 = history2.history

    # Combine
    combined_loss     = h1['loss']     + h2['loss']
    combined_val_loss = h1['val_loss'] + h2['val_loss']
    combined_acc      = h1['accuracy'] + h2['accuracy']
    combined_val_acc  = h1['val_accuracy'] + h2['val_accuracy']

    phase1_end  = len(h1['loss'])
    total_epochs = len(combined_loss)
    epochs       = range(1, total_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Complete Training History — Phase 1 + Phase 2',
                 fontsize=18, fontweight='bold')

    for ax, train_data, val_data, title, ylabel in [
        (axes[0], combined_loss,    combined_val_loss,
         'Combined Loss Curve',     'Loss'),
        (axes[1], combined_acc,     combined_val_acc,
         'Combined Accuracy Curve', 'Accuracy'),
    ]:
        ax.plot(epochs, train_data, 'b-o', label='Train',
                linewidth=2, markersize=4)
        ax.plot(epochs, val_data,   'r-o', label='Val',
                linewidth=2, markersize=4)

        # Phase divider
        ax.axvline(x=phase1_end + 0.5, color='orange', linestyle='--',
                   linewidth=2, alpha=0.8, label='Phase 1→2 boundary')

        ax.text(phase1_end/2, ax.get_ylim()[0],
                'Phase 1\n(Head)', ha='center', fontsize=10,
                color='blue', alpha=0.7)
        ax.text(phase1_end + (total_epochs - phase1_end)/2, ax.get_ylim()[0],
                'Phase 2\n(Fine-tune)', ha='center', fontsize=10,
                color='red', alpha=0.7)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Combined history plot saved: {save_path.name}")

    plt.show()