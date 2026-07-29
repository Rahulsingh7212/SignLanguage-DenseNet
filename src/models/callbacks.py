# ============================================================
# src/models/callbacks.py
# Custom Training Callbacks
# ============================================================

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def get_callbacks(
    phase           : int   = 1,
    checkpoint_path : Path  = config.CHECKPOINT_PATH,
    logs_dir        : Path  = config.LOGS_DIR
) -> list:
    """
    Get all callbacks for training.

    Args:
        phase: 1 = head training, 2 = fine-tuning
        checkpoint_path: Where to save best model weights
        logs_dir: TensorBoard log directory

    Returns:
        List of Keras callbacks
    """

    # ── 1. Model Checkpoint ──────────────────────────────────
    # Saves ONLY when val_accuracy improves
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath        = str(checkpoint_path),
        monitor         = 'val_accuracy',
        mode            = 'max',
        save_best_only  = True,
        save_weights_only = False,
        verbose         = 1
    )

    # ── 2. Early Stopping ────────────────────────────────────
    # Stops training if val_accuracy doesn't improve for N epochs
    early_stop = keras.callbacks.EarlyStopping(
        monitor             = 'val_accuracy',
        patience            = config.EARLY_STOP_PATIENCE,
        mode                = 'max',
        restore_best_weights = True,    # Restore best weights on stop
        verbose             = 1
    )

    # ── 3. Reduce LR on Plateau ──────────────────────────────
    # Reduces learning rate when training stagnates
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor     = 'val_loss',
        factor      = config.REDUCE_LR_FACTOR,
        patience    = config.REDUCE_LR_PATIENCE,
        min_lr      = config.REDUCE_LR_MIN,
        mode        = 'min',
        verbose     = 1
    )

    # ── 4. TensorBoard ───────────────────────────────────────
    log_dir = logs_dir / f"phase{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard = keras.callbacks.TensorBoard(
        log_dir             = str(log_dir),
        histogram_freq      = 1,
        write_graph         = True,
        write_images        = False,
        update_freq         = 'epoch',
        profile_batch       = 0
    )

    # ── 5. CSV Logger ─────────────────────────────────────────
    # Saves training history to CSV for later analysis
    csv_path = config.TRAINING_LOGS_DIR / f"phase{phase}_history.csv"
    csv_logger = keras.callbacks.CSVLogger(
        filename = str(csv_path),
        append   = False
    )

    # ── 6. Custom: Learning Rate Logger ──────────────────────
    lr_logger = LRLogger()

    # ── 7. Custom: Training Progress Printer ─────────────────
    progress_printer = TrainingProgressPrinter(phase=phase)

    callbacks = [
        checkpoint,
        early_stop,
        reduce_lr,
        tensorboard,
        csv_logger,
        lr_logger,
        progress_printer
    ]

    print(f"  ✅ Callbacks configured for Phase {phase}:")
    print(f"     ✓ ModelCheckpoint → {checkpoint_path.name}")
    print(f"     ✓ EarlyStopping (patience={config.EARLY_STOP_PATIENCE})")
    print(f"     ✓ ReduceLROnPlateau (patience={config.REDUCE_LR_PATIENCE})")
    print(f"     ✓ TensorBoard → {log_dir.name}")
    print(f"     ✓ CSVLogger → phase{phase}_history.csv")

    return callbacks


# ════════════════════════════════════════════════════════════
# CUSTOM CALLBACKS
# ════════════════════════════════════════════════════════════

class LRLogger(keras.callbacks.Callback):
    """Logs current learning rate at each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if logs is not None:
            logs['lr'] = lr


class TrainingProgressPrinter(keras.callbacks.Callback):
    """
    Prints a clean summary table after each epoch.
    """

    def __init__(self, phase: int = 1):
        super().__init__()
        self.phase      = phase
        self.start_time = None
        self.best_val   = 0.0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'='*70}")
        print(f"  🚀 PHASE {self.phase} TRAINING STARTED")
        print(f"{'='*70}")
        print(f"  {'Epoch':<8} {'Loss':<10} {'Acc':<10} "
              f"{'Val Loss':<12} {'Val Acc':<12} {'LR':<12}")
        print(f"  {'-'*68}")

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        loss    = logs.get('loss', 0)
        acc     = logs.get('accuracy', 0)
        val_loss= logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        lr      = float(keras.backend.get_value(self.model.optimizer.learning_rate))

        # Track best validation accuracy
        marker = ""
        if val_acc > self.best_val:
            self.best_val = val_acc
            marker = " ⭐"

        elapsed = (time.time() - self.start_time) / 60

        print(f"  {epoch+1:<8} {loss:<10.4f} {acc:<10.4f} "
              f"{val_loss:<12.4f} {val_acc:<12.4f} "
              f"{lr:<12.2e}{marker}")

    def on_train_end(self, logs=None):
        elapsed = (time.time() - self.start_time) / 60
        print(f"  {'-'*68}")
        print(f"  ✅ Phase {self.phase} Complete!")
        print(f"     Best Val Accuracy: {self.best_val:.4f} ({self.best_val*100:.2f}%)")
        print(f"     Time elapsed:      {elapsed:.1f} minutes")
        print(f"{'='*70}\n")