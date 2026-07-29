# ============================================================
# src/models/densenet_model.py
# DenseNet121 Model Architecture with Two-Phase Training
# ============================================================

import sys
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def build_densenet_model(
    num_classes     : int   = config.NUM_CLASSES,
    image_size      : int   = config.IMAGE_SIZE,
    dropout_rate    : float = config.DROPOUT_RATE,
    dense_units     : int   = config.DENSE_UNITS,
    weights         : str   = config.DENSENET_WEIGHTS,
    trainable_base  : bool  = False
) -> Model:
    """
    Build DenseNet121 model with custom classification head.

    Architecture:
        Input (224, 224, 3)
            ↓
        DenseNet121 Base (ImageNet pretrained)
            ↓
        GlobalAveragePooling2D
            ↓
        BatchNormalization
            ↓
        Dense(512, ReLU)
            ↓
        Dropout(0.4)
            ↓
        Dense(256, ReLU)
            ↓
        Dropout(0.3)
            ↓
        Dense(29, Softmax)
    """

    # ── Input Layer ──────────────────────────────────────────
    inputs = keras.Input(
        shape=(image_size, image_size, IMAGE_CHANNELS := 3),
        name="input_layer"
    )

    # ── Preprocessing ────────────────────────────────────────
    # DenseNet expects pixels in range [0, 255] normalized by preprocess_input
    x = layers.Lambda(
        lambda img: preprocess_input(img),
        name="densenet_preprocessing"
    )(inputs)

    # ── DenseNet121 Base ─────────────────────────────────────
    base_model = DenseNet121(
        include_top     = False,        # Remove original classifier
        weights         = weights,      # 'imagenet' or None
        input_tensor    = x,
        pooling         = None,         # We add our own pooling
        input_shape     = (image_size, image_size, 3)
    )
    base_model.trainable = trainable_base  # Freeze initially

    # Get base model output
    base_output = base_model.output     # Shape: (None, 7, 7, 1024)

    # ── Custom Classification Head ───────────────────────────
    # GlobalAveragePooling: (None, 7, 7, 1024) → (None, 1024)
    x = layers.GlobalAveragePooling2D(name="gap")(base_output)

    # Batch Normalization for stability
    x = layers.BatchNormalization(name="bn_head")(x)

    # Dense Block 1
    x = layers.Dense(
        dense_units,
        activation = 'relu',
        kernel_regularizer = keras.regularizers.l2(1e-4),
        name="dense_512"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # Dense Block 2
    x = layers.Dense(
        dense_units // 2,               # 256 units
        activation = 'relu',
        kernel_regularizer = keras.regularizers.l2(1e-4),
        name="dense_256"
    )(x)
    x = layers.Dropout(dropout_rate - 0.1, name="dropout_2")(x)

    # Output Layer
    outputs = layers.Dense(
        num_classes,
        activation = 'softmax',
        name       = "output_softmax"
    )(x)

    # ── Build Model ──────────────────────────────────────────
    model = Model(
        inputs  = base_model.input,
        outputs = outputs,
        name    = "DenseNet121_ASL"
    )

    return model, base_model


def compile_model(
    model       : Model,
    learning_rate: float,
    label_smoothing: float = 0.1
) -> Model:
    """
    Compile model with optimizer, loss, and metrics.
    Label smoothing helps with overconfident predictions.
    """
    optimizer = keras.optimizers.Adam(
        learning_rate = learning_rate,
        beta_1        = 0.9,
        beta_2        = 0.999,
        epsilon       = 1e-7
    )

    model.compile(
        optimizer = optimizer,
        loss      = keras.losses.CategoricalCrossentropy(
            label_smoothing = label_smoothing
        ),
        metrics   = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    return model


def freeze_base(model: Model, base_model: Model) -> Model:
    """Phase 1: Freeze ALL base layers."""
    base_model.trainable = False
    trainable   = sum(1 for l in model.layers if l.trainable)
    non_trainable = sum(1 for l in model.layers if not l.trainable)

    print(f"  🔒 Base model FROZEN")
    print(f"     Trainable layers:     {trainable}")
    print(f"     Non-trainable layers: {non_trainable}")

    return model


def unfreeze_top_layers(
    model       : Model,
    base_model  : Model,
    from_layer  : int = config.PHASE2_UNFREEZE_FROM
) -> Model:
    """
    Phase 2: Unfreeze layers from index `from_layer` onwards.
    This targets the last Dense Block of DenseNet121.
    """
    base_model.trainable = True

    # Refreeze everything before the threshold
    for layer in base_model.layers:
        if base_model.layers.index(layer) < from_layer:
            layer.trainable = False
        else:
            layer.trainable = True

    total       = len(base_model.layers)
    unfrozen    = sum(1 for l in base_model.layers if l.trainable)
    frozen      = total - unfrozen

    print(f"  🔓 Partially UNFROZEN (from layer {from_layer})")
    print(f"     Total base layers:  {total}")
    print(f"     Frozen layers:      {frozen}")
    print(f"     Unfrozen layers:    {unfrozen}")

    return model


def get_model_summary(model: Model) -> None:
    """Print detailed model summary with layer counts."""
    total_params     = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_trainable    = total_params - trainable_params

    print("\n" + "="*60)
    print(f"  📊 MODEL: {model.name}")
    print("="*60)
    print(f"  Total Parameters:     {total_params:>12,}")
    print(f"  Trainable:            {trainable_params:>12,}")
    print(f"  Non-trainable:        {non_trainable:>12,}")
    print(f"  Model size (approx):  {total_params * 4 / 1e6:>10.1f} MB")
    print("="*60)


def load_model(model_path: Path) -> Model:
    """Load a saved Keras model."""
    print(f"  📂 Loading model from: {model_path}")
    model = keras.models.load_model(str(model_path))
    print(f"  ✅ Model loaded successfully!")
    return model