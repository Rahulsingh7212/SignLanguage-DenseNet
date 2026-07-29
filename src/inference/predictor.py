# ============================================================
# src/inference/predictor.py
# DenseNet Inference Engine — Single Frame Prediction
# ============================================================

import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import tensorflow as tf
from tensorflow import keras

sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.models import densenet_model

class SignPredictor:
    """
    Runs DenseNet121 inference on a single preprocessed frame.

    Pipeline:
        BGR frame → resize → normalize → model.predict()
                 → top-K predictions with confidence scores
    """

    def __init__(
        self,
        model_path  : Path  = config.INFERENCE_MODEL_PATH,
        image_size  : int   = config.INFERENCE_IMAGE_SIZE,
        classes     : list  = config.CLASSES,
        top_k       : int   = 5
    ):
        self.image_size = image_size
        self.classes    = classes
        self.num_classes = len(classes)
        self.top_k      = top_k

        # Performance tracking
        self.inference_times : List[float] = []
        self.frame_count     : int         = 0

        # Load model
        self.model = self._load_model(model_path)

        # Warm up model (first inference is always slow)
        self._warmup()

    # ── MODEL LOADING ────────────────────────────────────────
    def _load_model(self, model_path: Path) -> keras.Model:
        """Load trained Keras model from disk."""
        print(f"\n  📂 Loading model: {model_path.name}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please complete Stage 3 training first."
            )

        start = time.time()
        model = keras.models.load_model(
    str(model_path),
    safe_mode=False,
    compile=False
)
        elapsed = time.time() - start

        print(f"  ✅ Model loaded in {elapsed:.2f}s")
        print(f"     Input shape:  {model.input_shape}")
        print(f"     Output shape: {model.output_shape}")
        print(f"     Parameters:   {model.count_params():,}")

        return model

    def _warmup(self) -> None:
        """Run dummy inference to warm up TensorFlow."""
        print("  🔥 Warming up model...")
        dummy = np.zeros(
            (1, self.image_size, self.image_size, 3),
            dtype=np.float32
        )
        for _ in range(3):
            _ = self.model.predict(dummy, verbose=0)
        print("  ✅ Model ready for real-time inference!")

    # ── PREPROCESSING ────────────────────────────────────────
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR crop for DenseNet inference.

        Steps:
            1. Resize to 224×224
            2. BGR → RGB
            3. Normalize to [0, 1]
            4. Add batch dimension → (1, 224, 224, 3)
        """
        # Resize
        resized = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LANCZOS4
        )

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize 0–1
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    # ── PREDICTION ───────────────────────────────────────────
    def predict(
        self,
        frame: np.ndarray
    ) -> Dict:
        """
        Run full prediction pipeline on one BGR frame.

        Args:
            frame: BGR numpy array (hand crop from MediaPipe)

        Returns:
            {
                'top1_class'      : str,    # e.g. 'A'
                'top1_confidence' : float,  # e.g. 0.92
                'top_k'           : list,   # [(class, conf), ...]
                'all_probs'       : array,  # All 29 probabilities
                'inference_ms'    : float   # Time in milliseconds
            }
        """
        self.frame_count += 1

        # Preprocess
        input_tensor = self.preprocess_frame(frame)

        # Run inference
        t_start = time.time()
        probs   = self.model.predict(input_tensor, verbose=0)[0]
        t_end   = time.time()

        # Timing
        inference_ms = (t_end - t_start) * 1000
        self.inference_times.append(inference_ms)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)

        # Top-1 prediction
        top1_idx        = int(np.argmax(probs))
        top1_class      = self.classes[top1_idx]
        top1_confidence = float(probs[top1_idx])

        # Top-K predictions
        top_k_idx = np.argsort(probs)[::-1][:self.top_k]
        top_k_preds = [
            (self.classes[i], float(probs[i]))
            for i in top_k_idx
        ]

        return {
            'top1_class'      : top1_class,
            'top1_confidence' : top1_confidence,
            'top_k'           : top_k_preds,
            'all_probs'       : probs,
            'inference_ms'    : inference_ms
        }

    def predict_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[Dict]:
        """Run prediction on multiple frames at once."""
        preprocessed = np.vstack([
            self.preprocess_frame(f) for f in frames
        ])

        t_start  = time.time()
        all_probs = self.model.predict(preprocessed, verbose=0)
        t_end    = time.time()

        results = []
        for probs in all_probs:
            top1_idx   = int(np.argmax(probs))
            top_k_idx  = np.argsort(probs)[::-1][:self.top_k]
            results.append({
                'top1_class'      : self.classes[top1_idx],
                'top1_confidence' : float(probs[top1_idx]),
                'top_k'           : [(self.classes[i], float(probs[i]))
                                      for i in top_k_idx],
                'all_probs'       : probs,
                'inference_ms'    : (t_end - t_start) * 1000 / len(frames)
            })

        return results

    # ── STATS ────────────────────────────────────────────────
    def get_avg_fps(self) -> float:
        """Calculate average inference FPS."""
        if not self.inference_times:
            return 0.0
        avg_ms = np.mean(self.inference_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def get_avg_latency(self) -> float:
        """Average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times))

    def reset_stats(self) -> None:
        """Reset performance counters."""
        self.inference_times = []
        self.frame_count     = 0