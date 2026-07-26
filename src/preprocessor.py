# ============================================================
# src/preprocessor.py
# Image Preprocessing: Resize + Normalize + Save
# ============================================================

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config


class ImagePreprocessor:
    """
    Handles all image preprocessing steps:
    1. Resize to 224×224
    2. Normalize (0–1 or ImageNet stats)
    3. Save processed images
    """

    def __init__(
        self,
        target_size : int   = config.IMAGE_SIZE,
        normalize   : bool  = False,    # False = save as uint8 image
        mean        : list  = config.NORMALIZE_MEAN,
        std         : list  = config.NORMALIZE_STD
    ):
        self.target_size = target_size
        self.normalize   = normalize
        self.mean        = np.array(mean, dtype=np.float32)
        self.std         = np.array(std,  dtype=np.float32)

    # ── CORE PIPELINE ────────────────────────────────────────
    def process(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Full preprocessing pipeline:
        BGR image → resize → convert to RGB → normalize (optional)

        Returns numpy array ready for the model
        """
        # Step 1: Resize
        resized = self._resize(image)

        # Step 2: BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Step 3: Normalize (for model inference)
        if self.normalize:
            return self._normalize_imagenet(rgb)

        return rgb  # uint8, 0-255

    def process_for_model(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Returns float32 tensor normalized for model input.
        Shape: (3, 224, 224) for PyTorch or (224, 224, 3) for TF
        """
        processed = self.process(image)

        # Convert to float and normalize 0–1
        float_img = processed.astype(np.float32) / 255.0

        # ImageNet normalization
        normalized = (float_img - self.mean) / self.std

        # HWC → CHW for PyTorch
        return normalized.transpose(2, 0, 1)

    # ── HELPER METHODS ───────────────────────────────────────
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize with aspect ratio preservation + center padding.
        This avoids distorting hand shapes.
        """
        h, w = image.shape[:2]
        target = self.target_size

        # Calculate scale keeping aspect ratio
        scale = min(target / w, target / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Create black canvas and paste centered
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        x_offset = (target - new_w) // 2
        y_offset = (target - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def _normalize_imagenet(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet normalization.
        Input: uint8 HWC RGB
        Output: float32 HWC normalized
        """
        img_float = image.astype(np.float32) / 255.0
        normalized = (img_float - self.mean) / self.std
        return normalized

    def save_image(
        self,
        image       : np.ndarray,
        output_path : Path,
        quality     : int = 95
    ) -> bool:
        """
        Save processed image to disk.
        Input should be uint8 RGB numpy array.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB → BGR for OpenCV saving
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
        except Exception as e:
            print(f"   ❌ Save error: {e}")
            return False

    def load_and_process(self, image_path: Path) -> Optional[np.ndarray]:
        """Load an image file and run full pipeline."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            return self.process(img)
        except Exception as e:
            print(f"   ❌ Load error {image_path.name}: {e}")
            return None