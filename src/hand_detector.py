# ============================================================
# src/hand_detector.py
# MediaPipe-based Hand Detection & ROI Extraction
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple


class HandDetector:
    """
    Detects hand region in an image using MediaPipe Hands.
    Returns cropped, padded bounding box of the hand.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        padding: float = 0.25,
        fallback_to_full: bool = True
    ):
        """
        Args:
            static_image_mode       : True for images (not video)
            max_num_hands           : Detect only 1 hand per image
            min_detection_confidence: Lower = catches more hands
            padding                 : % padding around bounding box
            fallback_to_full        : Return full image if no hand found
        """
        self.padding           = padding
        self.fallback_to_full  = fallback_to_full

        # Initialize MediaPipe
        self.mp_hands    = mp.solutions.hands
        self.mp_drawing  = mp.solutions.drawing_utils
        self.hands       = self.mp_hands.Hands(
            static_image_mode        = static_image_mode,
            max_num_hands            = max_num_hands,
            min_detection_confidence = min_detection_confidence
        )

        # Tracking
        self.stats = {
            'total'     : 0,
            'detected'  : 0,
            'fallback'  : 0,
            'failed'    : 0
        }

    # ── PUBLIC METHOD ────────────────────────────────────────
    def detect_and_crop(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Main method: detect hand and return cropped ROI.

        Args:
            image: BGR image (as loaded by cv2)

        Returns:
            (cropped_image, hand_detected)
            - cropped_image : numpy array BGR image of hand region
            - hand_detected : True if MediaPipe found a hand
        """
        self.stats['total'] += 1

        h, w = image.shape[:2]

        # MediaPipe needs RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(rgb_image)

        # ── CASE 1: Hand detected ────────────────────────────
        if results.multi_hand_landmarks:
            self.stats['detected'] += 1
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(0.0, min(x_coords) - self.padding)
            x_max = min(1.0, max(x_coords) + self.padding)
            y_min = max(0.0, min(y_coords) - self.padding)
            y_max = min(1.0, max(y_coords) + self.padding)

            # Convert to pixel coordinates
            px_x1 = int(x_min * w)
            px_y1 = int(y_min * h)
            px_x2 = int(x_max * w)
            px_y2 = int(y_max * h)

            # Safety check
            if px_x2 > px_x1 and px_y2 > px_y1:
                crop = image[px_y1:px_y2, px_x1:px_x2]
                return crop, True

        # ── CASE 2: No hand — use full image as fallback ─────
        if self.fallback_to_full:
            self.stats['fallback'] += 1
            return image, False

        # ── CASE 3: Complete failure ─────────────────────────
        self.stats['failed'] += 1
        return None, False

    def detect_with_landmarks(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """
        Returns the hand crop AND landmark data for visualization.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0], results
        return None, None

    def draw_landmarks(
        self,
        image: np.ndarray,
        results: object
    ) -> np.ndarray:
        """
        Draw MediaPipe hand skeleton on image for visualization.
        """
        annotated = image.copy()
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return annotated

    def get_stats(self) -> dict:
        """Return detection statistics."""
        total = self.stats['total']
        if total > 0:
            return {
                **self.stats,
                'detection_rate' : f"{self.stats['detected']/total*100:.1f}%",
                'fallback_rate'  : f"{self.stats['fallback']/total*100:.1f}%",
                'failure_rate'   : f"{self.stats['failed']/total*100:.1f}%"
            }
        return self.stats

    def reset_stats(self):
        """Reset tracking counters."""
        self.stats = {'total': 0, 'detected': 0, 'fallback': 0, 'failed': 0}

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()