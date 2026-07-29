# ============================================================
# src/inference/stability_buffer.py
# Stability Buffer — Prevents Flickering Predictions
# ============================================================

import sys
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class StabilityBuffer:
    """
    Prevents flickering by requiring CONSISTENT predictions
    over multiple consecutive frames before confirming a letter.

    How it works:
        Frame 1: prediction='A', confidence=0.92  → buffer=['A']
        Frame 2: prediction='A', confidence=0.88  → buffer=['A','A']
        Frame 3: prediction='M', confidence=0.71  → buffer=['A','A','M'] RESET!
        Frame 4: prediction='A', confidence=0.95  → buffer=['A']
        Frame 5: prediction='A', confidence=0.90  → buffer=['A','A']
        ...
        Frame 8: prediction='A', confidence=0.85  → buffer=['A','A','A','A','A']
                                                    CONFIRMED! → output 'A'
    """

    def __init__(
        self,
        required_frames     : int   = config.STABILITY_FRAMES,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        cooldown_frames     : int   = config.COOLDOWN_FRAMES
    ):
        """
        Args:
            required_frames      : Consecutive frames needed to confirm
            confidence_threshold : Minimum confidence to count a frame
            cooldown_frames      : Wait N frames after confirming a letter
        """
        self.required_frames        = required_frames
        self.confidence_threshold   = confidence_threshold
        self.cooldown_frames        = cooldown_frames

        # Internal state
        self._buffer        : deque = deque(maxlen=required_frames)
        self._cooldown_count: int   = 0
        self._confirmed_count: int  = 0

        # History for visualization
        self.last_confirmed : Optional[str]  = None
        self.buffer_history : List[str]      = []

    # ── MAIN METHOD ──────────────────────────────────────────
    def update(
        self,
        prediction  : str,
        confidence  : float
    ) -> Tuple[Optional[str], dict]:
        """
        Update buffer with new prediction.

        Args:
            prediction : Predicted class name (e.g., 'A')
            confidence : Model confidence 0.0–1.0

        Returns:
            (confirmed_letter, status_dict)
            - confirmed_letter: letter if confirmed, None otherwise
            - status_dict     : current buffer state for UI display
        """

        # ── Handle cooldown period ───────────────────────────
        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            status = self._build_status(
                prediction, confidence,
                state='cooldown',
                confirmed=None
            )
            return None, status

        # ── Check confidence threshold ───────────────────────
        if confidence < self.confidence_threshold:
            # Low confidence — clear buffer
            self._buffer.clear()
            status = self._build_status(
                prediction, confidence,
                state='low_confidence',
                confirmed=None
            )
            return None, status

        # ── Skip 'nothing' class ─────────────────────────────
        if prediction == 'nothing':
            self._buffer.clear()
            status = self._build_status(
                prediction, confidence,
                state='nothing',
                confirmed=None
            )
            return None, status

        # ── Add to buffer ────────────────────────────────────
        self._buffer.append(prediction)

        # ── Check if buffer is consistent ────────────────────
        if len(self._buffer) == self.required_frames:
            all_same = len(set(self._buffer)) == 1
            current  = self._buffer[-1]

            if all_same:
                # ── CONFIRMED! ───────────────────────────────
                confirmed = current
                self.last_confirmed  = confirmed
                self._confirmed_count += 1

                # Start cooldown
                self._cooldown_count = self.cooldown_frames
                self._buffer.clear()

                status = self._build_status(
                    prediction, confidence,
                    state='confirmed',
                    confirmed=confirmed
                )
                return confirmed, status

            else:
                # Inconsistent — remove oldest, keep recent
                status = self._build_status(
                    prediction, confidence,
                    state='building',
                    confirmed=None
                )
                return None, status

        # ── Still building buffer ────────────────────────────
        status = self._build_status(
            prediction, confidence,
            state='building',
            confirmed=None
        )
        return None, status

    # ── HELPER METHODS ───────────────────────────────────────
    def _build_status(
        self,
        prediction  : str,
        confidence  : float,
        state       : str,
        confirmed   : Optional[str]
    ) -> dict:
        """Build status dictionary for UI rendering."""
        return {
            'state'             : state,
            'current_prediction': prediction,
            'confidence'        : confidence,
            'buffer_content'    : list(self._buffer),
            'buffer_fill'       : len(self._buffer),
            'buffer_required'   : self.required_frames,
            'progress'          : len(self._buffer) / self.required_frames,
            'cooldown_remaining': self._cooldown_count,
            'confirmed'         : confirmed,
            'total_confirmed'   : self._confirmed_count,
            'is_cooling_down'   : self._cooldown_count > 0
        }

    def reset(self) -> None:
        """Reset the entire buffer state."""
        self._buffer.clear()
        self._cooldown_count  = 0
        self.last_confirmed   = None

    def force_confirm(self, letter: str) -> None:
        """Manually confirm a letter (for testing)."""
        self.last_confirmed   = letter
        self._confirmed_count += 1
        self._buffer.clear()
        self._cooldown_count  = self.cooldown_frames

    @property
    def is_cooling_down(self) -> bool:
        return self._cooldown_count > 0

    @property
    def current_buffer(self) -> List[str]:
        return list(self._buffer)

    @property
    def total_confirmed(self) -> int:
        return self._confirmed_count

    def get_stats(self) -> dict:
        """Return buffer statistics."""
        return {
            'total_confirmed'   : self._confirmed_count,
            'required_frames'   : self.required_frames,
            'confidence_threshold': self.confidence_threshold,
            'cooldown_frames'   : self.cooldown_frames,
            'last_confirmed'    : self.last_confirmed
        }