# ============================================================
# src/inference/sentence_builder.py
# Interactive Sentence Builder — Keyboard-Controlled
# ============================================================

import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class SentenceBuilder:
    """
    Builds words and sentences from confirmed sign letters.

    Keyboard Controls:
        SPACE     → Add word break (commit current word)
        BACKSPACE → Delete last letter
        ENTER     → Commit full sentence to history
        'c'       → Clear current word
        'C'       → Clear everything
        'del'     → Handled as backspace (ASL 'del' sign)

    ASL Special Signs:
        'space'   → Same as SPACE key
        'del'     → Same as BACKSPACE key
        'nothing' → Ignored (no hand / neutral position)
    """

    def __init__(
        self,
        max_sentence_length : int = config.MAX_SENTENCE_LENGTH,
        max_word_length     : int = config.MAX_WORD_LENGTH
    ):
        self.max_sentence_length = max_sentence_length
        self.max_word_length     = max_word_length

        # Current state
        self._current_word   : List[str] = []
        self._current_sentence: List[str] = []
        self._sentence_history: List[str] = []

        # Stats
        self._total_letters  : int   = 0
        self._total_words    : int   = 0
        self._start_time     : float = time.time()
        self._last_action    : str   = ""
        self._last_action_time: float = 0.0

    # ── MAIN INPUT METHODS ───────────────────────────────────

    def add_letter(self, letter: str) -> bool:
        """
        Add a confirmed letter to current word.

        Args:
            letter: Single character or special ('space', 'del', 'nothing')

        Returns:
            True if letter was added, False if ignored
        """

        # ── Handle special ASL signs ─────────────────────────
        if letter == 'nothing':
            return False

        if letter == 'space':
            return self.add_space()

        if letter == 'del':
            return self.backspace()

        # ── Regular letter ───────────────────────────────────
        if len(self._current_word) >= self.max_word_length:
            self._last_action = f"Word too long (max {self.max_word_length})"
            return False

        self._current_word.append(letter.upper())
        self._total_letters += 1
        self._last_action = f"Added '{letter.upper()}'"
        self._last_action_time = time.time()

        return True

    def add_space(self) -> bool:
        """Commit current word and add space."""
        if self._current_word:
            word = ''.join(self._current_word)
            self._current_sentence.append(word)
            self._total_words += 1
            self._current_word.clear()
            self._last_action = f"Word '{word}' committed"
            self._last_action_time = time.time()
            return True

        self._last_action = "No word to commit"
        return False

    def backspace(self) -> bool:
        """Delete last letter from current word."""
        if self._current_word:
            removed = self._current_word.pop()
            self._last_action = f"Deleted '{removed}'"
            self._last_action_time = time.time()
            return True

        elif self._current_sentence:
            # Delete last word
            last_word = self._current_sentence.pop()
            self._current_word = list(last_word)
            self._last_action = f"Restored word '{last_word}'"
            self._last_action_time = time.time()
            return True

        self._last_action = "Nothing to delete"
        return False

    def commit_sentence(self) -> str:
        """Commit full sentence to history."""
        # Add any remaining word
        if self._current_word:
            self.add_space()

        full_sentence = ' '.join(self._current_sentence)

        if full_sentence:
            self._sentence_history.append({
                'text'      : full_sentence,
                'timestamp' : time.time()
            })
            self._current_sentence.clear()
            self._last_action = "Sentence committed!"
            self._last_action_time = time.time()

        return full_sentence

    def clear_word(self) -> None:
        """Clear current word only."""
        self._current_word.clear()
        self._last_action = "Word cleared"
        self._last_action_time = time.time()

    def clear_all(self) -> None:
        """Clear everything."""
        self._current_word.clear()
        self._current_sentence.clear()
        self._last_action = "Everything cleared"
        self._last_action_time = time.time()

    # ── KEYBOARD HANDLER ─────────────────────────────────────
    def handle_keypress(self, key: int) -> str:
        """
        Handle keyboard input during demo.

        Args:
            key: cv2 waitKey() return value

        Returns:
            Action taken as string
        """
        if key == 32:       # SPACE
            self.add_space()
            return "space"

        elif key == 8:      # BACKSPACE
            self.backspace()
            return "backspace"

        elif key == 13:     # ENTER
            self.commit_sentence()
            return "enter"

        elif key == ord('c'):   # Clear word
            self.clear_word()
            return "clear_word"

        elif key == ord('C'):   # Clear all
            self.clear_all()
            return "clear_all"

        return "none"

    # ── DISPLAY PROPERTIES ───────────────────────────────────

    @property
    def current_word_str(self) -> str:
        """Current word as string."""
        return ''.join(self._current_word)

    @property
    def current_sentence_str(self) -> str:
        """Full sentence (committed words + current word)."""
        parts = self._current_sentence.copy()
        if self._current_word:
            parts.append(self.current_word_str)
        return ' '.join(parts)

    @property
    def display_sentence(self) -> str:
        """
        Sentence formatted for display — truncated if too long.
        Shows last N characters with ellipsis if truncated.
        """
        full = self.current_sentence_str
        if len(full) > self.max_sentence_length:
            return '...' + full[-(self.max_sentence_length - 3):]
        return full

    @property
    def word_count(self) -> int:
        return len(self._current_sentence)

    @property
    def letter_count(self) -> int:
        return self._total_letters

    @property
    def sentence_history(self) -> List[dict]:
        return self._sentence_history.copy()

    @property
    def last_action(self) -> str:
        # Fade out old actions
        if time.time() - self._last_action_time > 2.0:
            return ""
        return self._last_action

    def get_stats(self) -> dict:
        """Return typing statistics."""
        elapsed = time.time() - self._start_time
        return {
            'total_letters'     : self._total_letters,
            'total_words'       : self._total_words,
            'elapsed_seconds'   : elapsed,
            'letters_per_minute': self._total_letters / (elapsed / 60) if elapsed > 0 else 0,
            'current_word'      : self.current_word_str,
            'current_sentence'  : self.current_sentence_str,
            'history_count'     : len(self._sentence_history)
        }