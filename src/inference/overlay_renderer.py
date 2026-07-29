# ============================================================
# src/inference/overlay_renderer.py
# OpenCV UI Rendering — All Visual Overlays
# ============================================================

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class OverlayRenderer:
    """
    Handles all OpenCV drawing operations for the live demo.

    Renders:
    - Hand bounding box with color feedback
    - Prediction label + confidence bar
    - Stability progress bar
    - Top-5 predictions panel
    - Sentence builder display
    - Performance stats (FPS, latency)
    - Keyboard controls guide
    """

    def __init__(
        self,
        frame_width     : int = config.DISPLAY_WIDTH,
        frame_height    : int = config.DISPLAY_HEIGHT,
        panel_width     : int = config.PANEL_WIDTH
    ):
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.panel_width    = panel_width
        self.main_width     = frame_width - panel_width

        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ════════════════════════════════════════════════════════
    # MAIN RENDER METHOD
    # ════════════════════════════════════════════════════════

    def render(
        self,
        frame           : np.ndarray,
        prediction_result: Dict,
        buffer_status   : Dict,
        sentence_builder,
        hand_bbox       : Optional[Tuple] = None,
        hand_detected   : bool            = False,
        fps             : float           = 0.0,
        latency_ms      : float           = 0.0
    ) -> np.ndarray:
        """
        Main render method — composes the full display frame.

        Args:
            frame            : Raw BGR webcam frame
            prediction_result: Output from SignPredictor.predict()
            buffer_status    : Output from StabilityBuffer.update()
            sentence_builder : SentenceBuilder instance
            hand_bbox        : (x1,y1,x2,y2) hand bounding box
            hand_detected    : Whether MediaPipe found a hand
            fps              : Current inference FPS
            latency_ms       : Model latency in ms

        Returns:
            Complete annotated BGR frame for display
        """

        # Resize frame to main area
        display = cv2.resize(frame, (self.main_width, self.frame_height))

        # ── Draw layers on main frame ─────────────────────────
        display = self._draw_hand_box(
            display, hand_bbox, hand_detected, buffer_status
        )
        display = self._draw_prediction_overlay(
            display, prediction_result, buffer_status
        )
        display = self._draw_sentence_area(display, sentence_builder)
        display = self._draw_fps_counter(display, fps, latency_ms)

        # ── Build right side panel ────────────────────────────
        panel = self._build_info_panel(
            prediction_result, buffer_status,
            sentence_builder, hand_detected
        )

        # ── Combine main frame + panel ────────────────────────
        combined = np.hstack([display, panel])

        return combined

    # ════════════════════════════════════════════════════════
    # DRAWING METHODS
    # ════════════════════════════════════════════════════════

    def _draw_hand_box(
        self,
        frame       : np.ndarray,
        bbox        : Optional[Tuple],
        detected    : bool,
        status      : Dict
    ) -> np.ndarray:
        """Draw bounding box around detected hand."""

        if bbox is None:
            # No hand — draw guide text
            self._put_text_centered(
                frame,
                "Show your hand to the camera",
                y              = self.frame_height // 2,
                font_scale     = 0.9,
                color          = config.COLOR_YELLOW,
                thickness      = 2
            )
            return frame

        x1, y1, x2, y2 = bbox

        # Box color based on buffer state
        state = status.get('state', 'building')

        if state == 'confirmed':
            box_color = config.COLOR_GREEN
            thickness = 4
        elif state == 'cooldown':
            box_color = config.COLOR_CYAN
            thickness = 3
        elif state == 'low_confidence':
            box_color = config.COLOR_RED
            thickness = 2
        else:
            # Building — color shifts from red to green
            progress  = status.get('progress', 0)
            r = int(255 * (1 - progress))
            g = int(255 * progress)
            box_color = (0, g, r)
            thickness = 3

        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Draw corner decorations (professional look)
        corner_len = 20
        corners = [
            # Top-left
            [(x1, y1 + corner_len), (x1, y1), (x1 + corner_len, y1)],
            # Top-right
            [(x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len)],
            # Bottom-left
            [(x1, y2 - corner_len), (x1, y2), (x1 + corner_len, y2)],
            # Bottom-right
            [(x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], config.COLOR_WHITE, 3)

        return frame

    def _draw_prediction_overlay(
        self,
        frame   : np.ndarray,
        result  : Dict,
        status  : Dict
    ) -> np.ndarray:
        """Draw prediction label and confidence bar."""

        if not result:
            return frame

        pred        = result.get('top1_class', '?')
        conf        = result.get('top1_confidence', 0.0)
        state       = status.get('state', 'building')
        progress    = status.get('progress', 0.0)

        # ── Big prediction letter ─────────────────────────────
        label_color = (
            config.COLOR_GREEN  if state == 'confirmed'  else
            config.COLOR_CYAN   if state == 'cooldown'   else
            config.COLOR_RED    if state == 'low_confidence' else
            config.COLOR_WHITE
        )

        # Shadow
        cv2.putText(
            frame, pred,
            (32, 82), self.font, 3.5,
            config.COLOR_BLACK, 8
        )
        # Main text
        cv2.putText(
            frame, pred,
            (30, 80), self.font, 3.5,
            label_color, 5
        )

        # ── Confidence value ──────────────────────────────────
        conf_text = f"{conf*100:.1f}%"
        cv2.putText(
            frame, conf_text,
            (30, 120), self.font, 1.0,
            config.COLOR_LIGHT_GRAY, 2
        )

        # ── Confidence bar ────────────────────────────────────
        bar_x, bar_y    = 30, 135
        bar_w, bar_h    = 200, 12
        bar_fill        = int(bar_w * conf)
        bar_color       = (
            config.COLOR_GREEN  if conf >= 0.85  else
            config.COLOR_YELLOW if conf >= 0.70  else
            config.COLOR_RED
        )

        # Background
        cv2.rectangle(frame,
                       (bar_x, bar_y),
                       (bar_x + bar_w, bar_y + bar_h),
                       config.COLOR_DARK_GRAY, -1)
        # Fill
        if bar_fill > 0:
            cv2.rectangle(frame,
                           (bar_x, bar_y),
                           (bar_x + bar_fill, bar_y + bar_h),
                           bar_color, -1)
        # Border
        cv2.rectangle(frame,
                       (bar_x, bar_y),
                       (bar_x + bar_w, bar_y + bar_h),
                       config.COLOR_WHITE, 1)

        # Threshold line
        threshold_x = bar_x + int(bar_w * config.CONFIDENCE_THRESHOLD)
        cv2.line(frame,
                  (threshold_x, bar_y - 3),
                  (threshold_x, bar_y + bar_h + 3),
                  config.COLOR_YELLOW, 2)

        # ── Stability progress bar ────────────────────────────
        stab_y          = bar_y + bar_h + 20
        stab_label      = f"Stability: {status.get('buffer_fill', 0)}/{status.get('buffer_required', 5)}"

        cv2.putText(frame, stab_label,
                     (bar_x, stab_y - 2), self.font, 0.55,
                     config.COLOR_LIGHT_GRAY, 1)

        stab_fill = int(bar_w * progress)
        stab_color = config.COLOR_CYAN

        cv2.rectangle(frame,
                       (bar_x, stab_y + 5),
                       (bar_x + bar_w, stab_y + 15),
                       config.COLOR_DARK_GRAY, -1)
        if stab_fill > 0:
            cv2.rectangle(frame,
                           (bar_x, stab_y + 5),
                           (bar_x + stab_fill, stab_y + 15),
                           stab_color, -1)
        cv2.rectangle(frame,
                       (bar_x, stab_y + 5),
                       (bar_x + bar_w, stab_y + 15),
                       config.COLOR_WHITE, 1)

        # ── State badge ───────────────────────────────────────
        badge_map = {
            'confirmed'     : ('✓ CONFIRMED',  config.COLOR_GREEN),
            'cooldown'      : ('⏳ COOLDOWN',  config.COLOR_CYAN),
            'low_confidence': ('⚠ LOW CONF',   config.COLOR_RED),
            'nothing'       : ('∅ NO SIGN',    config.COLOR_LIGHT_GRAY),
            'building'      : ('⟳ BUILDING',   config.COLOR_YELLOW),
        }
        badge_text, badge_color = badge_map.get(
            state, ('UNKNOWN', config.COLOR_WHITE)
        )

        cv2.putText(frame, badge_text,
                     (bar_x, stab_y + 40), self.font, 0.7,
                     badge_color, 2)

        return frame

    def _draw_sentence_area(
        self,
        frame           : np.ndarray,
        sentence_builder
    ) -> np.ndarray:
        """Draw sentence builder output at bottom of frame."""

        h = self.frame_height

        # ── Background panel ──────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay,
                       (0, h - 130),
                       (self.main_width, h),
                       (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # ── Current word ──────────────────────────────────────
        word = sentence_builder.current_word_str
        word_display = f"Word: {word}_" if word else "Word: _"

        cv2.putText(frame, word_display,
                     (15, h - 95), self.font, 0.85,
                     config.COLOR_CYAN, 2)

        # ── Full sentence ─────────────────────────────────────
        sentence = sentence_builder.display_sentence
        cv2.putText(frame, "Sentence:",
                     (15, h - 60), self.font, 0.7,
                     config.COLOR_LIGHT_GRAY, 1)
        cv2.putText(frame, sentence,
                     (15, h - 30), self.font, 1.0,
                     config.COLOR_WHITE, 2)

        # ── Last action flash ─────────────────────────────────
        action = sentence_builder.last_action
        if action:
            cv2.putText(frame, action,
                         (self.main_width - 280, h - 95),
                         self.font, 0.65, config.COLOR_GREEN, 1)

        # ── Divider line ──────────────────────────────────────
        cv2.line(frame,
                  (0, h - 135),
                  (self.main_width, h - 135),
                  config.COLOR_DARK_GRAY, 1)

        return frame

    def _draw_fps_counter(
        self,
        frame       : np.ndarray,
        fps         : float,
        latency_ms  : float
    ) -> np.ndarray:
        """Draw FPS and latency in top-right corner."""

        fps_text     = f"FPS: {fps:.0f}"
        latency_text = f"{latency_ms:.0f}ms"

        x = self.main_width - 140
        cv2.putText(frame, fps_text,
                     (x, 30), self.font, 0.75,
                     config.COLOR_GREEN, 2)
        cv2.putText(frame, latency_text,
                     (x, 60), self.font, 0.65,
                     config.COLOR_LIGHT_GRAY, 1)

        return frame

    # ════════════════════════════════════════════════════════
    # RIGHT SIDE PANEL
    # ════════════════════════════════════════════════════════

    def _build_info_panel(
        self,
        result          : Dict,
        status          : Dict,
        sentence_builder,
        hand_detected   : bool
    ) -> np.ndarray:
        """Build the right-side info panel."""

        panel = np.zeros(
            (self.frame_height, self.panel_width, 3),
            dtype=np.uint8
        )
        panel[:] = (25, 25, 35)     # Dark background

        y = 20

        # ── Title ─────────────────────────────────────────────
        self._panel_text(panel, "ASL RECOGNITION", (10, y),
                          scale=0.65, color=config.COLOR_CYAN,
                          bold=True)
        y += 22
        self._panel_text(panel, "DenseNet-121", (10, y),
                          scale=0.5, color=config.COLOR_LIGHT_GRAY)
        y += 5
        cv2.line(panel, (5, y), (self.panel_width - 5, y),
                  config.COLOR_DARK_GRAY, 1)
        y += 20

        # ── Hand detection status ─────────────────────────────
        det_text  = "Hand: DETECTED ✓" if hand_detected else "Hand: NOT FOUND"
        det_color = config.COLOR_GREEN  if hand_detected else config.COLOR_RED
        self._panel_text(panel, det_text, (10, y),
                          scale=0.55, color=det_color)
        y += 30

        # ── Top-K predictions ─────────────────────────────────
        cv2.line(panel, (5, y), (self.panel_width - 5, y),
                  config.COLOR_DARK_GRAY, 1)
        y += 15
        self._panel_text(panel, "Top-5 Predictions:", (10, y),
                          scale=0.55, color=config.COLOR_YELLOW)
        y += 20

        top_k = result.get('top_k', [])
        for rank, (cls, prob) in enumerate(top_k):
            is_top = (rank == 0)

            # Bar
            bar_w   = self.panel_width - 80
            fill    = int(bar_w * prob)
            bar_y   = y + 8

            bar_color = config.COLOR_GREEN if is_top else (60, 60, 60)
            cv2.rectangle(panel,
                           (60, bar_y),
                           (60 + bar_w, bar_y + 14),
                           (40, 40, 40), -1)
            if fill > 0:
                cv2.rectangle(panel,
                               (60, bar_y),
                               (60 + fill, bar_y + 14),
                               bar_color, -1)

            # Label
            label = f"#{rank+1} {cls}"
            conf  = f"{prob*100:.1f}%"
            color = config.COLOR_WHITE if is_top else config.COLOR_LIGHT_GRAY
            scale = 0.60 if is_top else 0.50

            self._panel_text(panel, label, (5, y + 16), scale=scale, color=color)
            self._panel_text(panel, conf,
                              (self.panel_width - 55, y + 16),
                              scale=scale, color=color)
            y += 28

        # ── Buffer visualization ──────────────────────────────
        y += 5
        cv2.line(panel, (5, y), (self.panel_width - 5, y),
                  config.COLOR_DARK_GRAY, 1)
        y += 15
        self._panel_text(panel, "Stability Buffer:", (10, y),
                          scale=0.55, color=config.COLOR_YELLOW)
        y += 22

        # Draw buffer slots
        buf_content = status.get('buffer_content', [])
        required    = status.get('buffer_required', 5)
        slot_w      = (self.panel_width - 20) // required

        for i in range(required):
            sx = 10 + i * slot_w
            if i < len(buf_content):
                # Filled slot
                cv2.rectangle(panel,
                               (sx, y), (sx + slot_w - 4, y + 28),
                               config.COLOR_GREEN, -1)
                self._panel_text(panel, buf_content[i],
                                  (sx + 6, y + 20),
                                  scale=0.65, color=config.COLOR_BLACK,
                                  bold=True)
            else:
                # Empty slot
                cv2.rectangle(panel,
                               (sx, y), (sx + slot_w - 4, y + 28),
                               config.COLOR_DARK_GRAY, -1)
                cv2.rectangle(panel,
                               (sx, y), (sx + slot_w - 4, y + 28),
                               (80, 80, 80), 1)
        y += 45

        # ── Stats ─────────────────────────────────────────────
        cv2.line(panel, (5, y), (self.panel_width - 5, y),
                  config.COLOR_DARK_GRAY, 1)
        y += 15

        stats = sentence_builder.get_stats()
        stat_lines = [
            ("Letters typed:", str(stats['total_letters'])),
            ("Words typed:",   str(stats['total_words'])),
            ("Letters/min:",   f"{stats['letters_per_minute']:.1f}"),
        ]
        for label, val in stat_lines:
            self._panel_text(panel, label, (10, y),
                              scale=0.50, color=config.COLOR_LIGHT_GRAY)
            self._panel_text(panel, val,
                              (self.panel_width - 50, y),
                              scale=0.50, color=config.COLOR_WHITE)
            y += 22

        # ── Controls guide ────────────────────────────────────
        y = self.frame_height - 185
        cv2.line(panel, (5, y), (self.panel_width - 5, y),
                  config.COLOR_DARK_GRAY, 1)
        y += 12
        self._panel_text(panel, "⌨  CONTROLS:", (10, y),
                          scale=0.52, color=config.COLOR_YELLOW)
        y += 18

        controls = [
            ("SPACE",  "Add space / new word"),
            ("BKSP",   "Delete last letter"),
            ("ENTER",  "Save sentence"),
            ("C",      "Clear current word"),
            ("Shift+C","Clear everything"),
            ("P",      "Screenshot"),
            ("R",      "Start/stop recording"),
            ("Q / ESC","Quit demo"),
        ]
        for key, action in controls:
            self._panel_text(panel, f"  {key:<9}", (10, y),
                              scale=0.45, color=config.COLOR_CYAN)
            self._panel_text(panel, action, (75, y),
                              scale=0.45, color=config.COLOR_LIGHT_GRAY)
            y += 17

        return panel

    # ── UTILITY METHODS ──────────────────────────────────────
    def _panel_text(
        self,
        panel   : np.ndarray,
        text    : str,
        pos     : Tuple[int, int],
        scale   : float = 0.55,
        color   : Tuple  = (255, 255, 255),
        bold    : bool   = False
    ) -> None:
        """Helper for drawing text on the info panel."""
        thickness = 2 if bold else 1
        cv2.putText(panel, text, pos, self.font, scale, color, thickness)

    def _put_text_centered(
        self,
        frame   : np.ndarray,
        text    : str,
        y       : int,
        font_scale : float = 1.0,
        color   : Tuple    = (255, 255, 255),
        thickness: int     = 2
    ) -> None:
        """Center text horizontally on frame."""
        text_size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
        x = (self.main_width - text_size[0]) // 2
        cv2.putText(frame, text, (x, y), self.font,
                     font_scale, color, thickness)

    def draw_loading_screen(self, message: str = "Loading model...") -> np.ndarray:
        """Show loading screen while model is being initialized."""
        screen = np.zeros(
            (self.frame_height, self.frame_width, 3),
            dtype=np.uint8
        )
        screen[:] = (20, 20, 30)

        lines = [
            ("ASL Sign Language Recognition", 0.9, config.COLOR_CYAN),
            ("DenseNet-121", 0.7, config.COLOR_LIGHT_GRAY),
            ("", 0.5, config.COLOR_WHITE),
            (message, 0.75, config.COLOR_YELLOW),
        ]

        y = self.frame_height // 2 - 80
        for text, scale, color in lines:
            if text:
                size = cv2.getTextSize(text, self.font, scale, 2)[0]
                x    = (self.frame_width - size[0]) // 2
                cv2.putText(screen, text, (x, y), self.font, scale, color, 2)
            y += 45

        return screen