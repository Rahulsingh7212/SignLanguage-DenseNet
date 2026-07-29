# ============================================================
# src/inference/webcam_demo.py
# Main Real-Time Webcam Demo Loop
# ============================================================

import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.inference.predictor        import SignPredictor
from src.inference.stability_buffer import StabilityBuffer
from src.inference.sentence_builder import SentenceBuilder
from src.inference.overlay_renderer import OverlayRenderer


class WebcamDemo:
    """
    Full real-time ASL recognition demo.

    Architecture per frame:
        Webcam → MediaPipe hand detect → crop ROI
               → DenseNet predict → stability buffer
               → sentence builder → overlay render → display
    """

    def __init__(
        self,
        model_path      : Path  = config.INFERENCE_MODEL_PATH,
        webcam_index    : int   = config.WEBCAM_INDEX,
        record_video    : bool  = config.RECORD_DEMO
    ):
        self.model_path     = model_path
        self.webcam_index   = webcam_index
        self.record_video   = record_video

        # Components (initialized in setup())
        self.predictor      : Optional[SignPredictor]    = None
        self.buffer         : Optional[StabilityBuffer]  = None
        self.sentence       : Optional[SentenceBuilder]  = None
        self.renderer       : Optional[OverlayRenderer]  = None
        self.cap            : Optional[cv2.VideoCapture] = None
        self.writer         : Optional[cv2.VideoWriter]  = None

        # MediaPipe
        self.mp_hands       = mp.solutions.hands
        self.mp_drawing     = mp.solutions.drawing_utils
        self.hands_detector = None

        # State
        self.is_running     = False
        self.frame_count    = 0
        self.fps_counter    = FPSCounter()

        # Last prediction (for UI continuity)
        self._last_result   = {}
        self._last_status   = {}
        self._last_bbox     = None
        self._hand_detected = False

    # ── SETUP ────────────────────────────────────────────────
    def setup(self) -> bool:
        """Initialize all components."""
        print("\n" + "="*60)
        print("  🎬 ASL RECOGNITION DEMO — SETUP")
        print("="*60)

        # ── Renderer (for loading screen) ─────────────────────
        self.renderer = OverlayRenderer(
            frame_width  = config.DISPLAY_WIDTH,
            frame_height = config.DISPLAY_HEIGHT,
            panel_width  = config.PANEL_WIDTH
        )

        # ── Webcam ───────────────────────────────────────────
        print(f"\n  📷 Opening webcam (index={self.webcam_index})...")
        self.cap = cv2.VideoCapture(self.webcam_index)

        if not self.cap.isOpened():
            print("  ❌ Cannot open webcam!")
            return False

        # Configure webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.WEBCAM_FPS)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_f = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"  ✅ Webcam opened: {actual_w}×{actual_h} @ {actual_f}fps")

        # ── Show loading screen ───────────────────────────────
        loading = self.renderer.draw_loading_screen("Loading DenseNet model...")
        cv2.imshow("ASL Sign Language Recognition", loading)
        cv2.waitKey(1)

        # ── Load DenseNet Model ───────────────────────────────
        try:
            self.predictor = SignPredictor(
                model_path  = self.model_path,
                image_size  = config.INFERENCE_IMAGE_SIZE,
                classes     = config.CLASSES,
                top_k       = 5
            )
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            return False

        # ── MediaPipe Hands ───────────────────────────────────
        print("\n  🤚 Initializing MediaPipe Hands...")
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode        = False,   # Video mode (faster)
            max_num_hands            = 1,
            min_detection_confidence = config.MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence  = 0.5
        )
        print("  ✅ MediaPipe ready!")

        # ── Stability Buffer ──────────────────────────────────
        self.buffer = StabilityBuffer(
            required_frames      = config.STABILITY_FRAMES,
            confidence_threshold = config.CONFIDENCE_THRESHOLD,
            cooldown_frames      = config.COOLDOWN_FRAMES
        )

        # ── Sentence Builder ──────────────────────────────────
        self.sentence = SentenceBuilder(
            max_sentence_length = config.MAX_SENTENCE_LENGTH,
            max_word_length     = config.MAX_WORD_LENGTH
        )

        # ── Video Recorder ────────────────────────────────────
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(
                str(config.DEMO_VIDEO_PATH),
                fourcc,
                config.WEBCAM_FPS,
                (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            )
            print(f"  🔴 Recording to: {config.DEMO_VIDEO_PATH}")

        print("\n  ✅ All components initialized!")
        print("  🚀 Starting demo...\n")
        return True

    # ── MAIN LOOP ────────────────────────────────────────────
    def run(self) -> None:
        """Main real-time processing loop."""

        if not self.setup():
            self.cleanup()
            return

        self.is_running = True

        print("="*60)
        print("  🎯 DEMO RUNNING — Controls:")
        print("     SPACE   → Add space/word break")
        print("     BKSP    → Delete last letter")
        print("     ENTER   → Save sentence to history")
        print("     C       → Clear current word")
        print("     P       → Take screenshot")
        print("     Q/ESC   → Quit")
        print("="*60 + "\n")

        try:
            while self.is_running:
                # ── Read frame ───────────────────────────────
                ret, frame = self.cap.read()
                if not ret:
                    print("  ⚠️  Frame read failed — retrying...")
                    time.sleep(0.05)
                    continue

                self.frame_count += 1

                # Mirror flip (natural webcam feel)
                frame = cv2.flip(frame, 1)

                # ── Process frame ────────────────────────────
                self._process_frame(frame)

                # ── Keyboard input ───────────────────────────
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)

        except KeyboardInterrupt:
            print("\n  ⚠️  Interrupted by user")

        finally:
            self.cleanup()

    # ── FRAME PROCESSING ─────────────────────────────────────
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process one webcam frame through full pipeline."""

        # ── Step 1: MediaPipe Hand Detection ─────────────────
        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results  = self.hands_detector.process(rgb_frame)

        hand_crop, bbox, hand_detected = self._extract_hand(
            frame, mp_results
        )

        # ── Step 2: DenseNet Inference ────────────────────────
        if hand_crop is not None:
            result = self.predictor.predict(hand_crop)
            self._last_result   = result
            self._last_bbox     = bbox
            self._hand_detected = hand_detected

            # ── Step 3: Stability Buffer ──────────────────────
            confirmed, status = self.buffer.update(
                prediction = result['top1_class'],
                confidence = result['top1_confidence']
            )
            self._last_status = status

            # ── Step 4: Sentence Builder ──────────────────────
            if confirmed is not None:
                self.sentence.add_letter(confirmed)
                print(f"  ✅ Confirmed: '{confirmed}' → "
                      f"'{self.sentence.current_sentence_str}'")

        # ── Step 5: Render & Display ──────────────────────────
        fps         = self.fps_counter.tick()
        latency_ms  = self.predictor.get_avg_latency()

        display_frame = self.renderer.render(
            frame             = frame,
            prediction_result = self._last_result,
            buffer_status     = self._last_status,
            sentence_builder  = self.sentence,
            hand_bbox         = self._last_bbox,
            hand_detected     = self._hand_detected,
            fps               = fps,
            latency_ms        = latency_ms
        )

        # ── Step 6: Show & Record ─────────────────────────────
        cv2.imshow("ASL Sign Language Recognition", display_frame)

        if self.writer is not None:
            self.writer.write(display_frame)

    # ── HAND EXTRACTION ──────────────────────────────────────
    def _extract_hand(
        self,
        frame       : np.ndarray,
        mp_results  : object
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple], bool]:
        """
        Extract hand ROI from frame using MediaPipe landmarks.

        Returns:
            (hand_crop, bbox_tuple, hand_detected)
        """
        h, w = frame.shape[:2]

        if mp_results.multi_hand_landmarks:
            landmarks = mp_results.multi_hand_landmarks[0]

            # Get bounding box from landmarks
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]

            pad = config.HAND_CROP_PADDING

            x_min = max(0.0, min(x_coords) - pad)
            x_max = min(1.0, max(x_coords) + pad)
            y_min = max(0.0, min(y_coords) - pad)
            y_max = min(1.0, max(y_coords) + pad)

            x1 = int(x_min * w)
            y1 = int(y_min * h)
            x2 = int(x_max * w)
            y2 = int(y_max * h)

            # Scale bbox to display size
            scale_x = (config.DISPLAY_WIDTH - config.PANEL_WIDTH) / w
            scale_y = config.DISPLAY_HEIGHT / h
            disp_bbox = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                return crop, disp_bbox, True

        # Fallback: use full frame
        if config.FALLBACK_TO_FULL_IMAGE:
            return frame, None, False

        return None, None, False

    # ── KEYBOARD HANDLER ─────────────────────────────────────
    def _handle_key(self, key: int, frame: np.ndarray) -> None:
        """Process keyboard input."""

        if key in [ord('q'), 27]:       # Q or ESC
            print("\n  👋 Quitting demo...")
            self.is_running = False

        elif key == 32:                  # SPACE
            self.sentence.add_space()

        elif key == 8:                   # BACKSPACE
            self.sentence.backspace()

        elif key == 13:                  # ENTER
            sent = self.sentence.commit_sentence()
            if sent:
                print(f"  💾 Saved: '{sent}'")

        elif key == ord('c'):            # Clear word
            self.sentence.clear_word()

        elif key == ord('C'):            # Clear all
            self.sentence.clear_all()
            self.buffer.reset()

        elif key == config.SCREENSHOT_KEY:  # P — Screenshot
            self._take_screenshot(frame)

        elif key == ord('r'):            # R — Toggle recording
            self._toggle_recording()

        elif key == ord('b'):            # B — Reset buffer
            self.buffer.reset()
            print("  🔄 Buffer reset")

    def _take_screenshot(self, frame: np.ndarray) -> None:
        """Save current frame as screenshot."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.DEMO_SCREENSHOTS_DIR / f"screenshot_{timestamp}.png"
        cv2.imwrite(str(path), frame)
        print(f"  📸 Screenshot saved: {path.name}")

    def _toggle_recording(self) -> None:
        """Start or stop video recording."""
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(
                str(config.DEMO_VIDEO_PATH),
                fourcc,
                config.WEBCAM_FPS,
                (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            )
            print("  🔴 Recording STARTED")
        else:
            self.writer.release()
            self.writer = None
            print(f"  ⏹  Recording STOPPED → {config.DEMO_VIDEO_PATH}")

    # ── CLEANUP ──────────────────────────────────────────────
    def cleanup(self) -> None:
        """Release all resources."""
        print("\n  🧹 Cleaning up...")

        if self.cap is not None:
            self.cap.release()

        if self.writer is not None:
            self.writer.release()

        if self.hands_detector is not None:
            self.hands_detector.close()

        cv2.destroyAllWindows()

        # Print final stats
        if self.sentence is not None:
            stats = self.sentence.get_stats()
            print("\n" + "="*50)
            print("  📊 SESSION SUMMARY")
            print("="*50)
            print(f"  Total frames:    {self.frame_count:,}")
            print(f"  Letters typed:   {stats['total_letters']}")
            print(f"  Words typed:     {stats['total_words']}")
            print(f"  Sentences saved: {stats['history_count']}")

            if self.predictor:
                print(f"  Avg FPS:         {self.predictor.get_avg_fps():.1f}")
                print(f"  Avg latency:     {self.predictor.get_avg_latency():.1f}ms")

            # Print saved sentences
            history = self.sentence.sentence_history
            if history:
                print(f"\n  💬 Saved Sentences:")
                for i, entry in enumerate(history):
                    print(f"     {i+1}. {entry['text']}")

        print("="*50)
        print("  ✅ Demo closed cleanly.\n")


# ════════════════════════════════════════════════════════════
# FPS COUNTER UTILITY
# ════════════════════════════════════════════════════════════

class FPSCounter:
    """Smooth FPS calculation using rolling average."""

    def __init__(self, window: int = 30):
        self.timestamps = []
        self.window     = window

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)

        # Keep only last N timestamps
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0