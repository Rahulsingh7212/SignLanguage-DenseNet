# ============================================================
# run_demo.py
# One-click launcher for the ASL Recognition Demo
# ============================================================

import sys
from pathlib import Path

sys.path.append("D:/SignLanguage-DenseNet")

import config
from src.inference.webcam_demo import WebcamDemo


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     ASL SIGN LANGUAGE RECOGNITION — LIVE DEMO           ║
║     DenseNet-121 + MediaPipe + OpenCV                   ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Verify model exists
    model_path = config.INFERENCE_MODEL_PATH

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please complete Stage 3 training first.")
        print(f"   Expected: {model_path}")
        sys.exit(1)

    print(f"✅ Model found: {model_path.name}")
    print(f"   Webcam index:         {config.WEBCAM_INDEX}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"   Stability frames:     {config.STABILITY_FRAMES}")
    print(f"   Display:              {config.DISPLAY_WIDTH}×{config.DISPLAY_HEIGHT}")

    # Launch demo
    demo = WebcamDemo(
        model_path   = model_path,
        webcam_index = config.WEBCAM_INDEX,
        record_video = config.RECORD_DEMO
    )

    demo.run()


if __name__ == "__main__":
    main()