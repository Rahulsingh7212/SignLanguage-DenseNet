# ============================================================
# test_single_image.py
# Test the model on a single image file (no webcam needed)
# ============================================================

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("D:/SignLanguage-DenseNet")

import config
from src.inference.predictor        import SignPredictor
from src.inference.stability_buffer import StabilityBuffer
from src.hand_detector              import HandDetector


def test_on_image(image_path: str) -> None:
    """
    Run full pipeline on a single image and display results.
    """
    path = Path(image_path)

    if not path.exists():
        print(f"❌ Image not found: {path}")
        return

    print(f"\n🖼️  Testing on: {path.name}")

    # Load image
    image = cv2.imread(str(path))
    if image is None:
        print("❌ Cannot read image!")
        return

    print(f"   Shape: {image.shape}")

    # ── Hand Detection ────────────────────────────────────────
    detector = HandDetector(
        min_detection_confidence = config.MEDIAPIPE_CONFIDENCE,
        padding                  = config.HAND_CROP_PADDING,
        fallback_to_full         = True
    )
    hand_crop, detected = detector.detect_and_crop(image)
    print(f"   Hand detected: {detected}")

    # ── Model Prediction ──────────────────────────────────────
    predictor = SignPredictor(
        model_path  = config.INFERENCE_MODEL_PATH,
        image_size  = config.INFERENCE_IMAGE_SIZE,
        classes     = config.CLASSES
    )
    result = predictor.predict(hand_crop)

    # ── Display Results ───────────────────────────────────────
    print(f"\n📊 Prediction Results:")
    print(f"   Top-1: '{result['top1_class']}' ({result['top1_confidence']*100:.1f}%)")
    print(f"   Inference time: {result['inference_ms']:.1f}ms")
    print(f"\n   Top-5 Predictions:")
    for rank, (cls, prob) in enumerate(result['top_k']):
        bar = '█' * int(prob * 30)
        print(f"     #{rank+1} {cls:<10} {prob*100:>6.2f}%  {bar}")

    # ── Visualize ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Prediction: '{result['top1_class']}' "
        f"({result['top1_confidence']*100:.1f}% confidence)",
        fontsize=16, fontweight='bold',
        color='green' if result['top1_confidence'] > 0.7 else 'red'
    )

    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')

    # Hand crop
    axes[1].imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f'Hand Crop\n{"Detected ✅" if detected else "Full Image (Fallback) ⚠️"}',
        fontweight='bold', color='green' if detected else 'orange'
    )
    axes[1].axis('off')

    # Top-5 bar chart
    top5_classes = [c for c, _ in result['top_k']]
    top5_probs   = [p * 100 for _, p in result['top_k']]
    colors       = ['#2ecc71' if i == 0 else '#3498db'
                    for i in range(len(top5_classes))]

    axes[2].barh(top5_classes[::-1], top5_probs[::-1],
                  color=colors[::-1], edgecolor='black')
    axes[2].set_xlabel('Confidence (%)')
    axes[2].set_title('Top-5 Predictions', fontweight='bold')
    axes[2].axvline(x=70, color='red', linestyle='--',
                     alpha=0.7, label='70% threshold')
    axes[2].legend()
    axes[2].set_xlim([0, 110])

    for i, (cls, prob) in enumerate(
        zip(top5_classes[::-1], top5_probs[::-1])
    ):
        axes[2].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(
        config.EVALUATION_DIR / f"test_image_{path.stem}.png",
        dpi=150, bbox_inches='tight'
    )
    plt.show()

    detector.close()


def test_on_all_classes(n_per_class: int = 3) -> None:
    """Test model on sample images from each class."""
    print("\n🧪 Testing on sample images from ALL classes...")

    predictor = SignPredictor(model_path=config.INFERENCE_MODEL_PATH)
    detector  = HandDetector(
        min_detection_confidence = config.MEDIAPIPE_CONFIDENCE,
        padding                  = config.HAND_CROP_PADDING,
        fallback_to_full         = True
    )

    results = {}
    for class_name in config.CLASSES:
        class_dir = config.TEST_DIR / class_name
        if not class_dir.exists():
            continue

        img_files = sorted(class_dir.iterdir())[:n_per_class]
        class_results = []

        for img_path in img_files:
            img  = cv2.imread(str(img_path))
            if img is None:
                continue
            crop, _ = detector.detect_and_crop(img)
            result  = predictor.predict(crop)
            correct = result['top1_class'] == class_name
            class_results.append({
                'correct'   : correct,
                'predicted' : result['top1_class'],
                'confidence': result['top1_confidence']
            })

        acc = sum(r['correct'] for r in class_results) / len(class_results)
        results[class_name] = acc
        status = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.5 else "❌"
        print(f"   {status} {class_name:<10} {acc*100:.0f}% "
              f"({sum(r['correct'] for r in class_results)}/{len(class_results)})")

    overall = np.mean(list(results.values()))
    print(f"\n   Overall quick-test accuracy: {overall*100:.1f}%")

    detector.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific image
        test_on_image(sys.argv[1])
    else:
        # Test on sample images from all classes
        test_on_all_classes(n_per_class=5)