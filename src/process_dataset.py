# ============================================================
# src/process_dataset.py
# MAIN SCRIPT: Run full preprocessing pipeline on ASL dataset
# ============================================================

import cv2
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import config
from src.hand_detector import HandDetector
from src.preprocessor  import ImagePreprocessor
from src.dataset_builder import DatasetSplitter


def process_single_image(
    img_path    : Path,
    detector    : HandDetector,
    preprocessor: ImagePreprocessor,
    output_path : Path,
    skip_existing: bool = True
) -> dict:
    """
    Process one image through the full pipeline:
    Load → Detect Hand → Crop → Resize → Save
    """
    result = {
        'success'       : False,
        'hand_detected' : False,
        'error'         : None
    }

    # Skip if already processed
    if skip_existing and output_path.exists():
        result['success'] = True
        result['skipped'] = True
        return result

    try:
        # Step 1: Load image
        image = cv2.imread(str(img_path))
        if image is None:
            result['error'] = "Cannot read image"
            return result

        # Step 2: Detect & crop hand
        hand_crop, detected = detector.detect_and_crop(image)
        result['hand_detected'] = detected

        if hand_crop is None:
            result['error'] = "No hand detected and fallback disabled"
            return result

        # Step 3: Resize + convert to RGB
        processed = preprocessor.process(hand_crop)

        # Step 4: Save
        success = preprocessor.save_image(processed, output_path)
        result['success'] = success

    except Exception as e:
        result['error'] = str(e)

    return result


def run_full_pipeline(
    max_per_class   : int  = None,
    skip_existing   : bool = True,
    verbose         : bool = True
):
    """
    Process ALL images in the ASL dataset.
    For each class folder → detect hand → crop → resize → save.
    """

    print("\n" + "="*65)
    print("  🚀 STARTING FULL PREPROCESSING PIPELINE")
    print("="*65)
    print(f"  Source:  {config.RAW_DATA_DIR}")
    print(f"  Output:  {config.HAND_CROPS_DIR}")
    print(f"  Size:    {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"  Padding: {config.HAND_CROP_PADDING*100:.0f}%")

    start_time = time.time()

    # Initialize modules
    detector     = HandDetector(
        min_detection_confidence = config.MEDIAPIPE_CONFIDENCE,
        padding                  = config.HAND_CROP_PADDING,
        fallback_to_full         = config.FALLBACK_TO_FULL_IMAGE
    )
    preprocessor = ImagePreprocessor(target_size=config.IMAGE_SIZE)

    # Global stats
    global_stats = {
        'total'     : 0,
        'success'   : 0,
        'failed'    : 0,
        'skipped'   : 0,
        'hand_detected' : 0
    }

    # ── Process each class ───────────────────────────────────
    classes = sorted([d.name for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()])

    for class_name in classes:
        class_input_dir  = config.RAW_DATA_DIR / class_name
        class_output_dir = config.HAND_CROPS_DIR / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = sorted([
            f for f in class_input_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        # Limit if specified
        if max_per_class:
            image_files = image_files[:max_per_class]

        class_stats = {'success': 0, 'failed': 0, 'hand_detected': 0}

        # Progress bar per class
        with tqdm(
            image_files,
            desc=f"  {class_name:<8}",
            unit="img",
            leave=True
        ) as pbar:
            for img_path in pbar:
                output_path = class_output_dir / f"{img_path.stem}.jpg"

                result = process_single_image(
                    img_path, detector, preprocessor,
                    output_path, skip_existing
                )

                global_stats['total'] += 1

                if result.get('skipped'):
                    global_stats['skipped'] += 1
                    class_stats['success']  += 1
                elif result['success']:
                    global_stats['success'] += 1
                    class_stats['success']  += 1
                    if result['hand_detected']:
                        global_stats['hand_detected'] += 1
                        class_stats['hand_detected']  += 1
                else:
                    global_stats['failed'] += 1
                    class_stats['failed']  += 1

                # Update progress bar
                pbar.set_postfix({
                    'ok'    : class_stats['success'],
                    'err'   : class_stats['failed'],
                    'hand'  : class_stats['hand_detected']
                })

    # ── Detection Stats ──────────────────────────────────────
    det_stats = detector.get_stats()
    detector.close()

    elapsed = time.time() - start_time

    # ── Print Summary ────────────────────────────────────────
    print("\n" + "="*65)
    print("  ✅ PREPROCESSING COMPLETE")
    print("="*65)
    print(f"  Total images:     {global_stats['total']:,}")
    print(f"  Successful:       {global_stats['success']:,}")
    print(f"  Failed:           {global_stats['failed']:,}")
    print(f"  Skipped:          {global_stats['skipped']:,}")
    print(f"  Hands detected:   {det_stats.get('detected', 0):,} "
          f"({det_stats.get('detection_rate', '?')})")
    print(f"  Fallback (full):  {det_stats.get('fallback', 0):,} "
          f"({det_stats.get('fallback_rate', '?')})")
    print(f"  Time elapsed:     {elapsed/60:.1f} minutes")
    print("="*65)

    return global_stats


if __name__ == "__main__":
    # ── STEP 1: Process all images ───────────────────────────
    print("\n⚙️  STEP 1: Hand Detection + Preprocessing")
    stats = run_full_pipeline(
        max_per_class = None,        # None = all images
        skip_existing = True,        # Resume if interrupted
        verbose       = True
    )

    # ── STEP 2: Split into train/val/test ────────────────────
    print("\n⚙️  STEP 2: Stratified Train/Val/Test Split")
    splitter = DatasetSplitter()
    split_stats = splitter.split(verbose=True)

    print("\n🎉 Stage 2 Pipeline Complete!")
    print("   Ready for Stage 3: DenseNet Model Building")