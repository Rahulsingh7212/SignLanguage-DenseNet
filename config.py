# ============================================================
# config.py
# Central configuration for the entire project
# ============================================================

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
RAW_DATA_DIR    = PROJECT_ROOT / "datasets" / "asl_alphabet" / "asl_alphabet_train" / "asl_alphabet_train"
PROCESSED_DIR   = PROJECT_ROOT / "processed_data"
HAND_CROPS_DIR  = PROCESSED_DIR / "hand_crops"
TRAIN_DIR       = PROCESSED_DIR / "train"
VAL_DIR         = PROCESSED_DIR / "val"
TEST_DIR        = PROCESSED_DIR / "test"
PLOTS_DIR       = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR      = PROJECT_ROOT / "models"
LOGS_DIR        = PROJECT_ROOT / "logs"

# ── Dataset ──────────────────────────────────────────────────
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]
NUM_CLASSES     = len(CLASSES)          # 29
CLASS_TO_IDX    = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS    = {i: c for i, c in enumerate(CLASSES)}

# ── Image Settings ───────────────────────────────────────────
IMAGE_SIZE      = 224                   # DenseNet standard input
IMAGE_CHANNELS  = 3                     # RGB
BATCH_SIZE      = 32
NUM_WORKERS     = 4

# ── Split Ratios ─────────────────────────────────────────────
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
RANDOM_SEED     = 42

# ── MediaPipe Settings ───────────────────────────────────────
MEDIAPIPE_CONFIDENCE    = 0.3           # Lower = detect more hands
HAND_CROP_PADDING       = 0.25         # 25% padding around hand bbox
FALLBACK_TO_FULL_IMAGE  = True         # Use full image if no hand found

# ── Normalization ────────────────────────────────────────────
NORMALIZE_MEAN  = [0.485, 0.456, 0.406]   # ImageNet mean
NORMALIZE_STD   = [0.229, 0.224, 0.225]   # ImageNet std

# ── Augmentation Settings ────────────────────────────────────
AUG_HFLIP_PROB      = 0.5
AUG_ROTATION_LIMIT  = 15               # ±15 degrees
AUG_ZOOM_MIN        = 0.9
AUG_ZOOM_MAX        = 1.1
AUG_BRIGHTNESS      = 0.2
AUG_CONTRAST        = 0.2
AUG_BLUR_PROB       = 0.1

# ── Processing ───────────────────────────────────────────────
MAX_IMAGES_PER_CLASS    = None         # None = use all images
SAVE_HAND_CROPS         = True         # Save intermediate crops
SKIP_EXISTING           = True         # Resume if interrupted

print("✅ Config loaded successfully!")
print(f"   Classes: {NUM_CLASSES}")
print(f"   Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")