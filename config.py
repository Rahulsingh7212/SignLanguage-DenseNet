# ============================================================
# config.py  — UPDATED for Stage 3
# ============================================================

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).resolve().parent
RAW_DATA_DIR        = PROJECT_ROOT / "datasets" / "asl_alphabet" / "asl_alphabet_train" / "asl_alphabet_train"
PROCESSED_DIR       = PROJECT_ROOT / "processed_data"
HAND_CROPS_DIR      = PROCESSED_DIR / "hand_crops"
TRAIN_DIR           = PROCESSED_DIR / "train"
VAL_DIR             = PROCESSED_DIR / "val"
TEST_DIR            = PROCESSED_DIR / "test"
PLOTS_DIR           = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR          = PROJECT_ROOT / "models"
CHECKPOINTS_DIR     = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR    = MODELS_DIR / "final"
TRAINING_LOGS_DIR   = PROJECT_ROOT / "outputs" / "training_logs"
EVALUATION_DIR      = PROJECT_ROOT / "outputs" / "evaluation"
LOGS_DIR            = PROJECT_ROOT / "logs"

# ── Create all dirs ──────────────────────────────────────────
for d in [PLOTS_DIR, CHECKPOINTS_DIR, FINAL_MODELS_DIR,
          TRAINING_LOGS_DIR, EVALUATION_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]
NUM_CLASSES         = len(CLASSES)
CLASS_TO_IDX        = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS        = {i: c for i, c in enumerate(CLASSES)}

# ── Image Settings ───────────────────────────────────────────
IMAGE_SIZE          = 224
IMAGE_CHANNELS      = 3
RANDOM_SEED         = 42

# ── Split Ratios ─────────────────────────────────────────────
TRAIN_RATIO         = 0.80
VAL_RATIO           = 0.10
TEST_RATIO          = 0.10

# ── Normalization ─────────────────────────────────────────────
NORMALIZE_MEAN      = [0.485, 0.456, 0.406]
NORMALIZE_STD       = [0.229, 0.224, 0.225]

# ── MediaPipe ────────────────────────────────────────────────
MEDIAPIPE_CONFIDENCE    = 0.3
HAND_CROP_PADDING       = 0.25
FALLBACK_TO_FULL_IMAGE  = True

# ── Augmentation ─────────────────────────────────────────────
AUG_HFLIP_PROB      = 0.5
AUG_ROTATION_LIMIT  = 15
AUG_ZOOM_MIN        = 0.9
AUG_ZOOM_MAX        = 1.1
AUG_BRIGHTNESS      = 0.2
AUG_CONTRAST        = 0.2
AUG_BLUR_PROB       = 0.1

# ════════════════════════════════════════════════════════════
# STAGE 3: MODEL TRAINING CONFIGURATION
# ════════════════════════════════════════════════════════════

# ── Model ────────────────────────────────────────────────────
MODEL_NAME          = "DenseNet121"
DENSENET_WEIGHTS    = "imagenet"        # Pre-trained weights
DROPOUT_RATE        = 0.4
DENSE_UNITS         = 512

# ── Batch & DataLoader ───────────────────────────────────────
BATCH_SIZE          = 32
NUM_WORKERS         = 4

# ── Phase 1: Head Training ───────────────────────────────────
PHASE1_EPOCHS       = 10
PHASE1_LR           = 0.001
PHASE1_MODEL_PATH   = FINAL_MODELS_DIR / "densenet_phase1.keras"

# ── Phase 2: Fine-tuning ─────────────────────────────────────
PHASE2_EPOCHS       = 15
PHASE2_LR           = 1e-5
PHASE2_UNFREEZE_FROM = 310             # Unfreeze from layer 310+
PHASE2_MODEL_PATH   = FINAL_MODELS_DIR / "densenet_final.keras"

# ── Callbacks ────────────────────────────────────────────────
CHECKPOINT_PATH     = CHECKPOINTS_DIR / "best_model.keras"
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.3
REDUCE_LR_MIN       = 1e-7

# ── Evaluation ───────────────────────────────────────────────
CONFUSION_MATRIX_PATH       = EVALUATION_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH  = EVALUATION_DIR / "classification_report.txt"
PER_CLASS_F1_PATH           = EVALUATION_DIR / "per_class_f1.png"

print("✅ Config loaded successfully!")
print(f"   Model:       {MODEL_NAME}")
print(f"   Classes:     {NUM_CLASSES}")
print(f"   Image Size:  {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"   Phase 1:     {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}")
print(f"   Phase 2:     {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}")