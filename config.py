# ============================================================
# config.py — UPDATED for Stage 4
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
DEMO_SCREENSHOTS_DIR = PROJECT_ROOT / "outputs" / "demo_screenshots"
DEMO_VIDEOS_DIR     = PROJECT_ROOT / "outputs" / "demo_videos"
LOGS_DIR            = PROJECT_ROOT / "logs"

# ── Create all dirs ──────────────────────────────────────────
for d in [PLOTS_DIR, CHECKPOINTS_DIR, FINAL_MODELS_DIR,
          TRAINING_LOGS_DIR, EVALUATION_DIR, LOGS_DIR,
          DEMO_SCREENSHOTS_DIR, DEMO_VIDEOS_DIR]:
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

# ── Normalization ────────────────────────────────────────────
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

# ── Model ────────────────────────────────────────────────────
MODEL_NAME          = "DenseNet121"
DENSENET_WEIGHTS    = "imagenet"
DROPOUT_RATE        = 0.4
DENSE_UNITS         = 512
BATCH_SIZE          = 32
NUM_WORKERS         = 4

# ── Training ────────────────────────────────────────────────
PHASE1_EPOCHS       = 10
PHASE1_LR           = 0.001
PHASE1_MODEL_PATH   = FINAL_MODELS_DIR / "densenet_phase1.keras"
PHASE2_EPOCHS       = 15
PHASE2_LR           = 1e-5
PHASE2_UNFREEZE_FROM = 310
PHASE2_MODEL_PATH   = FINAL_MODELS_DIR / "densenet_final.keras"
CHECKPOINT_PATH     = CHECKPOINTS_DIR / "best_model.keras"
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.3
REDUCE_LR_MIN       = 1e-7
CONFUSION_MATRIX_PATH       = EVALUATION_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH  = EVALUATION_DIR / "classification_report.txt"
PER_CLASS_F1_PATH           = EVALUATION_DIR / "per_class_f1.png"

# ════════════════════════════════════════════════════════════
# STAGE 4: REAL-TIME INFERENCE CONFIGURATION
# ════════════════════════════════════════════════════════════

# ── Model for inference ──────────────────────────────────────
INFERENCE_MODEL_PATH    = CHECKPOINT_PATH       # Use best model

# ── Webcam ───────────────────────────────────────────────────
WEBCAM_INDEX            = 0                     # 0 = default camera
WEBCAM_WIDTH            = 1280                  # Camera resolution
WEBCAM_HEIGHT           = 720
WEBCAM_FPS              = 30
DISPLAY_WIDTH           = 1280                  # Display window size
DISPLAY_HEIGHT          = 720

# ── Inference ────────────────────────────────────────────────
INFERENCE_IMAGE_SIZE    = 224
INFERENCE_BATCH_SIZE    = 1

# ── Stability Buffer ─────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.70                  # Min 70% confidence
STABILITY_FRAMES        = 5                     # 5 consecutive frames
COOLDOWN_FRAMES         = 15                    # Wait N frames after confirm

# ── Sentence Builder ─────────────────────────────────────────
MAX_SENTENCE_LENGTH     = 50                    # Max chars on screen
MAX_WORD_LENGTH         = 20                    # Max letters per word

# ── UI Colors (BGR format for OpenCV) ────────────────────────
COLOR_GREEN             = (0,   255,  0  )
COLOR_RED               = (0,   0,    255)
COLOR_BLUE              = (255, 100,  0  )
COLOR_WHITE             = (255, 255,  255)
COLOR_BLACK             = (0,   0,    0  )
COLOR_YELLOW            = (0,   255,  255)
COLOR_ORANGE            = (0,   165,  255)
COLOR_CYAN              = (255, 255,  0  )
COLOR_PURPLE            = (255, 0,    255)
COLOR_DARK_GRAY         = (50,  50,   50 )
COLOR_LIGHT_GRAY        = (200, 200,  200)

# ── UI Layout ────────────────────────────────────────────────
PANEL_WIDTH             = 320                   # Right info panel
FONT                    = 0                     # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE        = 1.4
FONT_SCALE_MEDIUM       = 0.9
FONT_SCALE_SMALL        = 0.65
FONT_THICKNESS_BOLD     = 2
FONT_THICKNESS_NORMAL   = 1

# ── Demo Recording ───────────────────────────────────────────
RECORD_DEMO             = False                 # Set True to record video
DEMO_VIDEO_PATH         = DEMO_VIDEOS_DIR / "demo.avi"
SCREENSHOT_KEY          = ord('p')             # Press 'p' to screenshot

print("✅ Config loaded — Stage 4 settings ready!")
print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"   Stability frames:     {STABILITY_FRAMES}")
print(f"   Webcam index:         {WEBCAM_INDEX}")