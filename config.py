from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpg", ".bmp", ".tif", ".tiff"}

# Dataset paths.
DATASET_DIR = Path("letters_output")
FEATURES_PATH = Path("letter_features.npz")
RESULTS_PATH = Path("classifier_results.csv")

# Feature extraction settings.
IMAGE_SIZE = 64
USE_HOG = True
USE_RAW_PIXELS = False
HOG_CELL_SIZE = 8
HOG_BLOCK_CELLS = 2
HOG_BLOCK_STRIDE = 1
HOG_BINS = 9

# Preprocessing / segmentation settings.
MIN_AREA = 50
PADDING = 2
MORPH_KERNEL = 0
APPLY_SKELETON = True
SKELETON_DILATE = True
SKELETON_DILATE_KERNEL = 5
SKELETON_DILATE_ITERATIONS = 1

# Augmentation settings (applied during extraction).
AUGMENT_COUNT = 0
AUGMENT_SEED = 42
AUGMENT_ROTATE_DEG = 6
AUGMENT_TRANSLATE_PX = 2
AUGMENT_SCALE_RANGE = 0.08
AUGMENT_DILATE_PROB = 0.2
AUGMENT_ERODE_PROB = 0.2
AUGMENT_DILATE_KERNEL = 2
AUGMENT_ERODE_KERNEL = 2
AUGMENT_BLUR_PROB = 0.15
AUGMENT_NOISE_PROB = 0.2
AUGMENT_NOISE_AMOUNT = 0.01
