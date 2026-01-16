from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

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
