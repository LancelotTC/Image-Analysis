from pathlib import Path

import cv2

from config import FEATURES_PATH
from letter_ml import build_classifiers, extract_feature_vector, load_features, make_hog_descriptor
from utils import Image


# Edit these defaults for your dataset.
SAMPLE_IMAGE_PATH = Path("image.jpeg")
SAMPLE_IMAGE_PATH = Path("test/5.jpeg")
FEATURES_FILE = FEATURES_PATH


def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")
    if not SAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

    dataset = load_features(FEATURES_FILE)
    config = dataset.config

    image = Image(str(SAMPLE_IMAGE_PATH), color_order="bgr", load_mode=cv2.IMREAD_GRAYSCALE)
    image.resize(width=config.image_size, height=config.image_size)
    hog = make_hog_descriptor(config) if config.use_hog else None
    features = extract_feature_vector(image.data, config, hog).reshape(1, -1)

    print(f"Sample image: {SAMPLE_IMAGE_PATH}")
    print("Predictions:")
    for name, model in build_classifiers().items():
        model.fit(dataset.X, dataset.y)
        pred = model.predict(features)
        label = dataset.label_encoder.inverse_transform(pred)[0]
        print(f"{name}: {label}")


if __name__ == "__main__":
    main()
