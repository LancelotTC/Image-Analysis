from pathlib import Path

import cv2

from config import (
    APPLY_SKELETON,
    FEATURES_PATH,
    MIN_AREA,
    MORPH_KERNEL,
    PADDING,
    SKELETON_DILATE,
    SKELETON_DILATE_ITERATIONS,
    SKELETON_DILATE_KERNEL,
)
from letter_ml import build_classifiers, extract_feature_vector, load_features, make_hog_descriptor
from utils import Image


# Edit these defaults for your dataset.
WORD_IMAGE_PATH = Path("word1.jpeg")
FEATURES_FILE = FEATURES_PATH
INVERT = True


def load_word_components(image_path: Path) -> list[Image]:
    binary = (
        Image(str(image_path), color_order="bgr", load_mode=cv2.IMREAD_GRAYSCALE)
        .binarize(150)
        .invert(INVERT)
        .open_close(kernel_size=MORPH_KERNEL)
    )
    components = binary.component_stats(min_area=MIN_AREA, connectivity=8)
    if not components:
        raise ValueError("No components found. Adjust MIN_AREA or MORPH_KERNEL.")

    components.sort(key=lambda obj: obj.bbox[0])
    crops: list[Image] = []
    for component in components:
        crop = binary.copy().crop_with_padding(component.bbox, PADDING)

        if APPLY_SKELETON:
            crop.skeletonize(
                dilate=SKELETON_DILATE,
                dilate_kernel=SKELETON_DILATE_KERNEL,
                dilate_iterations=SKELETON_DILATE_ITERATIONS,
            )
        crop.median_denoise()

        crops.append(crop)
    return crops


def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")
    if not WORD_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Word image not found: {WORD_IMAGE_PATH}")

    dataset = load_features(FEATURES_FILE)
    config = dataset.config
    hog = make_hog_descriptor(config) if config.use_hog else None

    crops = load_word_components(WORD_IMAGE_PATH)
    features = []
    for crop in crops:
        crop.resize_with_padding(config.image_size).binarize(150).median_denoise()
        features.append(extract_feature_vector(crop.data, config, hog))

    print(f"Word image: {WORD_IMAGE_PATH}")
    print(f"Detected letters: {len(features)}")
    print("Predictions:")
    for name, model in build_classifiers().items():
        model.fit(dataset.X, dataset.y)
        predicted_chars = []
        for vector in features:
            pred = model.predict(vector.reshape(1, -1))
            label = dataset.label_encoder.inverse_transform(pred)[0]
            predicted_chars.append(str(label))
        predicted_word = "".join(predicted_chars)
        print(f"{name}: {predicted_word}")


if __name__ == "__main__":
    main()
