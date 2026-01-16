from pathlib import Path

import cv2
import numpy as np

from letter_ml import FEATURES_PATH, build_classifiers, extract_feature_vector, load_features, make_hog_descriptor


# Edit these defaults for your dataset.
WORD_IMAGE_PATH = Path("word.jpeg")
FEATURES_FILE = FEATURES_PATH
MIN_AREA = 50
PADDING = 2
MORPH_KERNEL = 0
APPLY_SKELETON = False
SKELETON_DILATE = False
SKELETON_DILATE_KERNEL = 3
SKELETON_DILATE_ITERATIONS = 1


def to_binary(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    unique = np.unique(gray)
    if unique.size > 2 or not set(unique.tolist()).issubset({0, 255}):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray.copy()

    white = np.count_nonzero(binary == 255)
    black = np.count_nonzero(binary == 0)
    if white > black:
        binary = cv2.bitwise_not(binary)

    return binary


def apply_morph(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def find_components(binary: np.ndarray, min_area: int) -> list[dict]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label].tolist()
        if area < min_area:
            continue
        components.append({"label": label, "x": x, "y": y, "w": w, "h": h, "area": area})
    components.sort(key=lambda c: c["x"])
    return components


def crop_with_padding(binary: np.ndarray, bbox: dict, pad: int) -> np.ndarray:
    x = max(0, bbox["x"] - pad)
    y = max(0, bbox["y"] - pad)
    x2 = min(binary.shape[1], bbox["x"] + bbox["w"] + pad)
    y2 = min(binary.shape[0], bbox["y"] + bbox["h"] + pad)
    return binary[y:y2, x:x2]


def resize_with_padding(binary: np.ndarray, size: int) -> np.ndarray:
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def skeletonize(binary: np.ndarray) -> np.ndarray:
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    work = binary.copy()
    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, temp)
        work = cv2.erode(work, element)
        if cv2.countNonZero(work) == 0:
            break

    if SKELETON_DILATE and SKELETON_DILATE_KERNEL > 0 and SKELETON_DILATE_ITERATIONS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (SKELETON_DILATE_KERNEL, SKELETON_DILATE_KERNEL),
        )
        skel = cv2.dilate(skel, kernel, iterations=SKELETON_DILATE_ITERATIONS)

    return skel


def load_word_components(image_path: Path) -> list[np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    binary = to_binary(image)
    binary = apply_morph(binary, MORPH_KERNEL)
    components = find_components(binary, MIN_AREA)
    if not components:
        raise ValueError("No components found. Adjust MIN_AREA or MORPH_KERNEL.")

    crops = []
    for component in components:
        crop = crop_with_padding(binary, component, PADDING)
        if APPLY_SKELETON:
            crop = skeletonize(crop)
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
        resized = resize_with_padding(crop, config.image_size)
        features.append(extract_feature_vector(resized, config, hog))

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
