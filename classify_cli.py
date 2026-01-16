from pathlib import Path

import cv2

from letter_ml import (
    FEATURES_PATH,
    build_classifiers,
    ensure_size,
    extract_feature_vector,
    load_features,
    make_hog_descriptor,
)


def prompt_classifier(classifiers: dict[str, object]) -> str | None:
    names = list(classifiers.keys())
    print("Available classifiers:")
    for idx, name in enumerate(names, start=1):
        print(f"{idx}. {name}")

    choice = input("Choose classifier by number or name (blank to quit): ").strip()
    if not choice:
        return None

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            return names[idx]
        print("Invalid selection.")
        return None

    choice = choice.lower()
    for name in names:
        if choice == name.lower():
            return name

    print("Unknown classifier name.")
    return None


def prompt_image_path() -> Path | None:
    raw = input("Image path to classify (blank to quit): ").strip().strip('"').strip("'")
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        print("Path not found.")
        return None
    return path


def classify_image() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    dataset = load_features(FEATURES_PATH)
    classifiers = build_classifiers()
    classifier_name = prompt_classifier(classifiers)
    if classifier_name is None:
        return

    image_path = prompt_image_path()
    if image_path is None:
        return

    model = classifiers[classifier_name]
    model.fit(dataset.X, dataset.y)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    config = dataset.config
    image = ensure_size(image, config.image_size)
    hog = make_hog_descriptor(config) if config.use_hog else None
    features = extract_feature_vector(image, config, hog).reshape(1, -1)

    pred = model.predict(features)
    label = dataset.label_encoder.inverse_transform(pred)[0]
    print(f"Predicted letter: {label}")


if __name__ == "__main__":
    classify_image()
