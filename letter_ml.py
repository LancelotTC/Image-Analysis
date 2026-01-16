from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Edit these defaults for your dataset.
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

# Evaluation settings.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Run control.
RUN_FEATURE_EXTRACTION = True
RUN_CLASSIFIERS = True


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int
    use_hog: bool
    use_raw_pixels: bool
    hog_cell_size: int
    hog_block_cells: int
    hog_block_stride: int
    hog_bins: int


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    label_encoder: LabelEncoder
    paths: list[str]
    config: FeatureConfig


def iter_label_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def iter_images(label_dir: Path) -> list[Path]:
    return sorted([p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def load_dataset_paths(root: Path) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    for label_dir in iter_label_dirs(root):
        for image_path in iter_images(label_dir):
            image_paths.append(image_path)
            labels.append(label_dir.name)
    return image_paths, labels


def default_feature_config() -> FeatureConfig:
    return FeatureConfig(
        image_size=IMAGE_SIZE,
        use_hog=USE_HOG,
        use_raw_pixels=USE_RAW_PIXELS,
        hog_cell_size=HOG_CELL_SIZE,
        hog_block_cells=HOG_BLOCK_CELLS,
        hog_block_stride=HOG_BLOCK_STRIDE,
        hog_bins=HOG_BINS,
    )


def make_hog_descriptor(config: FeatureConfig) -> cv2.HOGDescriptor:
    win_size = (config.image_size, config.image_size)
    cell = (config.hog_cell_size, config.hog_cell_size)
    block = (config.hog_cell_size * config.hog_block_cells, config.hog_cell_size * config.hog_block_cells)
    stride = (config.hog_cell_size * config.hog_block_stride, config.hog_cell_size * config.hog_block_stride)
    return cv2.HOGDescriptor(win_size, block, stride, cell, config.hog_bins)


def ensure_size(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def extract_feature_vector(image: np.ndarray, config: FeatureConfig, hog: cv2.HOGDescriptor | None) -> np.ndarray:
    features = []
    if config.use_hog and hog is not None:
        hog_vec = hog.compute(image)
        features.append(hog_vec.reshape(-1))
    if config.use_raw_pixels:
        features.append(image.astype(np.float32).reshape(-1) / 255.0)
    if not features:
        raise ValueError("No features enabled. Set USE_HOG and/or USE_RAW_PIXELS to True.")
    return np.concatenate(features).astype(np.float32)


def extract_features_dataset(root: Path, config: FeatureConfig | None = None) -> Dataset:
    if config is None:
        config = default_feature_config()
    image_paths, labels = load_dataset_paths(root)
    if not image_paths:
        raise ValueError(f"No images found under {root}")

    hog = make_hog_descriptor(config) if config.use_hog else None
    features: list[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = ensure_size(image, config.image_size)
        features.append(extract_feature_vector(image, config, hog))

    X = np.vstack(features)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return Dataset(X=X, y=y, label_encoder=label_encoder, paths=[str(p) for p in image_paths], config=config)


def save_features(dataset: Dataset, path: Path) -> None:
    np.savez(
        path,
        X=dataset.X,
        y=dataset.y,
        labels=dataset.label_encoder.classes_,
        paths=np.array(dataset.paths, dtype=object),
        feature_image_size=dataset.config.image_size,
        feature_use_hog=dataset.config.use_hog,
        feature_use_raw_pixels=dataset.config.use_raw_pixels,
        feature_hog_cell_size=dataset.config.hog_cell_size,
        feature_hog_block_cells=dataset.config.hog_block_cells,
        feature_hog_block_stride=dataset.config.hog_block_stride,
        feature_hog_bins=dataset.config.hog_bins,
    )


def load_features(path: Path) -> Dataset:
    data = np.load(path, allow_pickle=True)
    config = FeatureConfig(
        image_size=int(data["feature_image_size"]) if "feature_image_size" in data else IMAGE_SIZE,
        use_hog=bool(data["feature_use_hog"].item()) if "feature_use_hog" in data else USE_HOG,
        use_raw_pixels=(
            bool(data["feature_use_raw_pixels"].item()) if "feature_use_raw_pixels" in data else USE_RAW_PIXELS
        ),
        hog_cell_size=int(data["feature_hog_cell_size"]) if "feature_hog_cell_size" in data else HOG_CELL_SIZE,
        hog_block_cells=int(data["feature_hog_block_cells"]) if "feature_hog_block_cells" in data else HOG_BLOCK_CELLS,
        hog_block_stride=(
            int(data["feature_hog_block_stride"]) if "feature_hog_block_stride" in data else HOG_BLOCK_STRIDE
        ),
        hog_bins=int(data["feature_hog_bins"]) if "feature_hog_bins" in data else HOG_BINS,
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(data["labels"])
    return Dataset(
        X=data["X"],
        y=data["y"],
        label_encoder=label_encoder,
        paths=data["paths"].tolist(),
        config=config,
    )


def can_stratify(y: np.ndarray) -> bool:
    counts = Counter(y)
    return min(counts.values()) >= 2


def build_classifiers() -> dict[str, object]:
    return {
        "log_reg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, n_jobs=None, random_state=RANDOM_STATE),
        ),
        "linear_svc": make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", C=3.0, random_state=RANDOM_STATE)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "adaboost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def evaluate_classifiers(dataset: Dataset) -> list[dict]:
    stratify = dataset.y if can_stratify(dataset.y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    results: list[dict] = []
    for name, model in build_classifiers().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, preds)),
                "macro_f1": float(f1_score(y_test, preds, average="macro")),
            }
        )

    results.sort(key=lambda item: (item["accuracy"], item["macro_f1"]), reverse=True)
    return results


def save_results(results: list[dict], path: Path) -> None:
    header = "model,accuracy,macro_f1"
    lines = [header]
    for item in results:
        lines.append(f"{item['model']},{item['accuracy']:.4f},{item['macro_f1']:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset: Dataset | None = None
    if RUN_FEATURE_EXTRACTION:
        dataset = extract_features_dataset(DATASET_DIR)
        save_features(dataset, FEATURES_PATH)
        print(f"Saved features to {FEATURES_PATH}")
        print(f"Samples: {dataset.X.shape[0]}, Features: {dataset.X.shape[1]}")

    if RUN_CLASSIFIERS:
        if dataset is None:
            if not FEATURES_PATH.exists():
                raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
            dataset = load_features(FEATURES_PATH)

        results = evaluate_classifiers(dataset)
        for item in results:
            print(f"{item['model']}: acc={item['accuracy']:.4f}, macro_f1={item['macro_f1']:.4f}")
        save_results(results, RESULTS_PATH)
        print(f"Saved classifier results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
