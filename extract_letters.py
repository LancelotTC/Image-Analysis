import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Edit these defaults for your dataset.
INPUT_PATH = Path("letters")
OUTPUT_DIR = Path("letters_output")
OUTPUT_SIZE = 64
MIN_AREA = 50
PADDING = 2
MORPH_KERNEL = 0
KEEP_ALL = True
WRITE_MANIFEST = True


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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label].tolist()
        if area < min_area:
            continue
        components.append({"label": label, "x": x, "y": y, "w": w, "h": h, "area": area})
    components.sort(key=lambda c: c["area"], reverse=True)
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


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def empty_dir(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def process_image(
    image_path: Path,
    output_dir: Path,
    base_output_dir: Path,
    size: int,
    min_area: int,
    pad: int,
    morph: int,
    keep_all: bool,
    manifest_rows: list[dict],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    binary = to_binary(image)
    binary = apply_morph(binary, morph)
    components = find_components(binary, min_area)
    if not components:
        return

    targets = components if keep_all else [components[0]]
    for idx, component in enumerate(targets, start=1):
        crop = crop_with_padding(binary, component, pad)
        resized = resize_with_padding(crop, size)
        output_name = f"{image_path.stem}_obj{idx}.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), resized)
        output_rel = output_path.relative_to(base_output_dir).as_posix()
        manifest_rows.append(
            {
                "source": image_path.name,
                "output": output_rel,
                "x": component["x"],
                "y": component["y"],
                "w": component["w"],
                "h": component["h"],
                "area": component["area"],
            }
        )


def extract_letters(
    input_path: Path,
    output_dir: Path,
    size: int,
    min_area: int,
    pad: int,
    morph: int,
    keep_all: bool,
    write_manifest: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    empty_dir(output_dir)

    manifest_rows: list[dict] = []
    for image_path in iter_images(input_path):
        letter_dir = output_dir / image_path.stem
        letter_dir.mkdir(parents=True, exist_ok=True)
        process_image(
            image_path=image_path,
            output_dir=letter_dir,
            base_output_dir=output_dir,
            size=size,
            min_area=min_area,
            pad=pad,
            morph=morph,
            keep_all=keep_all,
            manifest_rows=manifest_rows,
        )

    if write_manifest:
        manifest_path = output_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "output", "x", "y", "w", "h", "area"])
            writer.writeheader()
            writer.writerows(manifest_rows)


if __name__ == "__main__":
    extract_letters(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        size=OUTPUT_SIZE,
        min_area=MIN_AREA,
        pad=PADDING,
        morph=MORPH_KERNEL,
        keep_all=KEEP_ALL,
        write_manifest=WRITE_MANIFEST,
    )
