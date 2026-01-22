import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import (
    APPLY_SKELETON,
    AUGMENT_BLUR_PROB,
    AUGMENT_COUNT,
    AUGMENT_DILATE_KERNEL,
    AUGMENT_DILATE_PROB,
    AUGMENT_ERODE_KERNEL,
    AUGMENT_ERODE_PROB,
    AUGMENT_NOISE_AMOUNT,
    AUGMENT_NOISE_PROB,
    AUGMENT_ROTATE_DEG,
    AUGMENT_SCALE_RANGE,
    AUGMENT_SEED,
    AUGMENT_TRANSLATE_PX,
    DATASET_DIR,
    IMAGE_SIZE,
    IMAGE_SUFFIXES,
    MIN_AREA,
    MORPH_KERNEL,
    PADDING,
    SKELETON_DILATE,
    SKELETON_DILATE_ITERATIONS,
    SKELETON_DILATE_KERNEL,
)
from utils import Image

# Edit these defaults for your dataset.
INPUT_PATH = Path("letters")
OUTPUT_DIR = DATASET_DIR
KEEP_ALL = True
WRITE_MANIFEST = True


def merge_components(components: list[dict]) -> dict:
    xs = [c["x"] for c in components]
    ys = [c["y"] for c in components]
    x2s = [c["x"] + c["w"] for c in components]
    y2s = [c["y"] + c["h"] for c in components]
    return {
        "label": 0,
        "x": min(xs),
        "y": min(ys),
        "w": max(x2s) - min(xs),
        "h": max(y2s) - min(ys),
        "area": sum(c["area"] for c in components),
    }


def augment_image(image: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    angle = rng.uniform(-AUGMENT_ROTATE_DEG, AUGMENT_ROTATE_DEG)
    scale = rng.uniform(1.0 - AUGMENT_SCALE_RANGE, 1.0 + AUGMENT_SCALE_RANGE)
    tx = rng.uniform(-AUGMENT_TRANSLATE_PX, AUGMENT_TRANSLATE_PX)
    ty = rng.uniform(-AUGMENT_TRANSLATE_PX, AUGMENT_TRANSLATE_PX)

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    if AUGMENT_DILATE_PROB > 0 and rng.random() < AUGMENT_DILATE_PROB:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (AUGMENT_DILATE_KERNEL, AUGMENT_DILATE_KERNEL))
        warped = cv2.dilate(warped, kernel, iterations=1)
    elif AUGMENT_ERODE_PROB > 0 and rng.random() < AUGMENT_ERODE_PROB:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (AUGMENT_ERODE_KERNEL, AUGMENT_ERODE_KERNEL))
        warped = cv2.erode(warped, kernel, iterations=1)

    if AUGMENT_BLUR_PROB > 0 and rng.random() < AUGMENT_BLUR_PROB:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)

    if AUGMENT_NOISE_PROB > 0 and rng.random() < AUGMENT_NOISE_PROB and AUGMENT_NOISE_AMOUNT > 0:
        count = int(AUGMENT_NOISE_AMOUNT * h * w)
        if count > 0:
            ys = np_rng.integers(0, h, size=count)
            xs = np_rng.integers(0, w, size=count)
            values = np_rng.integers(0, 2, size=count)
            warped[ys, xs] = np.where(values == 0, 0, 255)

    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    return warped


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
    rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    binary = (
        Image(str(image_path), color_order="bgr", load_mode=cv2.IMREAD_GRAYSCALE)
        .binarize(150)
        .invert(True)
        .median_denoise()
    )
    # .binarize_for_components(
    #     min_area=min_area,
    #     connectivity=8,
    # )

    # binary.open_close(kernel_size=morph)
    components = []
    for obj in binary.component_stats(min_area=min_area, connectivity=8):
        x, y, w, h = obj.bbox
        components.append({"x": x, "y": y, "w": w, "h": h, "area": obj.area})
    components.sort(key=lambda c: c["area"], reverse=True)
    if not components:
        return

    targets = components if keep_all else [merge_components(components)]
    for idx, component in enumerate(targets, start=1):
        crop = binary.copy().crop_with_padding((component["x"], component["y"], component["w"], component["h"]), pad)
        if APPLY_SKELETON:
            crop.skeletonize(
                dilate=SKELETON_DILATE,
                dilate_kernel=SKELETON_DILATE_KERNEL,
                dilate_iterations=SKELETON_DILATE_ITERATIONS,
            )
        crop.resize_with_padding(size)
        output_name = f"{image_path.stem}_obj{idx}.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), crop.data)
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

        if AUGMENT_COUNT <= 0:
            continue

        base = crop.data.copy()
        for aug_idx in range(1, AUGMENT_COUNT + 1):
            augmented = augment_image(base, rng, np_rng)
            aug_name = f"{image_path.stem}_obj{idx}_aug{aug_idx}.png"
            aug_path = output_dir / aug_name
            cv2.imwrite(str(aug_path), augmented)
            aug_rel = aug_path.relative_to(base_output_dir).as_posix()
            manifest_rows.append(
                {
                    "source": image_path.name,
                    "output": aug_rel,
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

    rng = random.Random(AUGMENT_SEED)
    np_rng = np.random.default_rng(AUGMENT_SEED)
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
            rng=rng,
            np_rng=np_rng,
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
        size=IMAGE_SIZE,
        min_area=MIN_AREA,
        pad=PADDING,
        morph=MORPH_KERNEL,
        keep_all=KEEP_ALL,
        write_manifest=WRITE_MANIFEST,
    )
