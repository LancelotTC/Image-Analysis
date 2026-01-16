import csv
import cv2
import shutil
from utils import Image
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Edit these defaults for your dataset.
INPUT_PATH = Path("letters")
OUTPUT_DIR = Path("letters_output")
OUTPUT_SIZE = 64
MIN_AREA = 50
PADDING = 2
MORPH_KERNEL = 0
APPLY_SKELETON = True
SKELETON_DILATE = True
SKELETON_DILATE_KERNEL = 5
SKELETON_DILATE_ITERATIONS = 1
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
