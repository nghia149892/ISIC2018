"""Image discovery and strict ISIC one-hot ground-truth parsing."""
import csv
from pathlib import Path
import numpy as np
from ..config import cfg

CLASS_LABELS = cfg.CLASS_LABELS
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Image folder not found: {input_dir}")
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported images in: {input_dir}")
    return images


def read_labels(csv_path: Path) -> dict[str, int]:
    labels = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"image", *CLASS_LABELS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        for row in reader:
            image_id = Path(row["image"].strip()).stem.casefold()
            values = np.array([float(row[label]) for label in CLASS_LABELS])
            if not image_id or image_id in labels:
                raise ValueError(f"Empty or duplicate image ID: {image_id!r}")
            if not np.all((values == 0) | (values == 1)) or values.sum() != 1:
                raise ValueError(f"Expected exactly one one-hot class for: {image_id}")
            labels[image_id] = int(values.argmax())
    return labels


def load_ground_truth(csv_path: Path, images: list[Path]) -> np.ndarray:
    labels = read_labels(csv_path)
    missing = [p.name for p in images if p.stem.casefold() not in labels]
    if missing:
        raise ValueError(f"Ground truth missing {len(missing)} image(s): {missing[:5]}")
    return np.array([labels[p.stem.casefold()] for p in images], dtype=np.int64)
