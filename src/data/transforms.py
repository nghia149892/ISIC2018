"""Shared preprocessing for validation, ONNX, and TensorRT."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
from ..config import cfg

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@lru_cache(maxsize=1)
def validation_transform():
    return transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_transforms():
    train = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return train, validation_transform()


def preprocess(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        tensor = validation_transform()(image.convert("RGB")).unsqueeze(0)
    return np.ascontiguousarray(tensor.numpy(), dtype=np.float32)


def prepare_batch(paths: list[Path]) -> np.ndarray:
    return np.concatenate([preprocess(path) for path in paths], axis=0)
