"""Datasets and deterministic stratified train/validation split."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from ..config import cfg
from .labels import read_labels
from .transforms import get_transforms


def load_dataframe(csv_path, img_dir):
    labels = read_labels(Path(csv_path))
    # Preserve actual filenames while matching CSV IDs case-insensitively.
    paths = {p.stem.casefold(): p for p in Path(img_dir).glob("*.jpg")}
    rows = []
    for name, index in labels.items():
        if name not in paths:
            raise FileNotFoundError(f"Training image not found: {name} in {img_dir}")
        rows.append({"image": name, "image_path": str(paths[name]), "label": cfg.CLASS_LABELS[index]})
    if not rows:
        raise ValueError("Training CSV contains no examples")
    return pd.DataFrame(rows)


class ISICClassificationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_paths = df["image_path"].tolist()
        self.labels_str = df["label"].tolist()
        self.transform = transform
        self.label_to_idx = {label: i for i, label in enumerate(cfg.CLASS_LABELS)}
        self.idx_to_label = dict(enumerate(cfg.CLASS_LABELS))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_to_idx[self.labels_str[idx]]


def create_dataloaders(df, train_transform, val_transform, batch_size=None, num_workers=None):
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=cfg.SEED,
    )
    batch_size = cfg.BATCH_SIZE if batch_size is None else batch_size
    workers = cfg.NUM_WORKERS if num_workers is None else num_workers
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    train_dataset = ISICClassificationDataset(train_df, train_transform)
    val_dataset = ISICClassificationDataset(val_df, val_transform)
    # A separate RNG keeps data order independent of teacher initialization.
    generator = torch.Generator().manual_seed(cfg.SEED)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=cfg.PIN_MEMORY)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, train_dataset.idx_to_label
