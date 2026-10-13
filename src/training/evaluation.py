"""Evaluate validation or official test images with explicit checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..benchmarking.metrics import classification_metrics
from ..checkpoints import find_checkpoint, load_student
from ..config import cfg
from ..data.dataset import create_dataloaders, get_transforms, load_dataframe
from ..data.labels import list_images, load_ground_truth
from ..data.transforms import prepare_batch
from .reproducibility import set_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an ISIC student checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-kind", choices=("student", "student_kd"), default="student_kd")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    if args.split == "validation" and (args.input_dir or args.ground_truth):
        parser.error("Custom image/CSV paths apply to --split test")
    set_seed(cfg.SEED)
    checkpoint = find_checkpoint(args.checkpoint, args.model_kind)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_student(checkpoint, device)
    logits, targets = [], []
    with torch.inference_mode():
        if args.split == "validation":
            frame = load_dataframe(cfg.TRAIN_GROUND_TRUTH, cfg.TRAIN_IMAGE_DIR)
            _, loader, _ = create_dataloaders(frame, *get_transforms(), args.batch_size, args.workers)
            for images, labels in loader:
                logits.append(model(images.to(device)).cpu().numpy())
                targets.extend(labels.numpy())
        else:
            folder = args.input_dir or cfg.PROJECT_DIR / "ISIC2018_Task3_Test_Input"
            csv = args.ground_truth or cfg.PROJECT_DIR / "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
            images = list_images(folder)
            targets = load_ground_truth(csv, images)
            for start in range(0, len(images), args.batch_size):
                array = prepare_batch(images[start:start + args.batch_size])
                logits.append(model(torch.from_numpy(array).to(device)).cpu().numpy())
    logits = np.concatenate(logits)
    targets = np.asarray(targets)
    report = {"checkpoint": str(checkpoint), "split": args.split,
              "class_labels": list(cfg.CLASS_LABELS), "metrics": classification_metrics(logits, targets)}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        from .plots import plot_confusion_matrix
        plot_confusion_matrix(targets, logits.argmax(1), cfg.CLASS_LABELS)
