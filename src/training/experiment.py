"""Shared CLI for teacher, student, and KD training."""
import argparse
from functools import partial
import json
from pathlib import Path
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from ..checkpoints import find_checkpoint
from ..config import cfg
from ..data.dataset import create_dataloaders, get_transforms, load_dataframe
from ..models.classifiers import DINOv2Classifier, StudentModel
from .loops import fit, fit_kd
from .optimization import EarlyStopping, create_optimizer, lr_lambda
from .reproducibility import set_seed


def main(kind="student_kd", argv=None):
    parser = argparse.ArgumentParser(description=f"Train ISIC {kind}")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=15 if kind == "teacher" else cfg.STUDENT_PATIENCE)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing training checkpoint")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize student without ImageNet weights")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error("epochs, batch size, patience must be positive; workers must be nonnegative")
    names = {"teacher": "best_teacher_model.pth", "student": "best_student_model.pth",
             "student_kd": "best_student_model_withKD.pth"}
    output = args.output or cfg.WEIGHTS_DIR / "train" / names[kind]
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint exists: {output}. Choose --output or use --overwrite.")
    teacher_checkpoint = find_checkpoint(args.teacher_checkpoint, "teacher") if kind == "student_kd" else None
    set_seed(cfg.SEED)
    frame = load_dataframe(cfg.TRAIN_GROUND_TRUTH, cfg.TRAIN_IMAGE_DIR)
    train_transform, val_transform = get_transforms()
    train, validation, classes = create_dataloaders(
        frame, train_transform, val_transform, args.batch_size, args.workers,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (DINOv2Classifier() if kind == "teacher"
             else StudentModel(pretrained=not args.no_pretrained)).to(device)
    teacher = None
    if kind == "student_kd":
        teacher = DINOv2Classifier().to(device)
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location="cpu", weights_only=True))
        teacher.requires_grad_(False).eval()
    optimizer = create_optimizer(model, "teacher" if kind == "teacher" else "student")
    scheduler = LambdaLR(optimizer, partial(lr_lambda, total_epochs=args.epochs))
    stopper = EarlyStopping(args.patience, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": kind, "seed": cfg.SEED, "class_labels": list(cfg.CLASS_LABELS),
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "patience": args.patience, "initial_learning_rates": [g["initial_lr"] for g in optimizer.param_groups],
        "weight_decay": cfg.WEIGHT_DECAY, "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
        "temperature": cfg.TEMPERATURE if teacher else None, "alpha": cfg.ALPHA if teacher else None,
        "train_images": train.dataset.df["image"].tolist(),
        "validation_images": validation.dataset.df["image"].tolist(),
        "pretrained_student": not args.no_pretrained if kind != "teacher" else None,
    }
    # Reset after teacher construction so augmentation/dropout RNG is comparable.
    set_seed(cfg.SEED)
    common = (train, validation, nn.CrossEntropyLoss(), optimizer, scheduler, device, args.epochs, stopper)
    if teacher is None:
        history = fit(model, *common)
    else:
        history = fit_kd(model, teacher, *common, T=cfg.TEMPERATURE, alpha=cfg.ALPHA)
    manifest["history"] = dict(zip(("train_loss", "validation_loss", "validation_accuracy"), history))
    manifest["best_validation_accuracy"] = stopper.best_acc
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        from .plots import plot_loss_acc
        plot_loss_acc(*history)
