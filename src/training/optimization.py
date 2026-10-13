"""Optimizer, LR schedule, and checkpoint/early-stopping policy."""
from pathlib import Path
import torch
from ..config import cfg


def lr_lambda(epoch, warmup_epochs=5, total_epochs=cfg.EPOCHS,
              init_scale=0.1, min_scale=0.3):
    warmup_epochs = min(warmup_epochs, max(0, total_epochs - 1))
    if warmup_epochs and epoch < warmup_epochs:
        return init_scale + (1.0 - init_scale) * epoch / warmup_epochs
    decay = (total_epochs - epoch) / max(1, total_epochs - warmup_epochs)
    return min_scale + (1.0 - min_scale) * max(0.0, decay)


def create_optimizer(model, trained_model="student", kd=False):
    # KD changes the loss, not optimizer settings, in the controlled comparison.
    if trained_model == "student":
        groups = [
            {"params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
             "lr": cfg.STUDENT_BACKBONE_LR},
            {"params": model.fc.parameters(), "lr": cfg.STUDENT_HEAD_LR},
        ]
    elif trained_model == "teacher":
        groups = [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
    else:
        raise ValueError(f"Unknown model kind: {trained_model}")
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


class EarlyStopping:
    """Save best validation accuracy; stop when validation loss stops improving."""
    def __init__(self, patience, path):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.path = Path(path)
        self.counter = 0
        self.best_loss = float("inf")
        self.best_acc = float("-inf")
        self.early_stop = False

    def __call__(self, val_loss, val_acc, model):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            print(f"Saved checkpoint: {self.path} (validation accuracy {val_acc:.4f})")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience


def create_early_stopper(trained_model="student", kd=False):
    if trained_model == "teacher":
        return EarlyStopping(15, cfg.WEIGHTS_DIR / "train" / "best_teacher_model.pth")
    name = "best_student_model_withKD.pth" if kd else "best_student_model.pth"
    return EarlyStopping(cfg.STUDENT_PATIENCE, cfg.WEIGHTS_DIR / "train" / name)
