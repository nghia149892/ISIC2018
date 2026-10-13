"""Shared checkpoint selection for training, export, and evaluation."""
import hashlib
from pathlib import Path
import warnings
import torch
from .config import cfg
from .models.classifiers import StudentModel


def find_checkpoint(path: Path | None = None, kind: str = "student_kd") -> Path:
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    names = {
        "student_kd": ("best_student_model_withKD.pth", "best_student_model_withKD", "best_model_withKD.pth"),
        "student": ("best_student_model.pth",),
        "teacher": ("best_teacher_model.pth",),
    }[kind]
    candidates = [
        directory / name
        for directory in (cfg.WEIGHTS_DIR / "train", cfg.WEIGHTS_DIR)
        for name in names
    ]
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise FileNotFoundError(f"No {kind} checkpoint found. Looked for: {candidates}")
    if len(found) > 1:
        warnings.warn(f"Multiple checkpoints found; using {found[0]}. Use --checkpoint to select explicitly.")
    return found[0]


def load_student(path: Path, device="cpu"):
    model = StudentModel(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
