"""Verify checkpoint keys and output shape using the shared strict loader."""
import argparse
from pathlib import Path
import torch

from ..src.checkpoints import find_checkpoint, load_student
from ..src.config import cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a student checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    checkpoint = find_checkpoint(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_student(checkpoint, device)
    with torch.inference_mode():
        output = model(torch.randn(1, 3, *cfg.IMAGE_SIZE, device=device))
    if output.shape != (1, cfg.NUM_CLASSES) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid classifier output: {output.shape}")
    print(f"Checkpoint OK: {checkpoint}")
    print(f"Device: {device}; output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
