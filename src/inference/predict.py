"""Single-image ONNX Runtime inference CLI."""
import argparse
from pathlib import Path
import numpy as np
from ..config import cfg
from ..data.transforms import preprocess
from .backends import DEFAULT_MODEL, make_session


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify one skin lesion image")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args(argv)
    if args.fp16 and args.backend != "tensorrt":
        parser.error("--fp16 applies only to --backend tensorrt")
    session = make_session(args.model, args.backend, args.fp16)
    scores = session.run(None, {session.get_inputs()[0].name: preprocess(args.image)})[0][0]
    exponentials = np.exp(scores - scores.max())
    probabilities = exponentials / exponentials.sum()
    labels = session.get_modelmeta().custom_metadata_map.get("class_labels", ",".join(cfg.CLASS_LABELS)).split(",")
    if len(labels) != len(probabilities):
        raise ValueError("Number of model outputs does not match class labels")
    print(f"Active providers: {session.get_providers()}")
    for index in np.argsort(probabilities)[::-1]:
        print(f"  {labels[index]}: {probabilities[index]:.4f}")
