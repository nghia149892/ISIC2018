"""Benchmark CLI; setup, timing, and reporting live in separate modules."""
import argparse
import importlib.util
from pathlib import Path
import torch
from ..checkpoints import file_sha256, find_checkpoint
from ..config import cfg
from ..data.labels import list_images, load_ground_truth
from ..inference.backends import DEFAULT_MODEL, load_backends
from .report import environment_info, print_batch, print_scaling, save_report
from .runner import benchmark_batch

DEFAULT_INPUT_DIR = cfg.PROJECT_DIR / "ISIC2018_Task3_Test_Input"
DEFAULT_GROUND_TRUTH = cfg.PROJECT_DIR / "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark ISIC classifier backends")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Repeated single-image inference")
    source.add_argument("--input-dir", type=Path, help="Folder inference; defaults to ISIC test images")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--engine-fp32", "--engine", dest="engine_fp32", type=Path)
    parser.add_argument("--engine-fp16", type=Path)
    parser.add_argument("--cpu-only", action="store_true", help="Disable standalone TensorRT")
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--no-ground-truth", action="store_true", help="Measure latency only")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50, help="Repetitions in single-image mode only")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--limit", type=int, help="Evaluate the first N sorted images for a quick check")
    parser.add_argument("--output", type=Path, default=cfg.PROJECT_DIR / "results/benchmark.json")
    args = parser.parse_args(argv)
    if args.warmup < 0 or min(args.runs, args.threads) < 1:
        parser.error("warmup must be >= 0; runs and threads must be positive")
    if any(n < 1 for n in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch sizes must be distinct positive integers")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.cpu_only and (args.engine_fp32 or args.engine_fp16):
        parser.error("--cpu-only cannot be combined with explicit engines")
    if args.ground_truth and args.no_ground_truth:
        parser.error("--ground-truth and --no-ground-truth are mutually exclusive")
    return args


def select_engines(args):
    if args.cpu_only:
        return {}
    explicit = {p: getattr(args, f"engine_{p}") for p in ("fp32", "fp16")}
    available = torch.cuda.is_available() and importlib.util.find_spec("tensorrt") is not None
    if not available:
        if any(explicit.values()):
            raise RuntimeError("Explicit TensorRT engines require TensorRT and a CUDA GPU")
        return {}
    return {
        precision: path for precision in ("fp32", "fp16")
        if (path := explicit[precision] or next((
            args.model.parent / name
            for name in (f"{args.model.stem}_{precision}_bs4.engine", f"{args.model.stem}_{precision}.engine")
            if (args.model.parent / name).is_file()
        ), None)) is not None
    }


def main(argv=None):
    args = parse_args(argv)
    folder = args.input_dir or DEFAULT_INPUT_DIR
    images = [args.image] if args.image else list_images(folder, args.recursive)
    if args.limit:
        images = images[:args.limit]
    csv_path = args.ground_truth
    if not args.image and folder.resolve() == DEFAULT_INPUT_DIR.resolve() and not args.no_ground_truth:
        csv_path = csv_path or DEFAULT_GROUND_TRUTH
    targets = load_ground_truth(csv_path, images) if csv_path and not args.no_ground_truth else None
    checkpoint = find_checkpoint(args.checkpoint)
    engines = select_engines(args)
    backends = load_backends(checkpoint, args.model, engines, args.threads)
    print(f"Images: {len(images)} | checkpoint: {checkpoint}")
    print(f"Backends: {', '.join(b.name for b in backends)}")
    print("Timing excludes decoding, preprocessing, and model loading.")
    print("TensorRT timing includes buffer allocation and CPU/GPU copies.")
    if targets is None:
        print("Ground truth not selected; classification metrics will not be calculated.")
    metadata = environment_info()
    metadata["configuration"] = {
        "mode": "single-image" if args.image else "folder",
        "image_count": len(images), "images": [str(p) for p in images],
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "onnx": str(args.model), "onnx_sha256": file_sha256(args.model),
        "onnx_external_data_sha256": (
            file_sha256(args.model.with_suffix(args.model.suffix + ".data"))
            if args.model.with_suffix(args.model.suffix + ".data").is_file() else None
        ),
        "ground_truth": str(csv_path) if targets is not None else None,
        "warmup": args.warmup, "runs": args.runs if args.image else None,
        "threads": args.threads, "batch_sizes": args.batch_sizes,
        "class_labels": list(cfg.CLASS_LABELS),
        "engines": {p: {"path": str(path), "sha256": file_sha256(path)} for p, path in engines.items()},
    }
    metadata["results"] = []
    for size in args.batch_sizes:
        result = benchmark_batch(backends, images, size, args.warmup, args.runs, bool(args.image), targets)
        metadata["results"].append(result)
        print_batch(result)
    print_scaling(metadata["results"])
    save_report(args.output, metadata)
