"""Build a standalone TensorRT engine from the ISIC2018 ONNX model.

Examples (run from the repository root):
    python -m ISIC2018 build-engine --precision fp32
    python -m ISIC2018 build-engine --precision fp16

The resulting engine is specific to this TensorRT/GPU environment.
"""

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch
from ..config import cfg
from ..checkpoints import file_sha256


PROJECT_DIR = cfg.PROJECT_DIR
DEFAULT_ONNX = PROJECT_DIR / "weights" / "resnet18.onnx"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Input ONNX path")
    parser.add_argument("--output", type=Path, help="Output engine path")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32",
        help="Engine precision; start with fp32, then try fp16",
    )
    parser.add_argument(
        "--workspace-gb", type=float, default=1.0,
        help="Maximum TensorRT builder workspace in GB",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=8,
        help="Maximum dynamic batch size supported by the engine",
    )
    parser.add_argument(
        "--opt-batch-size", type=int, default=4,
        help="Batch size TensorRT should optimize the profile around (default: 4)",
    )
    args = parser.parse_args(argv)

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if args.workspace_gb <= 0 or args.max_batch_size < 1:
        parser.error("--workspace-gb and --max-batch-size must be positive")
    if not 1 <= args.opt_batch_size <= args.max_batch_size:
        parser.error("--opt-batch-size must be between 1 and --max-batch-size")
    if not torch.cuda.is_available():
        raise RuntimeError("Building a TensorRT engine requires an available NVIDIA CUDA GPU.")

    output_path = args.output or args.onnx.with_name(
        f"{args.onnx.stem}_{args.precision}.engine"
    )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing engine: {output_path}. "
            "Choose another --output path or remove the old engine yourself."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # Explicit batch is the default in TensorRT 10; the ONNX graph has a dynamic batch.
    network = builder.create_network(0)
    parser_instance = trt.OnnxParser(network, logger)

    print(f"TensorRT version: {trt.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parsing ONNX: {args.onnx}")
    # Parse from disk so external ONNX initializers (the .onnx.data sidecar) are resolved.
    if not parser_instance.parse_from_file(str(args.onnx.resolve())):
        errors = "\n".join(
            str(parser_instance.get_error(i))
            for i in range(parser_instance.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse the ONNX model:\n{errors}")

    if network.num_inputs != 1:
        raise ValueError(f"Expected one model input, found {network.num_inputs}")
    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)
    if len(input_shape) != 4 or input_shape != (-1, 3, 224, 224):
        raise ValueError(f"Unexpected input shape for this classifier: {input_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 224, 224),
        opt=(args.opt_batch_size, 3, 224, 224),
        max=(args.max_batch_size, 3, 224, 224),
    )

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1024**3))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.add_optimization_profile(profile)
    if args.precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)

    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Input: {input_tensor.name} {input_shape}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Workspace limit: {args.workspace_gb:.2f} GB")
    print(
        "Building engine with profile "
        f"min=1 / opt={args.opt_batch_size} / max={args.max_batch_size}..."
    )
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the engine; see the log above.")

    output_path.write_bytes(bytes(serialized_engine))
    manifest = {
        "engine_sha256": file_sha256(output_path),
        "onnx_sha256": file_sha256(args.onnx),
        "onnx_external_data_sha256": (
            file_sha256(args.onnx.with_suffix(args.onnx.suffix + ".data"))
            if args.onnx.with_suffix(args.onnx.suffix + ".data").is_file() else None
        ),
        "precision": args.precision, "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "profile": {"min": 1, "opt": args.opt_batch_size, "max": args.max_batch_size},
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Engine saved: {output_path}")
    print(f"Engine size: {output_path.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
