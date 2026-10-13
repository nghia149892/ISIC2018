"""Export a student checkpoint and validate ONNX across dynamic batch sizes."""
import argparse
from pathlib import Path
import numpy as np
import onnx
import torch
from ..checkpoints import file_sha256, find_checkpoint, load_student
from ..config import cfg
from ..training.reproducibility import set_seed
from .backends import DEFAULT_MODEL, make_session


def export_student(checkpoint: Path, output: Path, overwrite=False) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"ONNX exists: {output}. Choose --output or use --overwrite.")
    set_seed(cfg.SEED)
    model = load_student(checkpoint)
    example = torch.randn(1, 3, *cfg.IMAGE_SIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Explicit legacy exporter keeps dynamic_axes behavior stable without onnxscript.
    torch.onnx.export(
        model, example, str(output), dynamo=False, opset_version=18,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(graph, {
        "class_labels": ",".join(cfg.CLASS_LABELS),
        "checkpoint_sha256": file_sha256(checkpoint),
        "preprocessing": "RGB;resize=224x224;ImageNet mean/std",
        "output_type": "logits",
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    session = make_session(output)
    with torch.inference_mode():
        for size in (1, 4, 8):
            array = torch.randn(size, 3, *cfg.IMAGE_SIZE)
            expected = model(array).numpy()
            actual = session.run(None, {"input": array.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            print(f"Verified ONNX batch {size}; max logit difference {np.max(np.abs(actual - expected)):.8g}")
    print(f"Exported checkpoint {checkpoint} to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export student checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    export_student(find_checkpoint(args.checkpoint), args.output, args.overwrite)
