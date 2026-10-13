"""Backend setup and a common CPU-array inference interface."""
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable
import numpy as np
import onnxruntime as ort
import torch
from ..checkpoints import file_sha256, load_student
from ..config import cfg

DEFAULT_MODEL = cfg.WEIGHTS_DIR / "resnet18.onnx"


def make_session(model_path: Path, backend="cpu", fp16=False, threads=None):
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    provider_names = {
        "cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    if backend not in provider_names:
        raise ValueError(f"Unknown ONNX Runtime backend: {backend}")
    required = provider_names[backend]
    available = ort.get_available_providers()
    if required not in available:
        raise RuntimeError(f"{required} unavailable; installed providers: {available}")
    providers = [required]
    if backend == "tensorrt":
        cache = cfg.WEIGHTS_DIR / "ort_tensorrt_cache" / ("fp16" if fp16 else "fp32")
        cache.mkdir(parents=True, exist_ok=True)
        providers = [(required, {"trt_engine_cache_enable": True,
                                "trt_engine_cache_path": str(cache), "trt_fp16_enable": fp16})]
    if backend != "cpu":
        providers += [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                      if p in available and p != required]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if required not in session.get_providers():
        raise RuntimeError(f"Requested provider {required} failed to initialize; active: {session.get_providers()}")
    return session


@dataclass
class InferenceBackend:
    name: str
    run: Callable[[np.ndarray], np.ndarray]
    details: dict
    min_batch_size: int = 1
    max_batch_size: int | None = None

    def validate_batch(self, size: int):
        if size < self.min_batch_size or (self.max_batch_size is not None and size > self.max_batch_size):
            raise ValueError(f"{self.name} supports batches {self.min_batch_size}..{self.max_batch_size}; got {size}")


def load_backends(checkpoint: Path, model_path: Path, engines: dict[str, Path], threads=1):
    torch.set_num_threads(threads)
    model = load_student(checkpoint)
    session = make_session(model_path, threads=threads)
    labels = session.get_modelmeta().custom_metadata_map.get("class_labels")
    if labels is not None and tuple(labels.split(",")) != cfg.CLASS_LABELS:
        raise ValueError("ONNX class order differs from the checkpoint convention")
    exported_checkpoint = session.get_modelmeta().custom_metadata_map.get("checkpoint_sha256")
    if exported_checkpoint and exported_checkpoint != file_sha256(checkpoint):
        raise ValueError("ONNX was exported from a different checkpoint; select --checkpoint or re-export")
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    batch_dim = input_meta.shape[0]
    fixed_batch = batch_dim if isinstance(batch_dim, int) else None
    backends = [
        InferenceBackend("PyTorch CPU", lambda array: model(torch.from_numpy(array)).numpy(),
                         {"checkpoint": str(checkpoint), "threads": threads}),
        InferenceBackend("ONNX Runtime CPU",
                         lambda array: session.run([output_meta.name], {input_meta.name: array})[0],
                         {"model": str(model_path), "providers": session.get_providers(), "threads": threads},
                         fixed_batch or 1, fixed_batch),
    ]
    if engines:
        from .tensorrt_runtime import TensorRTEngine
        for precision, path in engines.items():
            manifest_path = path.with_suffix(".json")
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("engine_sha256") and manifest["engine_sha256"] != file_sha256(path):
                    raise ValueError(f"Engine {path} does not match its build manifest")
                if manifest.get("precision") != precision:
                    raise ValueError(f"Engine {path} precision does not match its requested label {precision}")
                external = model_path.with_suffix(model_path.suffix + ".data")
                external_hash = file_sha256(external) if external.is_file() else None
                if (manifest["onnx_sha256"] != file_sha256(model_path)
                        or manifest.get("onnx_external_data_sha256") != external_hash):
                    raise ValueError(f"Engine {path} was built from a different ONNX artifact; rebuild it")
            engine = TensorRTEngine(path)
            backends.append(InferenceBackend(
                f"TensorRT {precision.upper()} GPU", engine.run,
                {"engine": str(path), "min_batch": engine.min_batch_size,
                 "opt_batch": engine.opt_batch_size, "max_batch": engine.max_batch_size,
                 "timing": "host input to host output, including allocation and CUDA copies"},
                engine.min_batch_size, engine.max_batch_size,
            ))
    return backends
