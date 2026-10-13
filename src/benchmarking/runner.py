"""Warmup and timed inference. Preprocessing stays outside the timer."""
import time
from pathlib import Path
import numpy as np
import torch
from ..data.transforms import prepare_batch
from ..config import cfg
from .metrics import agreement_metrics, classification_metrics, latency_metrics


def benchmark_batch(backends, images: list[Path], batch_size: int,
                    warmup=10, runs=50, repeated=False, targets=None) -> dict:
    groups = ([images[0]] * batch_size,) if repeated else [
        images[i:i + batch_size] for i in range(0, len(images), batch_size)
    ]
    # Also validate the final partial batch against the engine's minimum shape.
    for backend in backends:
        for size in {len(group) for group in groups}:
            backend.validate_batch(size)
    warmup_array = prepare_batch(groups[0])
    times = {b.name: [] for b in backends}
    counts = {b.name: [] for b in backends}
    outputs = {b.name: [] for b in backends}
    with torch.inference_mode():
        for backend in backends:
            for _ in range(warmup):
                backend.run(warmup_array)
        iterations = runs if repeated else len(groups)
        for index in range(iterations):
            array = warmup_array if repeated else prepare_batch(groups[index])
            # Alternate the order to reduce a persistent backend-order bias.
            order = backends if index % 2 == 0 else list(reversed(backends))
            for backend in order:
                start = time.perf_counter()
                output = backend.run(array)
                elapsed = (time.perf_counter() - start) * 1000
                if output.shape != (len(array), cfg.NUM_CLASSES) or not np.isfinite(output).all():
                    raise ValueError(f"{backend.name} returned invalid logits: {output.shape}")
                times[backend.name].append(elapsed)
                counts[backend.name].append(len(array))
                if not repeated or index == iterations - 1:
                    outputs[backend.name].append(output)
            if not repeated and ((index + 1) % max(1, 100 // batch_size) == 0 or index == iterations - 1):
                print(f"  batch {batch_size}: processed {min((index + 1) * batch_size, len(images))}/{len(images)}")
    logits = {name: np.concatenate(values) for name, values in outputs.items()}
    reference = logits["PyTorch CPU"]
    results = {}
    for backend in backends:
        name = backend.name
        results[name] = {"latency": latency_metrics(times[name], counts[name]),
                         "agreement_with_pytorch": agreement_metrics(reference, logits[name]),
                         "details": backend.details}
        if targets is not None:
            evaluation_targets = np.repeat(targets[:1], batch_size) if repeated else targets
            results[name]["classification"] = classification_metrics(logits[name], evaluation_targets)
    baseline = results["PyTorch CPU"]["latency"]["images_per_second"]
    for result in results.values():
        result["speedup_vs_pytorch_cpu"] = result["latency"]["images_per_second"] / baseline
    return {"batch_size": batch_size, "evaluated_images": len(reference), "backends": results}
