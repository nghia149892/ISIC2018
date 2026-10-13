"""Console reporting and portable JSON benchmark artifacts."""
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import torch


def environment_info() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "tensorrt"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}


def print_batch(result: dict) -> None:
    print(f"\n=== Batch size {result['batch_size']} ===")
    for name, backend in result["backends"].items():
        latency = backend["latency"]
        print(f"{name:<20} mean {latency['mean_batch_ms']:.2f} ms/batch | "
              f"p95 {latency['p95_batch_ms']:.2f} ms/batch | "
              f"{latency['images_per_second']:.2f} images/s | "
              f"{backend['speedup_vs_pytorch_cpu']:.2f}x PyTorch CPU")
        agreement = backend["agreement_with_pytorch"]
        print(f"  class agreement {agreement['predicted_class_agreement']:.2%} | "
              f"max logit diff {agreement['max_abs_diff']:.8g}")
        if "classification" in backend:
            metrics = backend["classification"]
            print(f"  accuracy {metrics['accuracy']:.2%} | macro-F1 {metrics['macro_f1']:.4f}")
            print("  Per-class recall: " + " | ".join(
                f"{label} {data['recall']:.1%} (n={data['support']})"
                if data["recall"] is not None else f"{label} no examples"
                for label, data in metrics["per_class_recall"].items()
            ))


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved benchmark report: {path}")


def print_scaling(results: list[dict]) -> None:
    baseline = next((r for r in results if r["batch_size"] == 1), None)
    if baseline is None or len(results) < 2:
        return
    print("\nThroughput scaling relative to batch size 1:")
    for name in baseline["backends"]:
        initial = baseline["backends"][name]["latency"]["images_per_second"]
        print(f"  {name}: " + " | ".join(
            f"batch {r['batch_size']}: {r['backends'][name]['latency']['images_per_second'] / initial:.2f}x"
            for r in results
        ))
