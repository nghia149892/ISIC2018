"""Classification, latency, and numerical agreement metrics."""
import numpy as np
from ..config import cfg


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    if logits.shape != (len(targets), cfg.NUM_CLASSES) or len(targets) == 0:
        raise ValueError("Expected nonempty logits with shape (number of targets, number of classes)")
    if not np.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits")
    predictions = logits.argmax(axis=1)
    recalls, f1_scores = {}, []
    for index, label in enumerate(cfg.CLASS_LABELS):
        actual, predicted = targets == index, predictions == index
        tp = int(np.sum(actual & predicted))
        fp, fn = int(np.sum(~actual & predicted)), int(np.sum(actual & ~predicted))
        denominator = 2 * tp + fp + fn
        f1_scores.append(2 * tp / denominator if denominator else 0.0)
        recalls[label] = {"support": int(actual.sum()),
                         "recall": tp / int(actual.sum()) if actual.any() else None}
    return {"accuracy": float(np.mean(predictions == targets)),
            "macro_f1": float(np.mean(f1_scores)), "per_class_recall": recalls}


def latency_metrics(times_ms: list[float], counts: list[int]) -> dict:
    if not times_ms or len(times_ms) != len(counts) or min(counts) < 1:
        raise ValueError("Latency samples and image counts must be nonempty and aligned")
    duration = sum(times_ms)
    return {"mean_batch_ms": float(np.mean(times_ms)),
            "median_batch_ms": float(np.median(times_ms)),
            "p95_batch_ms": float(np.percentile(times_ms, 95)),
            "images_per_second": sum(counts) / (duration / 1000),
            "ms_per_image": duration / sum(counts),
            "timed_batches": len(times_ms), "timed_images": sum(counts)}


def agreement_metrics(reference: np.ndarray, output: np.ndarray) -> dict:
    if output.shape != reference.shape or not np.isfinite(output).all():
        raise ValueError("Backend output shape/values do not match classifier requirements")
    return {"max_abs_diff": float(np.max(np.abs(reference - output))),
            "predicted_class_agreement": float(np.mean(reference.argmax(1) == output.argmax(1))),
            "allclose": bool(np.allclose(reference, output, rtol=1e-4, atol=1e-5))}
