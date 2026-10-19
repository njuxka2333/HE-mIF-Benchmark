"""Classification, survival, segmentation, and virtual-staining metrics.

Archived ``metrics.pyc`` is loaded here (pixel / instance segmentation
accumulators). Explicit siblings:

* ``segmentation`` — Dice / IoU / instance F1
* ``vstain`` — Cell-niches shared protocol + Table-4 + significance
* ``classification`` — DeLong / case bootstrap
* ``clinical`` — C-index / fold CIs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

from training._bytecode_loader import install_bytecode_shim

_TRAINING_ROOT = Path(__file__).resolve().parents[1] / "training"
install_bytecode_shim(
    "metrics",
    _TRAINING_ROOT,
    globals(),
    min_bytecode_size=2500,
)

_orig_linear_probe_metrics = linear_probe_metrics  # noqa: F821


def linear_probe_metrics(labels, predictions, probabilities, num_classes, *, prefix="lin_", class_names=None):
    import numpy as np
    from sklearn.metrics import average_precision_score

    metrics = _orig_linear_probe_metrics(
        labels,
        predictions,
        probabilities,
        num_classes,
        prefix=prefix,
        class_names=class_names,
    )
    key = f"{prefix}auprc"
    try:
        y = np.asarray(labels)
        prob = np.asarray(probabilities, dtype=np.float64)
        if prob.ndim == 1:
            score = float(average_precision_score(y, prob))
        else:
            score = float(average_precision_score(y, prob, average="macro"))
    except Exception:
        score = float("nan")
    metrics[key] = score
    return metrics
