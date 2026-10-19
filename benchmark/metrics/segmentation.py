"""Pixel and instance segmentation metrics (TLS, PathoCell, Lizard).

These accumulators live in the archived ``metrics.pyc`` loaded by
``metrics/__init__.py``. This module is the explicit surface so the package
listing includes segmentation, not only classification / vstain / clinical.
"""

from __future__ import annotations

from metrics import (
    CellAUPRCAccumulator,
    InstanceIoUMatcher,
    InstanceLevelMetricAccumulator,
    PixelPearsonAccumulator,
    SegmentationMetricAccumulator,
    get_instance_level_labels,
    predictions_from_threshold,
    tune_binary_threshold,
)

__all__ = [
    "CellAUPRCAccumulator",
    "InstanceIoUMatcher",
    "InstanceLevelMetricAccumulator",
    "PixelPearsonAccumulator",
    "SegmentationMetricAccumulator",
    "get_instance_level_labels",
    "predictions_from_threshold",
    "tune_binary_threshold",
]
