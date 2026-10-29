"""Build a layer-selective wrapper around a loaded benchmark encoder."""

from __future__ import annotations

import torch.nn as nn

from utils.experiment_config import ExtractionTask
from feature_extraction.layers import (
    compute_model_layers,
    format_feature_output,
    select_layers,
)


class LayerFeatureExtractor(nn.Module):
    """Extract configured layer(s) from a benchmark encoder wrapper."""

    def __init__(self, wrapper: nn.Module, task: ExtractionTask):
        super().__init__()
        self.wrapper = wrapper
        self.task = task
        self.layer_names = list(task.target_features)

    def forward(self, x):
        all_layers = compute_model_layers(
            self.task.model,
            self.wrapper,
            x,
            requested_layers=self.layer_names,
        )
        selected = select_layers(all_layers, self.layer_names)
        if self.task.dense_interface == "default":
            from heads.interfaces import validate_dense_runtime_features

            validate_dense_runtime_features(self.task.model, selected)
        return format_feature_output(selected, self.task.use_gap)


def build_feature_extractor(base_model: nn.Module, task: ExtractionTask) -> nn.Module:
    if not task.target_features:
        raise ValueError(
            "experiment target_features must list model layer names "
            "(see feature_extraction/LAYERS.md)"
        )
    return LayerFeatureExtractor(base_model, task)
