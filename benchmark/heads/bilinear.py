"""Bilinear 8G/4G/2G/G pyramid for equal-grid ViT maps (no learned deconv)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_BILINEAR_SCALE_FACTORS = (8, 4, 2, 1)


class BilinearViTFeaturePyramidBuilder(nn.Module):
    """Resize four equal-grid ViT maps to 8G / 4G / 2G / G with bilinear upsample."""

    def __init__(
        self,
        feature_layers: list[str],
        in_channels: list[int],
        **_: object,
    ) -> None:
        super().__init__()
        if len(feature_layers) != 4 or len(in_channels) != 4:
            raise ValueError(
                "BilinearViTFeaturePyramidBuilder requires exactly four feature maps"
            )
        if len(set(feature_layers)) != 4:
            raise ValueError(
                "BilinearViTFeaturePyramidBuilder feature names must be unique"
            )
        self.feature_layers = list(feature_layers)
        self.in_channels = [int(ch) for ch in in_channels]
        self.output_channels = list(self.in_channels)
        self.embed_dim = max(self.in_channels)
        self.last_trace: dict = {}

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        missing = [name for name in self.feature_layers if name not in features]
        if missing:
            raise KeyError(
                f"BilinearViTFeaturePyramidBuilder missing feature maps: {missing}"
            )

        selected = [features[name] for name in self.feature_layers]
        if any(feature.ndim != 4 for feature in selected):
            shapes = {name: list(features[name].shape) for name in self.feature_layers}
            raise ValueError(
                f"BilinearViTFeaturePyramidBuilder expects BCHW maps, got {shapes}"
            )
        observed_channels = [int(feature.shape[1]) for feature in selected]
        if observed_channels != self.in_channels:
            shapes = {name: list(features[name].shape) for name in self.feature_layers}
            raise ValueError(
                f"BilinearViTFeaturePyramidBuilder features {self.feature_layers} "
                f"require channels {self.in_channels}, got shapes={shapes}"
            )
        grids = [tuple(feature.shape[-2:]) for feature in selected]
        if len(set(grids)) != 1:
            raise ValueError(
                "BilinearViTFeaturePyramidBuilder requires four equal-grid ViT maps, "
                f"got {dict(zip(self.feature_layers, grids))}"
            )

        grid_h, grid_w = grids[0]
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, feature, scale in zip(
            self.feature_layers, selected, _BILINEAR_SCALE_FACTORS
        ):
            target_hw = (grid_h * scale, grid_w * scale)
            if tuple(feature.shape[-2:]) == target_hw:
                outputs[name] = feature
            else:
                outputs[name] = F.interpolate(
                    feature,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

        expected_grids = [
            (8 * grid_h, 8 * grid_w),
            (4 * grid_h, 4 * grid_w),
            (2 * grid_h, 2 * grid_w),
            (grid_h, grid_w),
        ]
        output_grids = [tuple(outputs[name].shape[-2:]) for name in self.feature_layers]
        if output_grids != expected_grids:
            output_shapes = {
                name: list(outputs[name].shape) for name in self.feature_layers
            }
            raise ValueError(
                f"BilinearViTFeaturePyramidBuilder features {self.feature_layers} "
                f"require 8G/4G/2G/G grids {expected_grids}, got shapes={output_shapes}"
            )

        self.last_trace = {
            "input_shapes": {
                name: list(features[name].shape) for name in self.feature_layers
            },
            "output_shapes": {
                name: list(outputs[name].shape) for name in self.feature_layers
            },
        }
        return outputs
