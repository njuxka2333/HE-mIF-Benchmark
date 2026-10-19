"""Reusable CellViT-style processing for equal-grid ViT feature maps."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn


class CellViTDeconvBlock(nn.Module):
    """The skip-processing block used by the PFMB/CellViT UNETR implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViTFeaturePyramidBuilder(nn.Module):
    """
    Convert four equal-grid ViT maps into CellViT-style decoder-ready skips.

    The first three transformer features pass through three, two, and one
    learned 2x deconvolution blocks respectively. The deepest feature is kept
    on its native patch grid. For an input grid ``G`` the output grids are
    therefore ``8G, 4G, 2G, G``.

    This module contains no task heads, prediction branches, losses, RGB path,
    or decoder fusion. Prefix-token removal, transformer LayerNorm selection,
    and token-to-map reshape remain encoder-extraction responsibilities.
    """

    def __init__(
        self,
        feature_layers: list[str],
        in_channels: list[int],
        *,
        embed_dim: int | None = None,
        drop_rate: float = 0.0,
        skip_dim_11: int | None = None,
        skip_dim_12: int | None = None,
        bottleneck_dim: int | None = None,
        mid_dim: int = 256,
        shallow_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(feature_layers) != 4 or len(in_channels) != 4:
            raise ValueError("ViTFeaturePyramidBuilder requires exactly four feature maps")
        if len(set(feature_layers)) != 4:
            raise ValueError("ViTFeaturePyramidBuilder feature names must be unique")

        self.feature_layers = list(feature_layers)
        self.in_channels = [int(ch) for ch in in_channels]
        self.embed_dim = int(embed_dim or max(self.in_channels))
        self.drop_rate = float(drop_rate)

        if self.embed_dim < 512:
            default_skip_11, default_skip_12, default_bottleneck = 256, 128, 312
        else:
            default_skip_11, default_skip_12, default_bottleneck = 512, 256, 512
        self.skip_dim_11 = int(skip_dim_11 or default_skip_11)
        self.skip_dim_12 = int(skip_dim_12 or default_skip_12)
        self.bottleneck_dim = int(bottleneck_dim or default_bottleneck)
        self.mid_dim = int(mid_dim)
        self.shallow_dim = int(shallow_dim)

        self.channel_projections = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if in_ch == self.embed_dim
                    else nn.Conv2d(in_ch, self.embed_dim, kernel_size=1, bias=False)
                )
                for name, in_ch in zip(self.feature_layers, self.in_channels)
            }
        )

        z1, z2, z3, z4 = self.feature_layers
        self.skip_processors = nn.ModuleDict(
            {
                z1: nn.Sequential(
                    CellViTDeconvBlock(
                        self.embed_dim,
                        self.skip_dim_11,
                        dropout=self.drop_rate,
                    ),
                    CellViTDeconvBlock(
                        self.skip_dim_11,
                        self.skip_dim_12,
                        dropout=self.drop_rate,
                    ),
                    CellViTDeconvBlock(
                        self.skip_dim_12,
                        self.shallow_dim,
                        dropout=self.drop_rate,
                    ),
                ),
                z2: nn.Sequential(
                    CellViTDeconvBlock(
                        self.embed_dim,
                        self.skip_dim_11,
                        dropout=self.drop_rate,
                    ),
                    CellViTDeconvBlock(
                        self.skip_dim_11,
                        self.mid_dim,
                        dropout=self.drop_rate,
                    ),
                ),
                z3: nn.Sequential(
                    CellViTDeconvBlock(
                        self.embed_dim,
                        self.bottleneck_dim,
                        dropout=self.drop_rate,
                    )
                ),
                z4: nn.Identity(),
            }
        )
        self.output_channels = [
            self.shallow_dim,
            self.mid_dim,
            self.bottleneck_dim,
            self.embed_dim,
        ]
        self.last_trace: dict[str, dict[str, list[int]]] = {}

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        missing = [name for name in self.feature_layers if name not in features]
        if missing:
            raise KeyError(f"ViTFeaturePyramidBuilder missing feature maps: {missing}")

        selected = [features[name] for name in self.feature_layers]
        if any(feature.ndim != 4 for feature in selected):
            shapes = {name: list(features[name].shape) for name in self.feature_layers}
            raise ValueError(f"ViTFeaturePyramidBuilder expects BCHW maps, got {shapes}")
        observed_channels = [int(feature.shape[1]) for feature in selected]
        if observed_channels != self.in_channels:
            shapes = {name: list(features[name].shape) for name in self.feature_layers}
            raise ValueError(
                f"ViTFeaturePyramidBuilder features {self.feature_layers} require "
                f"channels {self.in_channels}, got shapes={shapes}"
            )
        grids = [tuple(feature.shape[-2:]) for feature in selected]
        if len(set(grids)) != 1:
            raise ValueError(
                "ViTFeaturePyramidBuilder requires four equal-grid ViT maps, "
                f"got {dict(zip(self.feature_layers, grids))}"
            )

        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name in self.feature_layers:
            projected = self.channel_projections[name](features[name])
            outputs[name] = self.skip_processors[name](projected)

        grid_h, grid_w = grids[0]
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
                f"ViTFeaturePyramidBuilder features {self.feature_layers} require "
                f"8G/4G/2G/G grids {expected_grids}, got shapes={output_shapes}"
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
