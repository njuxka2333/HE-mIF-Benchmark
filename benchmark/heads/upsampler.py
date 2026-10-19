"""Vendored cellseg_models.pytorch EncoderUpsampler (no timm dependency)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["EncoderUpsampler", "FeatUpsampleBlock"]


class FeatUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.scale_factor = scale_factor
        self.out_channels = out_channels

        if isinstance(scale_factor, int):
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 ** (scale_factor - 1),
                stride=2 ** (scale_factor - 1),
                padding=0,
                output_padding=0,
            )
        else:
            self.up = nn.Upsample(
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=True,
            )

        self.conv_block = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_block(x)
        return x


class EncoderUpsampler(nn.Module):
    def __init__(
        self,
        feature_info: Tuple[dict, ...],
        out_channels: Tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(out_channels) != len(feature_info):
            raise ValueError(
                "`out_channels` must have the same len as the `backbone.feature_info.`"
                f"Got {len(out_channels)} and {len(feature_info)} respectively."
            )

        self.out_channels = out_channels
        self.feature_info = []

        self.bottleneck = nn.Conv2d(
            in_channels=feature_info[0]["num_chs"],
            out_channels=self.out_channels[0],
            kernel_size=1,
        )

        self.feature_info.append(
            {
                "num_chs": self.out_channels[0],
                "module": "bottleneck",
                "reduction": float(feature_info[0]["reduction"]),
            }
        )

        self.up_blocks = nn.ModuleDict()
        n_up_blocks = list(range(1, len(self.out_channels)))
        for i, (out_chls, finfo, n_blocks) in enumerate(
            zip(self.out_channels[1:], feature_info[1:], n_up_blocks)
        ):
            up_blocks = []
            squeeze_rates = list(range(n_blocks))[::-1]

            for j, squeeze_ratio in zip(range(n_blocks), squeeze_rates):
                if j == 0:
                    in_channels = finfo["num_chs"]
                else:
                    in_channels = up.out_channels  # noqa: F821

                up = FeatUpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_chls * (2**squeeze_ratio),
                    scale_factor=2,
                )
                up_blocks.append(up)

            self.feature_info.append(
                {
                    "num_chs": out_chls,
                    "module": f"up{i + 1}",
                    "reduction": finfo["reduction"] / 2**n_blocks,
                }
            )
            self.up_blocks[f"up{i + 1}"] = nn.Sequential(*up_blocks)

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        up_feat = self.bottleneck(feats[0])
        intermediate_features = [up_feat]

        for i, feat in enumerate(feats[1:]):
            up_feat = self.up_blocks[f"up{i + 1}"](feat)
            intermediate_features.append(up_feat)

        return tuple(intermediate_features)
