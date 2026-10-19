"""Unified cellseg-style UNETR decoder for frozen PathoCell encoder features."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.upsampler import EncoderUpsampler
from heads.unetr import Conv2DBlock

DEFAULT_DECODER_CHANNELS = (128, 256, 512, 512)


def _compute_reduction(height: int, grid_h: int) -> float:
    if grid_h <= 0:
        raise ValueError(f"Invalid feature grid height: {grid_h}")
    return float(height) / float(grid_h)


def _up_factors_from_grids(grids: Sequence[tuple[int, int]]) -> list[int]:
    if len(grids) < 2:
        raise ValueError(f"Need at least two feature grids, got {grids}")
    factors: list[int] = []
    for low, high in zip(grids[:-1], grids[1:]):
        ratio_h = high[0] / low[0]
        ratio_w = high[1] / low[1]
        if abs(ratio_h - ratio_w) > 0.001:
            raise ValueError(f"Non-isotropic pyramid {low} -> {high}")
        rounded = int(round(ratio_h))
        if abs(ratio_h - rounded) > 0.001:
            raise ValueError(
                f"Non-integer upsample ratio {ratio_h} for {low} -> {high}"
            )
        factors.append(max(1, rounded))
    return factors


class CellsegDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.up_factor = int(up_factor)
        if self.up_factor == 1:
            self.upsample = nn.Identity()
        elif self.up_factor == 2:
            self.upsample = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )
        else:
            raise ValueError(f"Unsupported decoder up_factor={self.up_factor}")

        merge_channels = in_channels + skip_channels
        self.merge = nn.Sequential(
            Conv2DBlock(merge_channels, out_channels, dropout=drop_rate),
            Conv2DBlock(out_channels, out_channels, dropout=drop_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        up_factor: int | None = None,
    ) -> torch.Tensor:
        factor = self.up_factor if up_factor is None else int(up_factor)
        if factor == 1:
            upsampled = x
        elif factor == 2:
            upsampled = self.upsample(x)
        else:
            raise ValueError(f"Unsupported decoder up_factor={factor}")

        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return self.merge(torch.cat([upsampled, skip], dim=1))


class CellsegStemSkip(nn.Module):
    """Full-resolution stem skip (cellseg StemSkip, cat merge).

    ``in_channels`` mirrors upstream ``cellseg_models_pytorch.decoders.long_skips.
    stem_skip.StemSkip(in_channels: int = 3)``. It stays 3 for the RGB frozen-encoder
    route; the virtual-stain route passes the generated mIF channel count so the stem
    consumes the marker stack instead of H&E.
    """

    def __init__(
        self,
        out_channels: int,
        drop_rate: float,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = nn.Sequential(
            Conv2DBlock(self.in_channels, out_channels // 2, dropout=drop_rate),
            Conv2DBlock(out_channels // 2, out_channels, dropout=drop_rate),
        )
        self.merge = nn.Sequential(
            Conv2DBlock(out_channels * 2, out_channels, dropout=drop_rate),
            Conv2DBlock(out_channels, out_channels, dropout=drop_rate),
        )

    def forward(self, image: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        stem_feat = self.stem(image)
        if stem_feat.shape[-2:] != decoder_feat.shape[-2:]:
            decoder_feat = F.interpolate(
                decoder_feat,
                size=stem_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.merge(torch.cat([stem_feat, decoder_feat], dim=1))


class CellsegUNETR(nn.Module):
    """
    Unified cellseg-style UNETR for PathoCell.

    ViT (equal-grid skips): ``EncoderUpsampler`` builds the spatial pyramid, then the
    shared U-Net decoder fuses bottleneck-first features.

    CNN / U-Net encoders: 1×1 projections align native skips to ``decoder_channels``,
    then the same decoder path runs with reduction-derived up-factors.

    ``decoder_channels`` are ordered shallow → deep (z1 … z4).
    """

    def __init__(
        self,
        feature_layers: list[str],
        in_channels: list[int],
        num_classes: int,
        decoder_channels: list[int] | None = None,
        drop_rate: float = 0.0,
        stem_in_channels: int = 3,
    ) -> None:
        super().__init__()
        if len(feature_layers) != len(in_channels):
            raise ValueError(
                "feature_layers and in_channels must have the same length"
            )
        if len(feature_layers) not in {4, 5}:
            raise ValueError(
                "CellsegUNETR currently supports four or five skip levels"
            )

        self.feature_layers = list(feature_layers)
        self.in_channels = [int(ch) for ch in in_channels]
        self.num_classes = int(num_classes)
        self.drop_rate = float(drop_rate)
        self.use_z0 = True

        if decoder_channels is None:
            decoder_channels = list(DEFAULT_DECODER_CHANNELS)
        if len(decoder_channels) != len(feature_layers):
            raise ValueError("decoder_channels must match feature_layers length")
        self.decoder_channels = [int(ch) for ch in decoder_channels]
        self._bottleneck_channels = list(reversed(self.decoder_channels))

        raw_info = tuple(
            {"num_chs": ch, "reduction": 1.0} for ch in self.in_channels
        )
        self.encoder_upsampler = EncoderUpsampler(
            feature_info=raw_info[::-1],
            out_channels=tuple(self._bottleneck_channels),
        )

        self.skip_projections = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if in_ch == out_ch
                    else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                )
                for name, in_ch, out_ch in zip(
                    self.feature_layers,
                    self.in_channels,
                    self.decoder_channels,
                )
            }
        )

        bottleneck_first = self._bottleneck_channels
        self.decoder_stages = nn.ModuleList(
            [
                CellsegDecoderStage(
                    in_channels=bottleneck_first[idx],
                    skip_channels=bottleneck_first[idx + 1],
                    out_channels=bottleneck_first[idx + 1],
                    up_factor=2,
                    drop_rate=self.drop_rate,
                )
                for idx in range(len(bottleneck_first) - 1)
            ]
        )
        self.stem_in_channels = int(stem_in_channels)
        self.stem_skip = CellsegStemSkip(
            out_channels=self.decoder_channels[0],
            drop_rate=self.drop_rate,
            in_channels=self.stem_in_channels,
        )
        self.classifier = nn.Conv2d(
            self.decoder_channels[0],
            self.num_classes,
            kernel_size=1,
        )

    @staticmethod
    def _equal_grids(
        features: Mapping[str, torch.Tensor],
        layer_names: list[str],
    ) -> bool:
        grids = {
            tuple(features[name].shape[-2:]) for name in layer_names
        }
        return len(grids) == 1

    def _encode_features(
        self,
        features: Mapping[str, torch.Tensor],
        image_hw: tuple[int, int],
    ) -> tuple[tuple[torch.Tensor, ...], list[float]]:
        ordered = [features[name] for name in self.feature_layers]
        grids = [tuple(f.shape[-2:]) for f in ordered]
        height, width = image_hw
        reductions = [
            _compute_reduction(height, grid_h) for grid_h, grid_w in grids
        ]

        if self._equal_grids(features, self.feature_layers):
            bottleneck_first = tuple(reversed(ordered))
            encoded = self.encoder_upsampler(bottleneck_first)
            out_reductions = [
                float(info["reduction"])
                for info in self.encoder_upsampler.feature_info
            ]
            return encoded, out_reductions

        projected = [
            self.skip_projections[name](features[name])
            for name in self.feature_layers
        ]
        encoded = tuple(reversed(projected))
        out_reductions = list(reversed(reductions))
        return encoded, out_reductions

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        image: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        missing = [name for name in self.feature_layers if name not in features]
        if missing:
            raise KeyError(f"CellsegUNETR missing features {missing}")
        if image is None:
            raise ValueError(
                "CellsegUNETR requires an RGB image tensor for stem skip"
            )

        image_hw = tuple(image.shape[-2:])
        encoded, _reductions = self._encode_features(features, image_hw)
        grids = [tuple(tensor.shape[-2:]) for tensor in encoded]
        up_factors = _up_factors_from_grids(grids)
        if len(up_factors) != len(self.decoder_stages):
            raise ValueError(
                f"Decoder stage count {len(self.decoder_stages)} "
                f"does not match up_factors {up_factors} from grids {grids}"
            )

        x = encoded[0]
        skips = encoded[1:]
        for stage, skip, up_factor in zip(
            self.decoder_stages, skips, up_factors
        ):
            x = stage(x, skip, up_factor=up_factor)

        x = self.stem_skip(image, x)
        logits = self.classifier(x)
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits
