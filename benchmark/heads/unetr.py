"""PFMB / CellViT-style UNETR decoder for frozen multi-level encoder features."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.vit_pyramid import CellViTDeconvBlock


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
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


Deconv2DBlock = CellViTDeconvBlock


def forms_native_encoder_pyramid(grids: list[tuple[int, int]]) -> bool:
    """
    Return True when four or five skips form a dyadic largest-to-smallest pyramid.

    ViT models (all layers on the same patch grid) return False and use the
    original PFMB uniform-grid decoder. CNN / U-Net encoders with 2× spacing
    between consecutive skips return True and fuse at native resolutions.
    """
    if len(grids) not in {4, 5}:
        return False
    if len({tuple(g) for g in grids}) == 1:
        return False
    for larger, smaller in zip(grids[:-1], grids[1:]):
        lh, lw = int(larger[0]), int(larger[1])
        sh, sw = int(smaller[0]), int(smaller[1])
        if lh != sh * 2 or lw != sw * 2:
            return False
    return True


def classify_encoder_pyramid(grids: list[tuple[int, int]]) -> str:
    """Classify canonical dense skips for PFMB routing."""
    if len(grids) not in {4, 5}:
        raise ValueError(f"UNETR requires four or five feature grids, got {grids}")
    if len({tuple(grid) for grid in grids}) == 1:
        if len(grids) != 4:
            raise ValueError(f"PFMB uniform-grid routing requires four features, got {grids}")
        return "uniform"
    if forms_native_encoder_pyramid(grids):
        return "native5" if len(grids) == 5 else "native"
    if (
        len(grids) == 4
        and grids[0][0] == 2 * grids[1][0]
        and grids[0][1] == 2 * grids[1][1]
        and grids[1][0] == 2 * grids[2][0]
        and grids[1][1] == 2 * grids[2][1]
        and grids[2] == grids[3]
    ):
        return "deepest_plateau"
    raise ValueError(
        "UNETR feature grids are neither four equal ViT maps, a four/five-level "
        f"2x native pyramid, nor the HEMIT deepest plateau: {grids}."
    )


class PfmbUNETR(nn.Module):
    """
    Decoder-only PFMB UNETR head for frozen encoder features.

    Expects four skip feature maps (z1-z4) plus RGB input (z0).
    When backbone skips use different channel widths, 1x1 projections align
    them to ``embed_dim`` before the PFMB decoder path.

    Routing:
    - **Uniform grid (ViT / UNI2 / MIPHEI):** all skips are resized to z4's
      spatial size, then the reference PFMB deconv pyramid is applied.
    - **Preprocessed ViT pyramid:** consumes an external 8×/4×/2×/1× pyramid
      and starts directly at PFMB decoder fusion.
    - **Native encoder pyramid (CNN / U-Net):** when z1-z4 already form a 2×
      dyadic pyramid, skips keep their native H×W and are fused during the
      learned upsampling path (no downsample-to-z4).
    """

    def __init__(
        self,
        feature_layers: list[str],
        in_channels: list[int],
        num_classes: int,
        embed_dim: int | None = None,
        drop_rate: float = 0.0,
        preprocessed_vit: bool = False,
        use_z0: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_layers) not in {4, 5} or len(in_channels) != len(feature_layers):
            raise ValueError("PfmbUNETR requires four or five feature layers")
        if preprocessed_vit and len(feature_layers) != 4:
            raise ValueError("PfmbUNETR preprocessed ViT mode requires exactly four layers")

        self.feature_layers = list(feature_layers)
        if not use_z0:
            raise ValueError("PfmbUNETR requires the decoder-side RGB z0 branch")
        self.use_z0 = True
        self.drop_rate = float(drop_rate)
        self.num_classes = int(num_classes)
        self.preprocessed_vit = bool(preprocessed_vit)

        if embed_dim is None:
            embed_dim = max(int(ch) for ch in in_channels)
        self.embed_dim = int(embed_dim)

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        if self.preprocessed_vit:
            expected_channels = [128, 256, self.bottleneck_dim, self.embed_dim]
            if [int(channel) for channel in in_channels] != expected_channels:
                raise ValueError(
                    "PfmbUNETR preprocessed ViT pyramid expects channels "
                    f"{expected_channels}, got {list(in_channels)}"
                )
            self.skip_projections = nn.ModuleDict(
                {name: nn.Identity() for name in self.feature_layers}
            )
        else:
            self.skip_projections = nn.ModuleDict(
                {
                    name: (
                        nn.Identity()
                        if int(in_ch) == self.embed_dim
                        else nn.Conv2d(int(in_ch), self.embed_dim, kernel_size=1, bias=False)
                    )
                    for name, in_ch in zip(self.feature_layers, in_channels)
                }
            )

        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )

        if self.preprocessed_vit:
            # The external ViTFeaturePyramidBuilder already produced these skips.
            self.decoder1 = nn.Identity()
            self.decoder2 = nn.Identity()
            self.decoder3 = nn.Identity()
            self.skip_adapter3 = nn.Identity()
            self.skip_adapter2 = nn.Identity()
            self.skip_adapter1 = nn.Identity()
        else:
            self.decoder1 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
                Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
            )
            self.decoder2 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
            )
            self.decoder3 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
            )
            # Native-pyramid path: channel adapters at each skip's native resolution.
            self.skip_adapter3 = Conv2DBlock(
                self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate
            )
            self.skip_adapter2 = Conv2DBlock(self.embed_dim, 256, dropout=self.drop_rate)
            self.skip_adapter1 = Conv2DBlock(self.embed_dim, 128, dropout=self.drop_rate)
        if len(self.feature_layers) == 5:
            self.skip_adapter0 = Conv2DBlock(
                self.embed_dim, 64, dropout=self.drop_rate
            )
            self.feature0_merge = Conv2DBlock(128, 64, dropout=self.drop_rate)
        else:
            self.skip_adapter0 = nn.Identity()
            self.feature0_merge = nn.Identity()
        self.branch_decoder = self._create_upsampling_branch(self.num_classes)

    def _create_upsampling_branch(self, num_classes: int) -> nn.Sequential:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        return nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

    def _project_skip_channels(
        self,
        features: Mapping[str, torch.Tensor],
        layer_name: str,
    ) -> torch.Tensor:
        if layer_name not in features:
            raise KeyError(f"PfmbUNETR missing feature layer '{layer_name}'")
        return self.skip_projections[layer_name](features[layer_name])

    def _project_skip_to_grid(
        self,
        features: Mapping[str, torch.Tensor],
        layer_name: str,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        x = self._project_skip_channels(features, layer_name)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _align_decoder_to_skip(
        decoder_feat: torch.Tensor,
        skip_feat: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_feat.shape[-2:] == skip_feat.shape[-2:]:
            return decoder_feat
        if (
            decoder_feat.shape[-2] < skip_feat.shape[-2]
            and decoder_feat.shape[-1] < skip_feat.shape[-1]
        ):
            return F.interpolate(
                decoder_feat,
                size=skip_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return F.interpolate(
            skip_feat,
            size=decoder_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _merge_z0(
        self,
        branch_decoder: nn.Sequential,
        z0: torch.Tensor | None,
        b1: torch.Tensor,
    ) -> torch.Tensor:
        if z0 is None:
            raise ValueError("PfmbUNETR requires an RGB image tensor for decoder-side z0")
        b0 = self.decoder0(z0)
        if b0.shape[-2:] != b1.shape[-2:]:
            b1 = F.interpolate(b1, size=b0.shape[-2:], mode="bilinear", align_corners=False)
        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

    def _forward_upsample_uniform(
        self,
        z0: torch.Tensor | None,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """PFMB reference path for ViT-style uniform patch grids."""
        branch_decoder = self.branch_decoder
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        return self._merge_z0(branch_decoder, z0, b1)

    def _forward_preprocessed_vit_pyramid(
        self,
        z0: torch.Tensor | None,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse an external 8×/4×/2×/1× ViT pyramid without rebuilding skips."""
        grids = [tuple(tensor.shape[-2:]) for tensor in (z1, z2, z3, z4)]
        if not forms_native_encoder_pyramid(grids):
            raise ValueError(
                "PfmbUNETR preprocessed ViT input must be an 8x/4x/2x/1x pyramid, "
                f"got {grids}"
            )
        branch_decoder = self.branch_decoder
        b4 = branch_decoder.bottleneck_upsampler(z4)
        if b4.shape[-2:] != z3.shape[-2:]:
            raise ValueError(
                f"PFMB z4 upsample {tuple(b4.shape[-2:])} does not match z3 "
                f"{tuple(z3.shape[-2:])}"
            )
        b3 = branch_decoder.decoder3_upsampler(torch.cat([z3, b4], dim=1))
        if b3.shape[-2:] != z2.shape[-2:]:
            raise ValueError(
                f"PFMB decoder3 output {tuple(b3.shape[-2:])} does not match z2 "
                f"{tuple(z2.shape[-2:])}"
            )
        b2 = branch_decoder.decoder2_upsampler(torch.cat([z2, b3], dim=1))
        if b2.shape[-2:] != z1.shape[-2:]:
            raise ValueError(
                f"PFMB decoder2 output {tuple(b2.shape[-2:])} does not match z1 "
                f"{tuple(z1.shape[-2:])}"
            )
        b1 = branch_decoder.decoder1_upsampler(torch.cat([z1, b2], dim=1))
        return self._merge_z0(branch_decoder, z0, b1)

    def _forward_upsample_native_pyramid(
        self,
        z0: torch.Tensor | None,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """PFMB upsampling branch with native-resolution skip fusion (CNN / U-Net)."""
        branch_decoder = self.branch_decoder
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b4 = self._align_decoder_to_skip(b4, z3)
        b3 = self.skip_adapter3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))

        b3 = self._align_decoder_to_skip(b3, z2)
        b2 = self.skip_adapter2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))

        b2 = self._align_decoder_to_skip(b2, z1)
        b1 = self.skip_adapter1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        return self._merge_z0(branch_decoder, z0, b1)

    def _forward_upsample_native_pyramid5(
        self,
        z0: torch.Tensor | None,
        shallow: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse a five-level native pyramid while retaining its shallowest skip."""
        branch_decoder = self.branch_decoder
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = branch_decoder.decoder3_upsampler(
            torch.cat([self.skip_adapter3(z3), b4], dim=1)
        )
        b2 = branch_decoder.decoder2_upsampler(
            torch.cat([self.skip_adapter2(z2), b3], dim=1)
        )
        b1 = branch_decoder.decoder1_upsampler(
            torch.cat([self.skip_adapter1(z1), b2], dim=1)
        )
        shallow = self.skip_adapter0(shallow)
        if b1.shape[-2:] != shallow.shape[-2:]:
            shapes = [tuple(t.shape) for t in (shallow, z1, z2, z3, z4)]
            raise ValueError(
                "PfmbUNETR five-level native decoder failed to reach the shallow "
                f"feature grid; observed projected shapes={shapes}, decoder={tuple(b1.shape)}"
            )
        b0 = self.feature0_merge(torch.cat([shallow, b1], dim=1))
        return self._merge_z0(branch_decoder, z0, b0)

    def _forward_upsample_deepest_plateau(
        self,
        z0: torch.Tensor | None,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse HEMIT H1 without resizing or pooling its equal-grid z3/z4 pair."""
        if self.embed_dim != self.bottleneck_dim:
            raise ValueError(
                "PfmbUNETR deepest-plateau routing requires embed_dim == "
                f"bottleneck_dim, got {self.embed_dim} and {self.bottleneck_dim}"
            )
        if z3.shape[-2:] != z4.shape[-2:]:
            raise ValueError(
                "PfmbUNETR HEMIT deepest plateau requires unchanged equal-grid "
                f"fused_z3/z4, got {tuple(z3.shape)} and {tuple(z4.shape)}"
            )
        branch_decoder = self.branch_decoder
        b3 = branch_decoder.decoder3_upsampler(
            torch.cat([self.skip_adapter3(z3), z4], dim=1)
        )
        b2 = branch_decoder.decoder2_upsampler(
            torch.cat([self.skip_adapter2(z2), b3], dim=1)
        )
        b1 = branch_decoder.decoder1_upsampler(
            torch.cat([self.skip_adapter1(z1), b2], dim=1)
        )
        return self._merge_z0(branch_decoder, z0, b1)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        image: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        missing = [name for name in self.feature_layers if name not in features]
        if missing:
            observed = {
                name: tuple(tensor.shape)
                for name, tensor in features.items()
                if hasattr(tensor, "shape")
            }
            raise KeyError(
                f"PfmbUNETR missing features {missing} from ordered interface "
                f"{self.feature_layers}; observed shapes={observed}"
            )
        if self.preprocessed_vit:
            z1_name, z2_name, z3_name, z4_name = self.feature_layers
            z1 = features[z1_name]
            z2 = features[z2_name]
            z3 = features[z3_name]
            z4 = features[z4_name]
            logits = self._forward_preprocessed_vit_pyramid(image, z1, z2, z3, z4)
            if output_size is not None and logits.shape[-2:] != output_size:
                logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            return logits
        grids = [tuple(features[name].shape[-2:]) for name in self.feature_layers]
        routing = classify_encoder_pyramid(grids)
        if routing == "native5":
            projected = [
                self._project_skip_channels(features, name) for name in self.feature_layers
            ]
            shallow, z1, z2, z3, z4 = projected
            logits = self._forward_upsample_native_pyramid5(
                image, shallow, z1, z2, z3, z4
            )
        elif routing == "native":
            projected = [
                self._project_skip_channels(features, name) for name in self.feature_layers
            ]
            z1, z2, z3, z4 = projected
            logits = self._forward_upsample_native_pyramid(image, z1, z2, z3, z4)
        elif routing == "deepest_plateau":
            projected = [
                self._project_skip_channels(features, name) for name in self.feature_layers
            ]
            z1, z2, z3, z4 = projected
            logits = self._forward_upsample_deepest_plateau(image, z1, z2, z3, z4)
        else:
            z1_name, z2_name, z3_name, z4_name = self.feature_layers
            z4 = self._project_skip_to_grid(features, z4_name, grids[-1])
            target_hw = z4.shape[-2:]
            z1 = self._project_skip_to_grid(features, z1_name, target_hw)
            z2 = self._project_skip_to_grid(features, z2_name, target_hw)
            z3 = self._project_skip_to_grid(features, z3_name, target_hw)
            logits = self._forward_upsample_uniform(image, z1, z2, z3, z4)
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
