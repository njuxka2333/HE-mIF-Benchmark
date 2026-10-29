"""Trainable timm encoder + CellsegUNETR decoder on a generated mIF stack.

On the virtual-staining route the frozen model is run to its *prediction head*
instead (``mif`` / ``output`` / ``final`` in ``feature_extraction.layers``), giving
a ``[C, H, W]`` generated marker stack at the foundation model's expected
size/mpp.  There are no multi-level encoder features to hand the decoder, so a
trainable encoder must supply the pyramid.  The stack is optionally resampled to
a **task-unified** UNet grid (cellseg 256 / TLS 512) **before** z-scoring::

    mif [C,H,W] --[resample?]--> [z-score]--> [TimmPyramidEncoder] --> {z1..z4}
                                         \\------------------ stem skip ------/
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import timm
import torch
import torch.nn as nn

from heads.cellseg import DEFAULT_DECODER_CHANNELS, CellsegUNETR
from utils.marker_stack import resample_mif_stack

__all__ = [
    "DEFAULT_FEATURE_LAYERS",
    "TimmPyramidEncoder",
    "MifUNet",
    "build_mif_unet",
]

DEFAULT_FEATURE_LAYERS = ("z1", "z2", "z3", "z4")


class TimmPyramidEncoder(nn.Module):
    """``timm`` ``features_only`` backbone that returns four skip maps."""

    def __init__(
        self,
        name: str,
        *,
        in_chans: int,
        out_indices: Sequence[int] = (1, 2, 3, 4),
        pretrained: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            name,
            features_only=True,
            pretrained=bool(pretrained),
            in_chans=int(in_chans),
            out_indices=tuple(int(i) for i in out_indices),
            **kwargs,
        )
        info = self.model.feature_info
        self.channels = [int(c) for c in info.channels()]
        self.reductions = [int(r) for r in info.reduction()]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x))


class MifUNet(nn.Module):
    """Trainable timm encoder + the frozen-route ``CellsegUNETR`` decoder."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        timm_encoder_name: str = "resnet50",
        out_indices: Sequence[int] = (1, 2, 3, 4),
        pretrained: bool = False,
        decoder_channels: Sequence[int] | None = None,
        drop_rate: float = 0.0,
        channel_proj_dim: int | None = None,
        feature_layers: Sequence[str] | None = None,
        encoder_kwargs: Mapping[str, Any] | None = None,
        generator: nn.Module | None = None,
        target_size: int | tuple[int, int] | None = None,
        upsample_mode: str = "bicubic",
        resample_clamp: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.generator = generator
        if generator is not None:
            for param in generator.parameters():
                param.requires_grad_(False)
        self.target_size = target_size
        self.upsample_mode = str(upsample_mode)
        self.resample_clamp = (
            None
            if resample_clamp is None
            else (float(resample_clamp[0]), float(resample_clamp[1]))
        )
        self.channel_proj_dim = (
            None if channel_proj_dim is None else int(channel_proj_dim)
        )
        stem_ch = int(self.channel_proj_dim or self.in_channels)
        self.channel_proj: nn.Module
        if self.channel_proj_dim is None:
            self.channel_proj = nn.Identity()
        else:
            self.channel_proj = nn.Conv2d(self.in_channels, self.channel_proj_dim, 1)

        self.register_buffer("channel_mean", torch.zeros(1, self.in_channels, 1, 1))
        self.register_buffer("channel_std", torch.ones(1, self.in_channels, 1, 1))

        self.feature_layers = list(feature_layers or DEFAULT_FEATURE_LAYERS)
        self.encoder = TimmPyramidEncoder(
            str(timm_encoder_name),
            in_chans=stem_ch,
            out_indices=out_indices,
            pretrained=pretrained,
            **dict(encoder_kwargs or {}),
        )
        if len(self.encoder.channels) != len(self.feature_layers):
            raise ValueError(
                f"encoder skips {self.encoder.channels} vs layers {self.feature_layers}"
            )
        channels = list(decoder_channels or DEFAULT_DECODER_CHANNELS)
        self.decoder = CellsegUNETR(
            feature_layers=self.feature_layers,
            in_channels=list(self.encoder.channels),
            num_classes=self.num_classes,
            decoder_channels=list(channels),
            drop_rate=float(drop_rate),
            stem_in_channels=stem_ch,
        )

    def set_channel_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1, 1).clamp_min(1e-6)
        if int(mean_t.shape[1]) != self.in_channels:
            raise ValueError(
                f"channel stats have {mean_t.shape[1]} channels, expected {self.in_channels}"
            )
        self.channel_mean.copy_(mean_t)
        self.channel_std.copy_(std_t)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.channel_mean.to(device=x.device, dtype=x.dtype)
        std = self.channel_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(
        self,
        x: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        *,
        image: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
        features: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict class logits from a generated marker stack.

        Offline: pass the stack as ``x`` / ``image=``.
        Online: pass H&E as ``rgb=`` when a frozen ``generator`` is attached
        (RGB should already be at the foundation model's expected size/mpp).
        ``features`` is accepted and ignored (this route has no frozen-encoder features).

        After generation (or offline load), the stack is resampled to
        ``self.target_size`` **before** z-scoring when that size is set.
        """
        del features
        if rgb is not None:
            if self.generator is None:
                raise ValueError("rgb= given but no generator attached")
            x = self.generator(rgb)
        elif x is None:
            x = image
        if x is None:
            raise ValueError("MifUNet requires the mIF stack as `x` (or `image=`), or `rgb=`")
        if x.dim() != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        x = resample_mif_stack(
            x,
            self.target_size,
            upsample_mode=self.upsample_mode,
            clamp=self.resample_clamp,
        )

        stride = int(self.encoder.reductions[-1])
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h % stride or w % stride:
            raise ValueError(
                f"input {h}x{w} must be divisible by the deepest encoder reduction "
                f"({stride})"
            )

        x = self.normalize(x)
        x = self.channel_proj(x)
        feats = self.encoder(x)
        feature_map = dict(zip(self.feature_layers, feats))
        return self.decoder(feature_map, image=x, output_size=output_size)


def build_mif_unet(config: Mapping[str, Any]) -> MifUNet:
    """Build a :class:`MifUNet` from a head config mapping."""
    cfg = dict(config)
    clamp = cfg.get("resample_clamp")
    if clamp is not None:
        clamp = tuple(float(v) for v in clamp)
    return MifUNet(
        in_channels=int(cfg["in_channels"]),
        num_classes=int(cfg["num_classes"]),
        timm_encoder_name=str(cfg.get("timm_encoder_name", "resnet50")),
        out_indices=tuple(cfg.get("out_indices", (1, 2, 3, 4))),
        pretrained=bool(cfg.get("pretrained", False)),
        decoder_channels=cfg.get("decoder_channels"),
        drop_rate=float(cfg.get("dropout", cfg.get("drop_rate", 0.0))),
        channel_proj_dim=cfg.get("channel_proj_dim"),
        feature_layers=cfg.get("feature_layers"),
        encoder_kwargs=cfg.get("encoder_kwargs"),
        generator=cfg.get("generator"),
        target_size=cfg.get("target_size"),
        upsample_mode=str(cfg.get("upsample_mode", "bicubic")),
        resample_clamp=clamp,
    )
