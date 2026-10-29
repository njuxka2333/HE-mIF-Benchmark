"""HE → frozen mIF generator → trainable timm encoder + FeatureUNETR (TLS)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from heads import FeatureUNETR
from utils.marker_stack import resample_mif_stack
from heads.mif_unet import DEFAULT_FEATURE_LAYERS, TimmPyramidEncoder
from models.mif_output import FrozenMifGenerator

__all__ = ["FrozenMifGenerator", "MifFeatureUNETR", "build_mif_feature_unetr"]


class MifFeatureUNETR(nn.Module):
    """Frozen generator + trainable timm pyramid + FeatureUNETR decoder."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        generator: nn.Module | None = None,
        timm_encoder_name: str = "resnet50",
        out_indices: Sequence[int] = (1, 2, 3, 4),
        pretrained: bool = False,
        adapter_channels: Sequence[int] | None = None,
        dropout: float = 0.0,
        z0_channels: int = 64,
        use_z0: bool = True,
        channel_proj_dim: int | None = None,
        feature_layers: Sequence[str] | None = None,
        encoder_kwargs: Mapping[str, Any] | None = None,
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
        self.channel_proj_dim = None if channel_proj_dim is None else int(channel_proj_dim)
        stem_ch = int(self.channel_proj_dim or self.in_channels)
        if self.channel_proj_dim is None:
            self.channel_proj: nn.Module = nn.Identity()
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
        self.decoder = FeatureUNETR(
            feature_layers=self.feature_layers,
            in_channels=list(self.encoder.channels),
            num_classes=self.num_classes,
            adapter_channels=None if adapter_channels is None else list(adapter_channels),
            dropout=float(dropout),
            z0_channels=int(z0_channels),
        )
        if hasattr(self.decoder, "use_z0"):
            self.decoder.use_z0 = bool(use_z0)

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
        """Predict logits.

        Online: pass H&E as ``rgb=`` at the foundation model's expected size/mpp;
        the frozen generator produces the stack, which is then resampled to
        ``self.target_size`` (e.g. TLS 512) **before** z-scoring.
        Offline: pass an already-generated stack as ``x`` (or ``image=``).
        ``features`` is accepted and ignored (this route has no frozen-encoder features).
        """
        del features
        if rgb is not None:
            if self.generator is None:
                raise ValueError("rgb= given but no generator attached")
            stack = self.generator(rgb)
        else:
            stack = x if x is not None else image
        if stack is None:
            raise ValueError("MifFeatureUNETR needs `rgb=` (online) or `x` (pre-generated)")
        if stack.dim() != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(stack.shape)}")
        if stack.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} generated channels, got {stack.shape[1]}"
            )

        stack = resample_mif_stack(
            stack,
            self.target_size,
            upsample_mode=self.upsample_mode,
            clamp=self.resample_clamp,
        )

        stride = int(self.encoder.reductions[-1])
        h, w = int(stack.shape[-2]), int(stack.shape[-1])
        if h % stride or w % stride:
            raise ValueError(
                f"input {h}x{w} must be divisible by the deepest encoder reduction ({stride})"
            )

        stack = self.normalize(stack.to(self.channel_mean.dtype))
        body = self.channel_proj(stack)
        feats = self.encoder(body)
        feature_map = dict(zip(self.feature_layers, feats))
        return self.decoder(
            feature_map,
            output_size=output_size,
            image=body if getattr(self.decoder, "use_z0", True) else None,
        )


def build_mif_feature_unetr(config: Mapping[str, Any]) -> MifFeatureUNETR:
    cfg = dict(config)
    clamp = cfg.get("resample_clamp")
    if clamp is not None:
        clamp = tuple(float(v) for v in clamp)
    return MifFeatureUNETR(
        in_channels=int(cfg["in_channels"]),
        num_classes=int(cfg["num_classes"]),
        generator=cfg.get("generator"),
        timm_encoder_name=str(cfg.get("timm_encoder_name", "resnet50")),
        out_indices=tuple(cfg.get("out_indices", (1, 2, 3, 4))),
        pretrained=bool(cfg.get("pretrained", False)),
        adapter_channels=cfg.get("adapter_channels"),
        dropout=float(cfg.get("dropout", cfg.get("drop_rate", 0.0))),
        z0_channels=int(cfg.get("z0_channels", 64)),
        use_z0=bool(cfg.get("use_z0", True)),
        channel_proj_dim=cfg.get("channel_proj_dim"),
        feature_layers=cfg.get("feature_layers"),
        encoder_kwargs=cfg.get("encoder_kwargs"),
        target_size=cfg.get("target_size"),
        upsample_mode=str(cfg.get("upsample_mode", "bicubic")),
        resample_clamp=clamp,
    )
