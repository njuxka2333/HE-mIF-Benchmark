"""
following PrePATH style.

Shared helpers for variable spatial size (including native 64x64 single-cell patches).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SINGLE_CELL_SIZE = 64


def log_native_64_patch(model_name: str, x: torch.Tensor) -> None:
    if x.dim() >= 4 and x.shape[-2] == SINGLE_CELL_SIZE and x.shape[-1] == SINGLE_CELL_SIZE:
        logger.info(f"Model {model_name} processing native 64x64 single-cell patch.")


def spatial_hw(x: torch.Tensor) -> Tuple[int, int]:
    return int(x.shape[-2]), int(x.shape[-1])


def ensure_min_spatial(
    x: torch.Tensor,
    min_size: int = 1,
) -> Tuple[torch.Tensor, Tuple[int, int], bool]:
    """Upsample with bilinear if H or W are below min_size; returns (tensor, original_hw, resized)."""
    h, w = spatial_hw(x)
    if h >= min_size and w >= min_size:
        return x, (h, w), False
    target = max(min_size, h, w)
    x_up = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    return x_up, (h, w), True


def restore_spatial(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
    resized: bool,
) -> torch.Tensor:
    if not resized:
        return x
    h, w = target_hw
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def safe_feature_map(
    feat: torch.Tensor,
    min_hw: int = 1,
) -> torch.Tensor:
    """Prevent 0x0 feature maps after aggressive downsampling."""
    if feat.dim() != 4:
        return feat
    h, w = feat.shape[-2], feat.shape[-1]
    if h >= min_hw and w >= min_hw:
        return feat
    return F.adaptive_avg_pool2d(feat, (max(min_hw, h), max(min_hw, w)))


def patch_swin_pos_drop(swin_module: nn.Module) -> None:
    """Newer timm drops pos_drop; HEMIT forward still references it."""
    if not hasattr(swin_module, "pos_drop"):
        swin_module.pos_drop = nn.Identity()


def try_vit_set_input_size(model: nn.Module, height: int, width: int) -> None:
    """Resize ViT/Swin token grid for the current spatial input (best-effort)."""
    if hasattr(model, "set_input_size"):
        try:
            model.set_input_size((height, width))
            return
        except ValueError:
            pass
    encoder = getattr(model, "encoder", None)
    vit = getattr(encoder, "vit", None) if encoder is not None else None
    if vit is not None and hasattr(vit, "set_input_size"):
        vit.set_input_size(img_size=(height, width))
