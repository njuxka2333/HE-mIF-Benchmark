"""
following PrePATH style.

GigaTIME baseline inference (conv U-Net++, 23 output channels).

Supports variable input sizes including native 64x64 single-cell patches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.patch_compat import (
    ensure_min_spatial,
    log_native_64_patch,
    restore_spatial,
    safe_feature_map,
    spatial_hw,
)

from utils.paths import reference_model_path

_GIGATIME_SCRIPTS = reference_model_path("gigatime") / "scripts"
if str(_GIGATIME_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_GIGATIME_SCRIPTS))

from archs import gigatime  # noqa: E402

NUM_CLASSES = 23
WINDOW_SIZE = 256


def _load_state_dict(model: nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "GigaTIME checkpoint loaded with non-strict key differences: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


class GigaTIMEWrapper(nn.Module):
    """GigaTIME U-Net++ with tiled inference for inputs larger than window_size."""

    def __init__(self, net: nn.Module, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.net = net
        self.window_size = window_size

    def _forward_tile(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h < self.window_size or w < self.window_size:
            pad_h = max(0, self.window_size - h)
            pad_w = max(0, self.window_size - w)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out = self.net(x)
            return out[:, :, :h, :w]
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_native_64_patch("GigaTIME", x)
        x, orig_hw, was_padded = ensure_min_spatial(x, min_size=1)

        b, _, height, width = x.shape
        if height <= self.window_size and width <= self.window_size:
            out = self._forward_tile(x)
            return restore_spatial(out, orig_hw, was_padded)

        output = torch.zeros(
            b, NUM_CLASSES, height, width, device=x.device, dtype=x.dtype
        )
        for i in range(0, height, self.window_size):
            for j in range(0, width, self.window_size):
                h_end = min(i + self.window_size, height)
                w_end = min(j + self.window_size, width)
                tile = x[:, :, i:h_end, j:w_end]
                pred = self._forward_tile(tile)
                pred = safe_feature_map(pred, min_hw=1)
                output[:, :, i:h_end, j:w_end] = pred[:, :, : h_end - i, : w_end - j]
        return restore_spatial(output, orig_hw, was_padded)


def get_model(device: torch.device, ckpt_path: str, gpu_num: int = 1) -> nn.Module:
    del gpu_num
    net = gigatime(num_classes=NUM_CLASSES, input_channels=3)
    _load_state_dict(net, ckpt_path)
    window_size = int(os.environ.get("GIGATIME_WINDOW_SIZE", str(WINDOW_SIZE)))
    model = GigaTIMEWrapper(net, window_size=window_size)
    return model.to(device).eval()


def get_trans():
    # Albumentations Normalize() in GigaTIME training (ImageNet stats on [0, 1] tensors).
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
