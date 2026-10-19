"""Tests for mIF stack resample (area down / bicubic up)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mif.resample import (  # noqa: E402
    CELLSEG_UNET_SIZE,
    TLS_UNET_SIZE,
    align_prediction_to_gt,
    resample_mif_stack,
    resample_mif_stack_np,
)


def test_downsample_uses_area_shape() -> None:
    x = torch.randn(2, 5, 512, 512)
    y = resample_mif_stack(x, CELLSEG_UNET_SIZE)
    assert tuple(y.shape) == (2, 5, 256, 256)


def test_upsample_bicubic_and_clamp() -> None:
    x = torch.full((1, 3, 128, 128), 0.9)
    y = resample_mif_stack(x, TLS_UNET_SIZE, upsample_mode="bicubic", clamp=(-1.0, 1.0))
    assert tuple(y.shape) == (1, 3, 512, 512)
    assert float(y.min()) >= -1.0 - 1e-5
    assert float(y.max()) <= 1.0 + 1e-5


def test_identity_when_matching() -> None:
    x = torch.randn(1, 4, 256, 256)
    y = resample_mif_stack(x, 256)
    assert y.data_ptr() == x.data_ptr() or torch.equal(y, x)


def test_mif_unet_resamples_before_normalize() -> None:
    from heads.mif_unet import MifUNet

    model = MifUNet(
        in_channels=3,
        num_classes=14,
        timm_encoder_name="resnet18",
        pretrained=False,
        decoder_channels=[64, 128, 256, 256],
        target_size=128,
    )
    model.set_channel_stats(torch.zeros(3), torch.ones(3))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256), output_size=(128, 128))
    assert tuple(out.shape) == (1, 14, 128, 128)


def test_np_area_down_matches_torch_shape() -> None:
    arr = np.random.randn(7, 384, 384).astype(np.float32)
    out = resample_mif_stack_np(arr, 256, 256)
    assert out.shape == (7, 256, 256)


def test_align_prediction_to_gt_matches_gt_hw() -> None:
    pred = np.random.randn(11, 128, 128).astype(np.float32)
    gt = np.random.randn(12, 788, 788).astype(np.float32)
    aligned = align_prediction_to_gt(pred, gt)
    assert aligned.shape == (11, 788, 788)
    same = align_prediction_to_gt(gt[:3], gt)
    assert same.shape == (3, 788, 788)
    np.testing.assert_array_equal(same, gt[:3].astype(np.float32))
