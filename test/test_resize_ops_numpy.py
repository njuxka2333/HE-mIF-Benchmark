"""Regression tests for NumPy-native MPP geometry vs legacy PIL path."""

from __future__ import annotations

import numpy as np
from PIL import Image

import cv2

from utils.resize_ops import (
    apply_mpp_patch_geometry_at,
    apply_mpp_patch_geometry_at_np,
    apply_mpp_patch_geometry_triple_at_np,
    resize_continuous,
)


def _synthetic_volume(size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_hwc = np.zeros((size, size, 3), dtype=np.uint8)
    image_hwc[..., 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
    image_hwc[..., 1] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
    semantic = np.zeros((size, size), dtype=np.int32)
    semantic[50:150, 60:160] = 1
    semantic[200:320, 220:340] = 2
    instance = np.zeros((size, size), dtype=np.int32)
    instance[50:150, 60:160] = 1
    instance[200:320, 220:340] = 2
    return image_hwc, semantic, instance


def test_numpy_single_patch_matches_pil_path():
    image_hwc, semantic, _ = _synthetic_volume()
    left, top, read_size, output_size = 120, 80, 297, 224

    pil_img = Image.fromarray(image_hwc, mode="RGB")
    pil_out, pil_mask = apply_mpp_patch_geometry_at(
        pil_img,
        semantic,
        left,
        top,
        read_size,
        output_size,
    )
    np_out, np_mask = apply_mpp_patch_geometry_at_np(
        image_hwc,
        semantic,
        left,
        top,
        read_size,
        output_size,
    )

    assert pil_out.size == (output_size, output_size)
    assert np_out.shape == (output_size, output_size, 3)
    assert np.allclose(np.array(pil_out), np_out, atol=2)
    assert pil_mask is not None and np_mask is not None
    assert np.array_equal(pil_mask, np_mask)


def test_triple_numpy_keeps_instance_labels_for_offset_tile():
    image_hwc, semantic, instance = _synthetic_volume()
    left, top, read_size, output_size = 297, 0, 297, 224

    _, semantic_out, instance_out = apply_mpp_patch_geometry_triple_at_np(
        image_hwc,
        semantic,
        instance,
        left,
        top,
        read_size,
        output_size,
    )

    assert semantic_out is not None and instance_out is not None
    assert semantic_out.shape == (output_size, output_size)
    assert instance_out.shape == (output_size, output_size)
    assert np.any(instance_out > 0)
    assert np.any(semantic_out > 0)


def test_resize_continuous_area_down_cubic_up():
    src = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    up = resize_continuous(src, (8, 8))
    cubic = cv2.resize(src, (8, 8), interpolation=cv2.INTER_CUBIC)
    nearest = cv2.resize(src, (8, 8), interpolation=cv2.INTER_NEAREST)
    assert np.allclose(up, cubic)
    assert not np.allclose(up, nearest)

    big = np.arange(64, dtype=np.float32).reshape(8, 8)
    down = resize_continuous(big, (4, 4))
    area = cv2.resize(big, (4, 4), interpolation=cv2.INTER_AREA)
    assert np.allclose(down, area)
    same = resize_continuous(big, (8, 8))
    assert np.allclose(same, big)
