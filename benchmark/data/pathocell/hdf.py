"""Read PathoCell HDF volumes into arrays used by dense-prediction loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image

PATHOCELL_NATIVE_MPP = 0.377


def _he_chw_uint16_to_hwc_uint8(array: np.ndarray) -> np.ndarray:
    """Convert the HDF ``img`` dataset from CHW uint16 to HWC uint8 RGB."""
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    peak = float(image.max()) if image.size and image.max() > 0 else 1.0
    return (image.astype(np.float32) / peak * 255.0).clip(0, 255).astype(np.uint8)


def pathocell_hdf_metadata(hdf_path: str | Path) -> tuple[int, int, bool, bool]:
    """Return image dimensions and the semantic/instance-mask availability."""
    path = Path(hdf_path)
    with h5py.File(path, "r") as handle:
        if "img" not in handle:
            raise KeyError(f"{path} missing dataset 'img'")
        image = handle["img"]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            height, width = int(image.shape[1]), int(image.shape[2])
        elif image.ndim == 2:
            height, width = int(image.shape[0]), int(image.shape[1])
        else:
            raise ValueError(f"{path} dataset 'img' has unsupported shape {image.shape}")
        has_semantic = "gt_ct_coarse" in handle
        has_instance = "gt_inst" in handle
    return height, width, has_semantic, has_instance


def load_pathocell_volume(
    hdf_path: str | Path,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load H&E RGB, coarse semantic labels, and instance labels from one HDF."""
    path = Path(hdf_path)
    with h5py.File(path, "r") as handle:
        if "img" not in handle:
            raise KeyError(f"{path} missing dataset 'img'")
        image = _he_chw_uint16_to_hwc_uint8(handle["img"][:])
        semantic_mask = None
        if "gt_ct_coarse" in handle:
            semantic_mask = np.asarray(handle["gt_ct_coarse"], dtype=np.int32)
            if semantic_mask.ndim == 3 and semantic_mask.shape[0] == 1:
                semantic_mask = semantic_mask[0]
            if semantic_mask.ndim != 2:
                raise ValueError(
                    f"{path} dataset 'gt_ct_coarse' must be HW or 1xHxW, "
                    f"got {semantic_mask.shape}"
                )
        instance_mask = None
        if "gt_inst" in handle:
            instance_mask = np.asarray(handle["gt_inst"], dtype=np.int32)
            if instance_mask.ndim == 3 and instance_mask.shape[0] == 1:
                instance_mask = instance_mask[0]
            if instance_mask.ndim != 2:
                raise ValueError(
                    f"{path} dataset 'gt_inst' must be HW or 1xHxW, got {instance_mask.shape}"
                )
    return image, semantic_mask, instance_mask


def read_pathocell_hdf(
    hdf_path: str | Path,
) -> tuple[Image.Image, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PathoCell image and masks, returning RGB as a PIL image."""
    image, semantic_mask, instance_mask = load_pathocell_volume(hdf_path)
    return Image.fromarray(image), semantic_mask, instance_mask
