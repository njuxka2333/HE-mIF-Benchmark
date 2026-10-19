"""PathoCell HDF volume loader: grid-tile large slide volumes into model patches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

from utils.resize_ops import (
    crop_or_pad_image_mask_np,
    minimal_cover_grid,
    mpp_patch_sizes,
    resize_image_np,
    resize_mask,
)

# PathoCell HDF pixels match the original Schürch CODEX grid (~377 nm/px at 20x).
PATHOCELL_NATIVE_MPP = 0.377
PATHOCELL_DEFAULT_NATIVE_TILE = 256

PATHOCELL_TILING_PFMB = "pfmb"
PATHOCELL_TILING_MINIMAL = "minimal_cover"
PathoCellTile = Tuple[int, int, int, int]  # y0, y1, x0, x1


def resolve_pathocell_tiling(exp: Any, cfg: Any | None = None) -> str:
    """Resolve PathoCell tiling protocol (PFMB linspace+center-pad vs minimal-cover grid)."""
    if cfg is not None:
        data = getattr(cfg, "data", None)
        if isinstance(data, dict) and data.get("pathocell_tiling"):
            return str(data["pathocell_tiling"]).strip().lower()
    if getattr(exp, "pathocell_tiling", None):
        return str(exp.pathocell_tiling).strip().lower()
    return PATHOCELL_TILING_PFMB


def is_pfmb_tiling(protocol: str) -> bool:
    return str(protocol).strip().lower() in {PATHOCELL_TILING_PFMB, "pfmb_linspace"}


def tile_top_left(tile: PathoCellTile) -> Tuple[int, int]:
    """Return top-left ``(x, y)`` for a tile bounds tuple."""
    return int(tile[2]), int(tile[0])


def pfmb_linspace_tiles(height: int, width: int, tile_size: int) -> List[PathoCellTile]:
    """PFMB ``split_to_tiles``: linspace boundaries, variable slices before center-pad."""
    tile_size = max(1, int(tile_size))
    n_tiles_y = int(np.ceil(height / tile_size))
    n_tiles_x = int(np.ceil(width / tile_size))
    y_indices = np.linspace(0, height, n_tiles_y + 1, dtype=int)
    x_indices = np.linspace(0, width, n_tiles_x + 1, dtype=int)
    tiles: List[PathoCellTile] = []
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            tiles.append(
                (
                    int(y_indices[i]),
                    int(y_indices[i + 1]),
                    int(x_indices[j]),
                    int(x_indices[j + 1]),
                )
            )
    return tiles


def center_pad_hwc(image_hwc: np.ndarray, out_h: int, out_w: int, fill: int = 0) -> np.ndarray:
    """Center-pad HWC RGB to ``(out_h, out_w)``."""
    out_h = max(1, int(out_h))
    out_w = max(1, int(out_w))
    h, w = image_hwc.shape[:2]
    pad_top = max(0, (out_h - h) // 2)
    pad_left = max(0, (out_w - w) // 2)
    pad_bottom = max(0, out_h - h - pad_top)
    pad_right = max(0, out_w - w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        image_hwc = np.pad(
            image_hwc,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=fill,
        )
    return image_hwc


def center_pad_2d(mask: np.ndarray, out_h: int, out_w: int, fill: int = 0) -> np.ndarray:
    """Center-pad a 2D mask to ``(out_h, out_w)``."""
    out_h = max(1, int(out_h))
    out_w = max(1, int(out_w))
    h, w = mask.shape[:2]
    pad_top = max(0, (out_h - h) // 2)
    pad_left = max(0, (out_w - w) // 2)
    pad_bottom = max(0, out_h - h - pad_top)
    pad_right = max(0, out_w - w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill,
        )
    return mask


def renumber_instance_tile(instance: np.ndarray) -> np.ndarray:
    """Renumber non-zero instance IDs in one tile to ``1..N`` (PFMB LMDB convention)."""
    unique = np.unique(instance)
    unique = unique[unique != 0]
    out = np.zeros_like(instance)
    for new_id, old_id in enumerate(unique, start=1):
        out[instance == old_id] = new_id
    return out


def extract_pathocell_tile_triple_at_np(
    image_hwc: np.ndarray,
    semantic_hw: Optional[np.ndarray],
    instance_hw: Optional[np.ndarray],
    tile: PathoCellTile,
    read_size: int,
    output_size: int,
    *,
    protocol: str = PATHOCELL_TILING_MINIMAL,
    renumber_instances: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract one PathoCell tile with PFMB or minimal-cover geometry."""
    y0, y1, x0, x1 = tile
    read_size = max(1, int(read_size))
    output_size = max(1, int(output_size))
    if is_pfmb_tiling(protocol):
        image_crop = image_hwc[y0:y1, x0:x1]
        semantic = semantic_hw[y0:y1, x0:x1] if semantic_hw is not None else None
        instance = instance_hw[y0:y1, x0:x1] if instance_hw is not None else None
        image_crop = center_pad_hwc(image_crop, read_size, read_size)
        if semantic is not None:
            semantic = center_pad_2d(semantic, read_size, read_size)
        if instance is not None:
            instance = center_pad_2d(instance, read_size, read_size)
    else:
        image_crop, semantic = crop_or_pad_image_mask_np(
            image_hwc,
            semantic_hw,
            x0,
            y0,
            read_size,
        )
        _, instance = crop_or_pad_image_mask_np(
            image_hwc,
            instance_hw,
            x0,
            y0,
            read_size,
        )
    if image_crop.shape[0] != output_size or image_crop.shape[1] != output_size:
        image_crop = resize_image_np(image_crop, (output_size, output_size))
        if semantic is not None:
            semantic = resize_mask(semantic, (output_size, output_size))
        if instance is not None:
            instance = resize_mask(instance, (output_size, output_size))
    if renumber_instances and instance is not None:
        instance = renumber_instance_tile(instance)
    return image_crop, semantic, instance


def pathocell_tile_geometry(
    target_patch_size: int,
    target_mpp: Optional[float],
    *,
    native_mpp: float = PATHOCELL_NATIVE_MPP,
    default_native_tile: int = PATHOCELL_DEFAULT_NATIVE_TILE,
) -> Tuple[int, int]:
    """Return ``(native_read_px, output_px)`` for one PathoCell tile."""
    return mpp_patch_sizes(
        target_patch_size,
        target_mpp,
        native_mpp=native_mpp,
        default_patch_size=default_native_tile,
    )


def iter_grid_coords(
    height: int,
    width: int,
    tile_size: int,
    step: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Top-left coordinates for ``tile_size`` windows with minimal overlap by default."""
    if step is None:
        return minimal_cover_grid(height, width, tile_size)
    stride = max(1, int(step))
    tile_size = max(1, int(tile_size))
    coords: List[Tuple[int, int]] = []
    if height <= tile_size or width <= tile_size:
        return minimal_cover_grid(height, width, tile_size)
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            coords.append((x, y))
    if coords:
        if coords[-1][0] != width - tile_size:
            for y in sorted({coord[1] for coord in coords}):
                coords.append((width - tile_size, y))
        if coords[-1][1] != height - tile_size:
            for x in sorted({coord[0] for coord in coords}):
                coords.append((x, height - tile_size))
    return coords


def plan_pathocell_tiles(
    height: int,
    width: int,
    read_size: int,
    step: Optional[int] = None,
    *,
    protocol: str = PATHOCELL_TILING_MINIMAL,
) -> Tuple[int, List[PathoCellTile]]:
    """
    Plan native read size and tile bounds for one PathoCell FOV.

    ``protocol=pfmb`` — PFMB ``linspace`` grid with center-pad to ``read_size``.
    ``protocol=minimal_cover`` — fewest fixed ``read_size`` windows (legacy grid).
    """
    read_size = max(1, int(read_size))
    if is_pfmb_tiling(protocol):
        return read_size, pfmb_linspace_tiles(height, width, read_size)
    stride = read_size if step is None else step
    coords = iter_grid_coords(height, width, read_size, stride)
    return read_size, [(y, y + read_size, x, x + read_size) for x, y in coords]


def _he_chw_uint16_to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert PathoCell ``img`` dataset (C, H, W) uint16 to HWC uint8 RGB."""
    data = np.asarray(arr)
    if data.ndim == 3 and data.shape[0] in (1, 3):
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    peak = float(data.max()) if data.size and data.max() > 0 else 1.0
    return (data.astype(np.float32) / peak * 255.0).clip(0, 255).astype(np.uint8)


def pathocell_hdf_metadata(hdf_path: str | Path) -> Tuple[int, int, bool, bool]:
    """Return ``(height, width, has_semantic, has_instance)`` without loading arrays."""
    path = Path(hdf_path)
    with h5py.File(path, "r") as handle:
        if "img" not in handle:
            raise KeyError(f"{path} missing dataset 'img'")
        img = handle["img"]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            height, width = int(img.shape[1]), int(img.shape[2])
        elif img.ndim == 2:
            height, width = int(img.shape[0]), int(img.shape[1])
        else:
            raise ValueError(f"{path} dataset 'img' has unsupported shape {img.shape}")
        has_semantic = "gt_ct_coarse" in handle
        has_instance = "gt_inst" in handle
    return height, width, has_semantic, has_instance


def load_pathocell_volume(
    hdf_path: str | Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load full H&E RGB (HWC uint8), coarse semantic mask, and instance mask from one HDF."""
    path = Path(hdf_path)
    with h5py.File(path, "r") as handle:
        if "img" not in handle:
            raise KeyError(f"{path} missing dataset 'img'")
        image = _he_chw_uint16_to_hwc_uint8(handle["img"][:])
        mask = None
        if "gt_ct_coarse" in handle:
            raw_mask = np.asarray(handle["gt_ct_coarse"], dtype=np.int32)
            if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
                raw_mask = raw_mask[0]
            if raw_mask.ndim != 2:
                raise ValueError(
                    f"{path} dataset 'gt_ct_coarse' must be HW or 1xHxW, got {raw_mask.shape}"
                )
            mask = raw_mask
        instance_mask = None
        if "gt_inst" in handle:
            raw_inst = np.asarray(handle["gt_inst"], dtype=np.int32)
            if raw_inst.ndim == 3 and raw_inst.shape[0] == 1:
                raw_inst = raw_inst[0]
            if raw_inst.ndim != 2:
                raise ValueError(
                    f"{path} dataset 'gt_inst' must be HW or 1xHxW, got {raw_inst.shape}"
                )
            instance_mask = raw_inst
    return image, mask, instance_mask


def read_pathocell_hdf(
    hdf_path: str | Path,
) -> Tuple[Image.Image, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load H&E RGB, coarse semantic mask, and instance mask from a PathoCell ``.hdf`` file."""
    image, mask, instance_mask = load_pathocell_volume(hdf_path)
    return Image.fromarray(image, mode="RGB"), mask, instance_mask
