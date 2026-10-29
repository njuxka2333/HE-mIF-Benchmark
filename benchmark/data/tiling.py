"""Shared patch geometry and array extraction for dense prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.resize_ops import mpp_patch_sizes, resize_image_np, resize_mask

PFMB_TILE_PROTOCOL = "pfmb"
TileBounds = tuple[int, int, int, int]  # y0, y1, x0, x1


def is_pfmb_tiling(protocol: str) -> bool:
    """Validate the supported tiling protocol and return whether it is PFMB."""
    value = str(protocol).strip().lower()
    if value not in {PFMB_TILE_PROTOCOL, "pfmb_linspace"}:
        raise ValueError(
            f"unsupported tiling protocol {protocol!r}; use '{PFMB_TILE_PROTOCOL}'"
        )
    return True


def tile_top_left(tile: TileBounds) -> tuple[int, int]:
    """Return ``(x, y)`` for one ``(y0, y1, x0, x1)`` tile."""
    return int(tile[2]), int(tile[0])


def pfmb_grid(height: int, width: int, tile_size: int) -> list[TileBounds]:
    """Plan PFMB linspace bounds; edge tiles may be short before padding."""
    tile_size = max(1, int(tile_size))
    tile_rows = int(np.ceil(height / tile_size))
    tile_columns = int(np.ceil(width / tile_size))
    y_bounds = np.linspace(0, height, tile_rows + 1, dtype=int)
    x_bounds = np.linspace(0, width, tile_columns + 1, dtype=int)
    return [
        (
            int(y_bounds[row]),
            int(y_bounds[row + 1]),
            int(x_bounds[column]),
            int(x_bounds[column + 1]),
        )
        for row in range(tile_rows)
        for column in range(tile_columns)
    ]


def center_pad_hwc(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    fill_value: int = 0,
) -> np.ndarray:
    """Center-pad an HWC image to the requested spatial dimensions."""
    output_height = max(1, int(output_height))
    output_width = max(1, int(output_width))
    height, width = image.shape[:2]
    pad_top = max(0, (output_height - height) // 2)
    pad_left = max(0, (output_width - width) // 2)
    pad_bottom = max(0, output_height - height - pad_top)
    pad_right = max(0, output_width - width - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )
    return image


def center_pad_2d(
    mask: np.ndarray,
    output_height: int,
    output_width: int,
    fill_value: int = 0,
) -> np.ndarray:
    """Center-pad a two-dimensional mask."""
    output_height = max(1, int(output_height))
    output_width = max(1, int(output_width))
    height, width = mask.shape[:2]
    pad_top = max(0, (output_height - height) // 2)
    pad_left = max(0, (output_width - width) // 2)
    pad_bottom = max(0, output_height - height - pad_top)
    pad_right = max(0, output_width - width - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )
    return mask


def renumber_instance_ids(instance_mask: np.ndarray) -> np.ndarray:
    """Renumber non-zero instance ids within a tile to consecutive integers."""
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]
    output = np.zeros_like(instance_mask)
    for new_id, old_id in enumerate(instance_ids, start=1):
        output[instance_mask == old_id] = new_id
    return output


def extract_tile_arrays(
    image: np.ndarray,
    semantic_mask: Optional[np.ndarray],
    instance_mask: Optional[np.ndarray],
    tile: TileBounds,
    read_size: int,
    output_size: int,
    *,
    protocol: str = PFMB_TILE_PROTOCOL,
    renumber_instances: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Crop, center-pad, and resize image, semantic, and instance arrays together."""
    is_pfmb_tiling(protocol)
    y0, y1, x0, x1 = tile
    read_size = max(1, int(read_size))
    output_size = max(1, int(output_size))
    image_tile = center_pad_hwc(image[y0:y1, x0:x1], read_size, read_size)
    semantic_tile = (
        center_pad_2d(semantic_mask[y0:y1, x0:x1], read_size, read_size)
        if semantic_mask is not None
        else None
    )
    instance_tile = (
        center_pad_2d(instance_mask[y0:y1, x0:x1], read_size, read_size)
        if instance_mask is not None
        else None
    )
    if image_tile.shape[:2] != (output_size, output_size):
        image_tile = resize_image_np(image_tile, (output_size, output_size))
        if semantic_tile is not None:
            semantic_tile = resize_mask(semantic_tile, (output_size, output_size))
        if instance_tile is not None:
            instance_tile = resize_mask(instance_tile, (output_size, output_size))
    if renumber_instances and instance_tile is not None:
        instance_tile = renumber_instance_ids(instance_tile)
    return image_tile, semantic_tile, instance_tile


def tile_geometry(
    target_patch_size: int,
    target_mpp: Optional[float],
    *,
    native_mpp: float,
    default_native_tile: int = 256,
) -> tuple[int, int]:
    """Return native read size and model input size for a dataset's pixel scale."""
    return mpp_patch_sizes(
        target_patch_size,
        target_mpp,
        native_mpp=native_mpp,
        default_patch_size=default_native_tile,
    )


def plan_tiles(
    height: int,
    width: int,
    read_size: int,
    *,
    protocol: str = PFMB_TILE_PROTOCOL,
) -> tuple[int, list[TileBounds]]:
    """Return the normalized read size and corresponding tile bounds."""
    read_size = max(1, int(read_size))
    is_pfmb_tiling(protocol)
    return read_size, pfmb_grid(height, width, read_size)
