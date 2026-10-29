"""WSI coordinate grids and raster-mask sampling helpers for HookNet-TLS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from data.loader import DatasetConfig, SplitSample
from .annotations import load_official_wsa, semantic_mask_from_xml, xml_path_for_slide

_WSA_UNSET = object()


def tls_raster_mask_path(cfg: DatasetConfig, slide_id: str) -> Path | None:
    if cfg.mask_dir is None:
        return None
    path = cfg.mask_dir / f"{slide_id}{cfg.mask_suffix}"
    return path if path.is_file() else None

def _level0_patch_rect(
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
) -> tuple[int, int, int, int]:
    ratio = int(wsi.get_downsampling_from_spacing(spacing))
    patch_w_l0 = patch_size * ratio
    patch_h_l0 = patch_size * ratio
    x_l0 = int(x)
    y_l0 = int(y)
    if center:
        x_l0 -= patch_w_l0 // 2
        y_l0 -= patch_h_l0 // 2
    return x_l0, y_l0, patch_w_l0, patch_h_l0

def _map_l0_rect_to_mask_slice(
    mask: np.ndarray,
    wsi,
    x_l0: int,
    y_l0: int,
    patch_w_l0: int,
    patch_h_l0: int,
) -> np.ndarray | None:
    wsi_w, wsi_h = wsi.shapes[0][:2]
    sx = mask.shape[1] / wsi_w
    sy = mask.shape[0] / wsi_h

    x0 = int(x_l0 * sx)
    y0 = int(y_l0 * sy)
    x1 = int((x_l0 + patch_w_l0) * sx)
    y1 = int((y_l0 + patch_h_l0) * sy)

    x0 = max(0, min(x0, mask.shape[1]))
    x1 = max(0, min(x1, mask.shape[1]))
    y0 = max(0, min(y0, mask.shape[0]))
    y1 = max(0, min(y1, mask.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return None
    return mask[y0:y1, x0:x1]

def tls_raster_foreground_pixels(
    raster_mask: np.ndarray,
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
) -> int:
    x_l0, y_l0, pw, ph = _level0_patch_rect(wsi, x, y, patch_size, spacing, center=center)
    region = _map_l0_rect_to_mask_slice(raster_mask, wsi, x_l0, y_l0, pw, ph)
    if region is None:
        return 0
    return int(np.count_nonzero(region > 0))

def _iter_sliding_patch_grid(
    wsi,
    *,
    patch_size: int,
    spacing: float,
    overlap: tuple[int, int] = (0, 0),
    offset: tuple[int, int] = (0, 0),
    center: bool = False,
    grid_stride: int = 1,
) -> Iterator[dict[str, Any]]:
    """Replicate wholeslidedata SlidingPatchCommander level-0 grid."""
    grid_stride = max(1, int(grid_stride))
    ratio = int(wsi.get_downsampling_from_spacing(spacing))
    x_dims, y_dims = wsi.shapes[0][:2]

    step_shape = (patch_size - overlap[0], patch_size - overlap[1])
    step_shape_scaled = (step_shape[0] * ratio * grid_stride, step_shape[1] * ratio * grid_stride)
    offset_scaled = (offset[0] % step_shape_scaled[0], offset[1] % step_shape_scaled[1])
    patch_shape_scaled = (patch_size * ratio, patch_size * ratio)

    if center:
        first_coord = [
            offset_scaled[0]
            - (
                (offset_scaled[0] + patch_shape_scaled[0] // 2) // step_shape_scaled[0]
                - (1 if (offset_scaled[0] + patch_shape_scaled[0] // 2) % step_shape_scaled[0] == 0 else 0)
            )
            * step_shape_scaled[0],
            offset_scaled[1]
            - (
                (offset_scaled[1] + patch_shape_scaled[1] // 2) // step_shape_scaled[1]
                - (1 if (offset_scaled[1] + patch_shape_scaled[1] // 2) % step_shape_scaled[1] == 0 else 0)
            )
            * step_shape_scaled[1],
        ]
    else:
        first_coord = [
            offset_scaled[0]
            - (
                (offset_scaled[0] + patch_shape_scaled[0]) // step_shape_scaled[0]
                - (1 if (offset_scaled[0] + patch_shape_scaled[0]) % step_shape_scaled[0] == 0 else 0)
            )
            * step_shape_scaled[0],
            offset_scaled[1]
            - (
                (offset_scaled[1] + patch_shape_scaled[1]) // step_shape_scaled[1]
                - (1 if (offset_scaled[1] + patch_shape_scaled[1]) % step_shape_scaled[1] == 0 else 0)
            )
            * step_shape_scaled[1],
        ]

    max_i = int(x_dims + offset_scaled[0] + patch_shape_scaled[0] // 2) if center else int(x_dims + offset_scaled[0])
    max_j = int(y_dims + offset_scaled[1] + patch_shape_scaled[0] // 2) if center else int(y_dims + offset_scaled[1])
    if center:
        max_i += patch_shape_scaled[0] // 2
        max_j += patch_shape_scaled[1] // 2

    for row in range(first_coord[1], max_j, step_shape_scaled[1]):
        for col in range(first_coord[0], max_i, step_shape_scaled[0]):
            yield {"x": col, "y": row, "center": center, "spacing": spacing}

def iter_tls_sliding_patch_coords(
    wsi,
    tls_raster_mask: np.ndarray,
    *,
    patch_size: int,
    spacing: float,
    min_foreground_pixels: int = 1,
    overlap: tuple[int, int] = (0, 0),
    offset: tuple[int, int] = (0, 0),
    center: bool = False,
    grid_stride: int = 1,
) -> Iterator[dict[str, Any]]:
    """Sliding grid @ spacing; keep patches whose TLS raster mask has foreground."""
    min_foreground_pixels = max(1, int(min_foreground_pixels))
    for info in _iter_sliding_patch_grid(
        wsi,
        patch_size=patch_size,
        spacing=spacing,
        overlap=overlap,
        offset=offset,
        center=center,
        grid_stride=grid_stride,
    ):
        if (
            tls_raster_foreground_pixels(
                tls_raster_mask,
                wsi,
                int(info["x"]),
                int(info["y"]),
                patch_size,
                spacing,
                center=bool(info.get("center", False)),
            )
            < min_foreground_pixels
        ):
            continue
        yield info

def iter_xml_tls_sliding_patch_coords(
    wsi,
    wsa,
    *,
    patch_size: int,
    spacing: float,
    min_foreground_pixels: int = 1,
    overlap: tuple[int, int] = (0, 0),
    offset: tuple[int, int] = (0, 0),
    center: bool = False,
    grid_stride: int = 1,
) -> Iterator[dict[str, Any]]:
    """Sliding grid @ spacing; keep patches with tls/gc pixels from ASAP XML."""
    min_foreground_pixels = max(1, int(min_foreground_pixels))
    for info in _iter_sliding_patch_grid(
        wsi,
        patch_size=patch_size,
        spacing=spacing,
        overlap=overlap,
        offset=offset,
        center=center,
        grid_stride=grid_stride,
    ):
        sem = semantic_mask_from_xml(
            wsa,
            wsi,
            int(info["x"]),
            int(info["y"]),
            patch_size,
            spacing,
            center=bool(info.get("center", False)),
        )
        if int(np.count_nonzero(sem > 0)) < min_foreground_pixels:
            continue
        yield info

def iter_hooknet_tls_patch_coords(
    wsi,
    *,
    patch_size: int,
    spacing: float,
    raster_mask: np.ndarray | None = None,
    wsa: Any = None,
    min_foreground_pixels: int = 1,
    overlap: tuple[int, int] = (0, 0),
    offset: tuple[int, int] = (0, 0),
    center: bool = False,
    grid_stride: int = 1,
) -> Iterator[dict[str, Any]]:
    """Benchmark patch coords: same spacing grid, TLS-supervised patches only."""
    if raster_mask is not None:
        yield from iter_tls_sliding_patch_coords(
            wsi,
            raster_mask,
            patch_size=patch_size,
            spacing=spacing,
            min_foreground_pixels=min_foreground_pixels,
            overlap=overlap,
            offset=offset,
            center=center,
            grid_stride=grid_stride,
        )
        return
    if wsa is not None:
        yield from iter_xml_tls_sliding_patch_coords(
            wsi,
            wsa,
            patch_size=patch_size,
            spacing=spacing,
            min_foreground_pixels=min_foreground_pixels,
            overlap=overlap,
            offset=offset,
            center=center,
            grid_stride=grid_stride,
        )
        return
    raise ValueError("HookNet-TLS extraction requires a TLS raster mask or tls/gc XML annotations")

def load_tls_raster_mask(path: Path) -> np.ndarray:
    """Load full-slide TLS raster mask (OpenSlide often fails on these TIFFs)."""
    import tifffile

    arr = tifffile.imread(path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return np.asarray(arr)

def semantic_mask_from_raster_array(
    mask: np.ndarray,
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
) -> np.ndarray:
    x_l0, y_l0, pw, ph = _level0_patch_rect(wsi, x, y, patch_size, spacing, center=center)
    region = _map_l0_rect_to_mask_slice(mask, wsi, x_l0, y_l0, pw, ph)
    if region is None:
        return np.zeros((patch_size, patch_size), dtype=np.int64)
    if region.shape != (patch_size, patch_size):
        region = np.array(
            Image.fromarray(region.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST),
            dtype=np.uint8,
        )
    out = np.zeros((patch_size, patch_size), dtype=np.int64)
    out[region > 0] = 1
    return out

def semantic_mask_from_raster(
    mask_path: Path,
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
    backend: str = "openslide",
    raster_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = raster_mask if raster_mask is not None else load_tls_raster_mask(mask_path)
    return semantic_mask_from_raster_array(
        mask,
        wsi,
        x,
        y,
        patch_size,
        spacing,
        center=center,
    )

def resolve_semantic_mask(
    sample: SplitSample,
    cfg: DatasetConfig,
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
    wsa: Any = _WSA_UNSET,
    backend: str = "openslide",
    raster_mask: np.ndarray | None = None,
    mask_source: str = "auto",
) -> np.ndarray:
    """Resolve a patch semantic mask.

    ``mask_source``:
      - ``official_xml`` / ``xml``: 4-class ASAP XML (tls/gc/rest) via ``load_official_wsa``
      - ``raster``: binary ``*_tls_mask.tif`` (TLS=1 only)
      - ``auto``: prefer official XML when present, else raster
    """
    source = str(mask_source).lower().strip()
    xml = xml_path_for_slide(cfg, sample.sample_id)

    def _mask_from_official_xml() -> np.ndarray | None:
        if xml is None or not Path(xml).is_file():
            return None
        active_wsa = wsa if wsa is not _WSA_UNSET else load_official_wsa(xml, backend=backend)
        if active_wsa is None:
            return None
        return semantic_mask_from_xml(
            active_wsa,
            wsi,
            x,
            y,
            patch_size,
            spacing,
            center=center,
        )

    def _mask_from_raster() -> np.ndarray | None:
        raster = sample.mask_path or tls_raster_mask_path(cfg, sample.sample_id)
        if raster is None or not Path(raster).is_file():
            return None
        return semantic_mask_from_raster(
            Path(raster),
            wsi,
            x,
            y,
            patch_size,
            spacing,
            center=center,
            backend=backend,
            raster_mask=raster_mask,
        )

    if source in {"official_xml", "xml"}:
        mask = _mask_from_official_xml()
        return mask if mask is not None else np.zeros((patch_size, patch_size), dtype=np.int64)

    if source == "raster":
        mask = _mask_from_raster()
        return mask if mask is not None else np.zeros((patch_size, patch_size), dtype=np.int64)

    mask = _mask_from_official_xml()
    if mask is not None:
        return mask
    raster_mask_out = _mask_from_raster()
    if raster_mask_out is not None:
        return raster_mask_out
    return np.zeros((patch_size, patch_size), dtype=np.int64)
