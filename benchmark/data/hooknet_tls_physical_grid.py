"""Canonical-physical-grid extraction for materialized HookNet-TLS patches.

Every output patch is ``output_size`` pixels at ``reference_mpp``.  The source
level-0 read size is slide-specific, so the physical grid cells are disjoint
even when input WSI base MPP differs between slides.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from data.hooknet_tls import (
    IGNORE_LABEL,
    SUPERVISED_CLASS_IDS,
    PatchPlanRecord,
    SlidePatchPlan,
    _class_counts,
    patient_id_from_slide,
)
from data.hooknet_tls_wsd import load_official_wsa


def physical_fov_um(output_size: int, reference_mpp: float) -> float:
    return float(output_size) * float(reference_mpp)


def physical_grid_edges(index: int, *, source_mpp: float, fov_um: float) -> tuple[int, int]:
    """Adjacent indices share an edge exactly after integer level-0 rounding."""
    scale = float(fov_um) / float(source_mpp)
    return int(round(index * scale)), int(round((index + 1) * scale))


def physical_bounds_to_grid_indices(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    *,
    source_mpp: float,
    fov_um: float,
    epsilon: float = 1e-6,
) -> set[tuple[int, int]]:
    scale = float(source_mpp) / float(fov_um)
    ix0, iy0 = math.floor(min_x * scale), math.floor(min_y * scale)
    ix1, iy1 = math.floor((max_x - epsilon) * scale), math.floor((max_y - epsilon) * scale)
    return {(ix, iy) for iy in range(iy0, iy1 + 1) for ix in range(ix0, ix1 + 1)}


def _source_mpp(wsi_path: Path, backend: str) -> float:
    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    wsi = WholeSlideImage(str(wsi_path), backend=backend)
    try:
        mpp = float(wsi.spacings[0])
    finally:
        wsi.close()
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError(f"Invalid base MPP {mpp!r} for {wsi_path}")
    return mpp


def rasterize_physical_tile(
    wsa,
    *,
    x_level0: int,
    y_level0: int,
    read_width_l0: int,
    read_height_l0: int,
    output_size: int,
) -> np.ndarray:
    """Rasterize a level-0 rect then nearest-resize to canonical output pixels."""
    raw = np.full((read_height_l0, read_width_l0), IGNORE_LABEL, dtype=np.uint8)
    center = (x_level0 + read_width_l0 // 2, y_level0 + read_height_l0 // 2)
    annotations = wsa.select_annotations(center[0], center[1], read_width_l0 - 1, read_height_l0 - 1)
    for annotation in annotations:
        value = int(annotation.label.value)
        if value not in SUPERVISED_CLASS_IDS:
            continue
        coords = np.asarray(annotation.coordinates, dtype=np.float64)
        pts = np.round(coords - np.array([x_level0, y_level0])).astype(np.int32)
        cv2.fillPoly(raw, [pts], color=value)
        for hole in getattr(annotation, "holes", None) or ():
            hpts = np.round(np.asarray(hole, dtype=np.float64) - np.array([x_level0, y_level0])).astype(np.int32)
            cv2.fillPoly(raw, [hpts], color=IGNORE_LABEL)
    if raw.shape != (output_size, output_size):
        raw = cv2.resize(raw, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return raw


def build_physical_slide_plan(
    *,
    slide_id: str,
    split: str,
    wsi_path: Path,
    xml_path: Path,
    output_size: int = 1024,
    reference_mpp: float = 0.25,
    backend: str = "openslide",
) -> tuple[SlidePatchPlan, float]:
    """Plan non-overlapping  physical-FOV cells and canonical masks for one WSI."""
    source_mpp = _source_mpp(wsi_path, backend)
    fov_um = physical_fov_um(output_size, reference_mpp)
    wsa = load_official_wsa(xml_path, backend=backend)
    if wsa is None:
        raise RuntimeError(f"No official annotations in {xml_path}")

    candidates: set[tuple[int, int]] = set()
    for annotation in wsa.sampling_annotations:
        b = annotation.bounds
        candidates |= physical_bounds_to_grid_indices(
            float(b[0]), float(b[1]), float(b[2]), float(b[3]),
            source_mpp=source_mpp, fov_um=fov_um,
        )

    records: list[PatchPlanRecord] = []
    for ix, iy in sorted(candidates):
        x0, x1 = physical_grid_edges(ix, source_mpp=source_mpp, fov_um=fov_um)
        y0, y1 = physical_grid_edges(iy, source_mpp=source_mpp, fov_um=fov_um)
        mask = rasterize_physical_tile(
            wsa, x_level0=x0, y_level0=y0, read_width_l0=x1 - x0,
            read_height_l0=y1 - y0, output_size=output_size,
        )
        valid, tls, gc, rest = _class_counts(mask)
        if valid <= 0:
            continue
        records.append(PatchPlanRecord(
            slide_id=slide_id, patient_id=patient_id_from_slide(slide_id), split=split,
            grid_ix=ix, grid_iy=iy, x_level0=x0, y_level0=y0,
            patch_size=output_size, stride=output_size, wsi_path=str(Path(wsi_path).resolve()),
            valid_pixel_count=valid, count_tls=tls, count_gc=gc, count_rest=rest,
            valid_fraction=valid / float(output_size * output_size),
            source_width_l0=x1 - x0,
            source_height_l0=y1 - y0,
            source_mpp=source_mpp,
        ))
    return (
        SlidePatchPlan(
            slide_id=slide_id, patient_id=patient_id_from_slide(slide_id), split=split,
            wsi_path=Path(wsi_path), xml_path=Path(xml_path), patch_size=output_size,
            stride=output_size, candidate_cell_count=len(candidates), records=records,
        ),
        source_mpp,
    )


def materialize_physical_record(
    record: PatchPlanRecord,
    *,
    wsi,
    wsa,
    source_mpp: float,
    reference_mpp: float,
    images_dir: Path,
    masks_dir: Path,
    png_compress_level: int = 1,
) -> PatchPlanRecord:
    """Read a canonical physical cell and save paired RGB/mask PNGs."""
    if record.source_width_l0 > 0 and record.source_height_l0 > 0:
        x0, y0 = record.x_level0, record.y_level0
        width, height = record.source_width_l0, record.source_height_l0
    else:
        fov_um = physical_fov_um(record.patch_size, reference_mpp)
        x0, x1 = physical_grid_edges(record.grid_ix, source_mpp=source_mpp, fov_um=fov_um)
        y0, y1 = physical_grid_edges(record.grid_iy, source_mpp=source_mpp, fov_um=fov_um)
        width, height = x1 - x0, y1 - y0
    rgb = np.asarray(
        wsi.get_patch(x0, y0, width, height, spacing=float(wsi.spacings[0]), center=False),
        dtype=np.uint8,
    )
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[:2] != (record.patch_size, record.patch_size):
        rgb = np.asarray(
            Image.fromarray(rgb[..., :3]).resize((record.patch_size, record.patch_size), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    mask = rasterize_physical_tile(
        wsa, x_level0=x0, y_level0=y0, read_width_l0=width, read_height_l0=height,
        output_size=record.patch_size,
    )
    stem = f"{record.slide_id}_ix{record.grid_ix}_iy{record.grid_iy}"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image_path, mask_path = images_dir / f"{stem}.png", masks_dir / f"{stem}.png"
    Image.fromarray(rgb).save(image_path, compress_level=png_compress_level)
    Image.fromarray(mask, mode="L").save(mask_path, compress_level=png_compress_level)
    return replace(
        record,
        image_relpath=f"{record.split}/images/{stem}.png",
        mask_relpath=f"{record.split}/masks/{stem}.png",
        source_width_l0=width,
        source_height_l0=height,
        source_mpp=source_mpp,
    )
