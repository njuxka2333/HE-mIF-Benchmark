"""Shared extracted-patch catalog (Cell-niches ``patches.csv`` geometry columns).

On-disk rows use the same crop fields as virtual staining:

``x_he, y_he, w_he, h_he, he_mpp, size_um, stride_um, he_relpath, patch_id``

``x_he..h_he`` / ``he_mpp`` are the **source** crop (WSI level-0 for TLS, native HE
for Cell-niches). The stored PNG may be resampled; ``size_um`` is the physical
window. TLS adds ``slide_id``, ``split``, ``mask_relpath``, ``grid_ix``,
``grid_iy``, and class-pixel counts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from data.hooknet_tls import PatchPlanRecord
from data.hooknet_tls_physical_grid import physical_fov_um

TLS_PATCH_FIELDS = [
    "slide_id",
    "patient_id",
    "split",
    "patch_id",
    "x_he",
    "y_he",
    "w_he",
    "h_he",
    "he_mpp",
    "size_um",
    "stride_um",
    "he_relpath",
    "mask_relpath",
    "grid_ix",
    "grid_iy",
    "count_tls",
    "count_gc",
    "count_rest",
    "valid_pixel_count",
    "valid_fraction",
]

OUTPUT_SIZE = 1024
REFERENCE_MPP = 0.25
SIZE_UM = physical_fov_um(OUTPUT_SIZE, REFERENCE_MPP)


def _he_relpath(split: str, image_relpath: str) -> str:
    name = Path(image_relpath).name if image_relpath else ""
    if not name:
        return ""
    posix = str(image_relpath).replace("\\", "/")
    if "/" in posix:
        return posix
    return f"{split}/images/{name}"


def _mask_relpath(split: str, mask_relpath: str, he_relpath: str) -> str:
    name = Path(mask_relpath).name if mask_relpath else Path(he_relpath).name
    if not name:
        return ""
    posix = str(mask_relpath).replace("\\", "/") if mask_relpath else ""
    if posix and "/" in posix:
        return posix
    return f"{split}/masks/{name}"


def recover_slide_scale(rows: Sequence[dict[str, Any]]) -> float:
    """Pixels per 256 µm cell from stored (grid index, origin) pairs."""
    vals: list[float] = []
    for row in rows:
        ix, iy = int(row["grid_ix"]), int(row["grid_iy"])
        x0 = int(row.get("x_he", 0))
        y0 = int(row.get("y_he", 0))
        if ix:
            vals.append(x0 / ix)
        if iy:
            vals.append(y0 / iy)
    if not vals:
        raise ValueError("cannot recover TLS physical scale from rows")
    vals.sort()
    return float(vals[len(vals) // 2])


def source_box(grid_ix: int, grid_iy: int, scale: float) -> tuple[int, int, int, int]:
    x0, x1 = int(round(grid_ix * scale)), int(round((grid_ix + 1) * scale))
    y0, y1 = int(round(grid_iy * scale)), int(round((grid_iy + 1) * scale))
    return x0, y0, x1 - x0, y1 - y0


def catalog_row_from_record(record: PatchPlanRecord, scale: float | None = None) -> dict[str, Any]:
    if record.source_width_l0 > 0 and record.source_height_l0 > 0:
        w, h = record.source_width_l0, record.source_height_l0
        x0, y0 = record.x_level0, record.y_level0
    else:
        if scale is None:
            raise ValueError(f"need scale to write {record.slide_id} ix={record.grid_ix}")
        x0, y0, w, h = source_box(record.grid_ix, record.grid_iy, scale)
        x0, y0 = record.x_level0, record.y_level0
    mpp = record.source_mpp if record.source_mpp > 0 else (SIZE_UM / w if w else 0.0)
    he_rel = _he_relpath(record.split, record.image_relpath)
    return {
        "slide_id": record.slide_id,
        "patient_id": record.patient_id,
        "split": record.split,
        "patch_id": Path(he_rel).stem if he_rel else f"{record.slide_id}_ix{record.grid_ix}_iy{record.grid_iy}",
        "x_he": x0,
        "y_he": y0,
        "w_he": w,
        "h_he": h,
        "he_mpp": mpp,
        "size_um": SIZE_UM,
        "stride_um": SIZE_UM,
        "he_relpath": he_rel,
        "mask_relpath": _mask_relpath(record.split, record.mask_relpath, he_rel),
        "grid_ix": record.grid_ix,
        "grid_iy": record.grid_iy,
        "count_tls": record.count_tls,
        "count_gc": record.count_gc,
        "count_rest": record.count_rest,
        "valid_pixel_count": record.valid_pixel_count,
        "valid_fraction": record.valid_fraction,
    }


def record_from_catalog_row(row: dict[str, Any]) -> PatchPlanRecord:
    w = int(float(row["w_he"]))
    h = int(float(row["h_he"]))
    mpp = float(row["he_mpp"])
    return PatchPlanRecord(
        slide_id=str(row["slide_id"]),
        patient_id=str(row.get("patient_id") or ""),
        split=str(row["split"]),
        grid_ix=int(row["grid_ix"]),
        grid_iy=int(row["grid_iy"]),
        x_level0=int(row["x_he"]),
        y_level0=int(row["y_he"]),
        patch_size=OUTPUT_SIZE,
        stride=OUTPUT_SIZE,
        wsi_path="",
        valid_pixel_count=int(float(row.get("valid_pixel_count") or 0)),
        count_tls=int(float(row.get("count_tls") or 0)),
        count_gc=int(float(row.get("count_gc") or 0)),
        count_rest=int(float(row.get("count_rest") or 0)),
        valid_fraction=float(row.get("valid_fraction") or 0.0),
        image_relpath=str(row.get("he_relpath") or ""),
        mask_relpath=str(row.get("mask_relpath") or ""),
        source_width_l0=w,
        source_height_l0=h,
        source_mpp=mpp,
    )


def write_tls_patches_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TLS_PATCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_tls_patches_csv(path: Path) -> list[PatchPlanRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    if "x_he" not in rows[0]:
        raise ValueError(f"{path} is not a patches.csv catalog (missing x_he)")
    return [record_from_catalog_row(row) for row in rows]
