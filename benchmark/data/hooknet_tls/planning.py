"""Deterministic level-0 fixed-grid HookNet-TLS patch planning and materialization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .annotations import load_official_wsa, xml_path_for_slide
from data.loader import DatasetConfig

IGNORE_LABEL = 255
SUPERVISED_CLASS_IDS = (1, 2, 3)
DEFAULT_PATCH_SIZE_L0 = 1024
GRID_EPSILON = 1e-6
DEFAULT_REST_KEEP_FRACTION = 0.10
DEFAULT_MANIFEST_RANDOM_SEED = 42

TCGA_CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


@dataclass(frozen=True)
class PatchPlanRecord:
    """One retained grid cell; shared by preview and extract modes."""

    slide_id: str
    patient_id: str
    split: str
    grid_ix: int
    grid_iy: int
    x_level0: int
    y_level0: int
    patch_size: int
    stride: int
    wsi_path: str
    valid_pixel_count: int
    count_tls: int
    count_gc: int
    count_rest: int
    valid_fraction: float
    image_relpath: str = ""
    mask_relpath: str = ""
    source_width_l0: int = 0
    source_height_l0: int = 0
    source_mpp: float = 0.0

    @property
    def grid_key(self) -> tuple[str, int, int]:
        return (self.slide_id, self.grid_ix, self.grid_iy)


@dataclass
class SlidePatchPlan:
    slide_id: str
    patient_id: str
    split: str
    wsi_path: Path
    xml_path: Path
    patch_size: int
    stride: int
    candidate_cell_count: int
    records: list[PatchPlanRecord] = field(default_factory=list)


def patient_id_from_slide(slide_id: str) -> str:
    match = TCGA_CASE_RE.search(slide_id)
    return match.group(1).upper() if match else slide_id


def polygon_bounds_to_grid_indices(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    patch_size: int,
    *,
    epsilon: float = GRID_EPSILON,
) -> set[tuple[int, int]]:
    """Map polygon AABB to unique grid indices (level-0, half-open tiles)."""
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    ix0 = int(math.floor(min_x / patch_size))
    iy0 = int(math.floor(min_y / patch_size))
    ix1 = int(math.floor((max_x - epsilon) / patch_size))
    iy1 = int(math.floor((max_y - epsilon) / patch_size))
    cells: set[tuple[int, int]] = set()
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            cells.add((ix, iy))
    return cells


def collect_candidate_cells(
    wsa,
    patch_size: int,
) -> set[tuple[int, int]]:
    """BBox-only culling over official sampling annotations."""
    cells: set[tuple[int, int]] = set()
    for annotation in wsa.sampling_annotations:
        bounds = annotation.bounds
        min_x, min_y, max_x, max_y = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        )
        cells |= polygon_bounds_to_grid_indices(min_x, min_y, max_x, max_y, patch_size)
    return cells


def rasterize_patch_mask(
    wsa,
    x_level0: int,
    y_level0: int,
    patch_size: int,
    *,
    sampler=None,
) -> np.ndarray:
    """Rasterize tls/gc/rest into a P×P uint8 mask (255 = ignore).

    Uses OpenCV polygon fill (matches wholeslidedata SegmentationPatchLabelSampler
    for HookNet-TLS official annotations, ~40× faster than per-patch sampler).
    ``sampler`` is accepted for API compatibility but unused.
    """
    del sampler  # kept for call-site compatibility
    import cv2

    x0 = int(x_level0)
    y0 = int(y_level0)
    p = int(patch_size)
    x1 = x0 + p
    y1 = y0 + p
    mask = np.full((p, p), IGNORE_LABEL, dtype=np.uint8)

    for annotation in wsa.sampling_annotations:
        bounds = annotation.bounds
        if (
            float(bounds[2]) <= x0
            or float(bounds[0]) >= x1
            or float(bounds[3]) <= y0
            or float(bounds[1]) >= y1
        ):
            continue
        value = int(annotation.label.value)
        coords = np.asarray(annotation.coordinates, dtype=np.float64)
        local = np.empty_like(coords)
        local[:, 0] = coords[:, 0] - x0
        local[:, 1] = coords[:, 1] - y0
        pts = np.round(local).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=value)
        holes = getattr(annotation, "holes", None) or ()
        for hole in holes:
            hole_coords = np.asarray(hole, dtype=np.float64)
            hole_local = np.empty_like(hole_coords)
            hole_local[:, 0] = hole_coords[:, 0] - x0
            hole_local[:, 1] = hole_coords[:, 1] - y0
            cv2.fillPoly(
                mask,
                [np.round(hole_local).astype(np.int32)],
                color=int(IGNORE_LABEL),
            )

    out = np.full((p, p), IGNORE_LABEL, dtype=np.uint8)
    for cls in SUPERVISED_CLASS_IDS:
        out[mask == cls] = cls
    return out


def _class_counts(mask: np.ndarray) -> tuple[int, int, int, int]:
    count_tls = int(np.count_nonzero(mask == 1))
    count_gc = int(np.count_nonzero(mask == 2))
    count_rest = int(np.count_nonzero(mask == 3))
    valid = count_tls + count_gc + count_rest
    return valid, count_tls, count_gc, count_rest


def is_positive_patch(record: PatchPlanRecord) -> bool:
    """TLS or GC present (any pixel count)."""
    return record.count_tls > 0 or record.count_gc > 0


def is_rest_only_patch(record: PatchPlanRecord) -> bool:
    """No TLS/GC and at least one Rest pixel."""
    return record.count_tls == 0 and record.count_gc == 0 and record.count_rest > 0


def rest_keep_count(rest_candidate_count: int, rest_keep_fraction: float) -> int:
    if rest_candidate_count <= 0:
        return 0
    if rest_keep_fraction <= 0.0:
        return 0
    if rest_keep_fraction >= 1.0:
        return rest_candidate_count
    return min(rest_candidate_count, int(round(rest_keep_fraction * rest_candidate_count)))


def apply_manifest_selection(
    records: Sequence[PatchPlanRecord],
    *,
    rest_keep_fraction: float,
    random_seed: int = DEFAULT_MANIFEST_RANDOM_SEED,
) -> tuple[list[PatchPlanRecord], dict[str, Any]]:
    """Positive patches (all) + deterministic subsample of Rest-only patches."""
    if not 0.0 <= rest_keep_fraction <= 1.0:
        raise ValueError(f"rest_keep_fraction must be in [0, 1], got {rest_keep_fraction}")

    positive: list[PatchPlanRecord] = []
    rest_only: list[PatchPlanRecord] = []
    for record in records:
        if is_positive_patch(record):
            positive.append(record)
        elif is_rest_only_patch(record):
            rest_only.append(record)
        elif record.valid_pixel_count > 0:
            raise ValueError(
                f"patch {record.slide_id} ({record.grid_ix},{record.grid_iy}) has supervised "
                f"pixels but is neither positive nor rest-only: "
                f"tls={record.count_tls} gc={record.count_gc} rest={record.count_rest}"
            )

    k = rest_keep_count(len(rest_only), rest_keep_fraction)
    if k <= 0:
        selected_rest: list[PatchPlanRecord] = []
    elif k >= len(rest_only):
        selected_rest = list(rest_only)
    else:
        rng = np.random.default_rng(int(random_seed))
        chosen = np.sort(rng.choice(len(rest_only), size=k, replace=False))
        selected_rest = [rest_only[int(i)] for i in chosen]

    final_records = positive + selected_rest
    stats = {
        "positive_patch_count": len(positive),
        "rest_candidate_count": len(rest_only),
        "selected_rest_count": len(selected_rest),
        "final_patch_count": len(final_records),
        "rest_keep_fraction": float(rest_keep_fraction),
        "random_seed": int(random_seed),
    }
    return final_records, stats


def format_split_manifest_summary(split: str, stats: dict[str, Any]) -> str:
    return (
        f"Split: {split}\n\n"
        f"Positive patches:\n    {stats['positive_patch_count']}\n\n"
        f"Rest-only candidates:\n    {stats['rest_candidate_count']}\n\n"
        f"Selected Rest-only:\n    {stats['selected_rest_count']}\n\n"
        f"Final dataset:\n    {stats['final_patch_count']}\n"
    )


def write_manifest_summary_json(path: Path, split: str, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, **stats}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fixed_grid_build_config(config_path: Path | None) -> dict[str, Any]:
    """Optional YAML: rest_keep_fraction, random_seed."""
    if config_path is None:
        return {}
    import yaml

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"fixed-grid config must be a mapping: {config_path}")
    return payload


def read_rgb_patch_from_open_wsi(
    wsi,
    x_level0: int,
    y_level0: int,
    patch_size: int,
) -> np.ndarray:
    """Read RGB patch from an already-open WholeSlideImage handle."""
    spacing = float(wsi.spacings[0])
    wsi_w, wsi_h = wsi.shapes[0][:2]
    x0 = max(0, int(x_level0))
    y0 = max(0, int(y_level0))
    x1 = min(wsi_w, x0 + patch_size)
    y1 = min(wsi_h, y0 + patch_size)
    read_w = max(0, x1 - x0)
    read_h = max(0, y1 - y0)
    if read_w == 0 or read_h == 0:
        return np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    patch = np.asarray(
        wsi.get_patch(x0, y0, read_w, read_h, spacing=spacing, center=False),
        dtype=np.uint8,
    )
    if patch.ndim == 2:
        patch = np.stack([patch] * 3, axis=-1)
    if patch.shape[0] != read_h or patch.shape[1] != read_w:
        patch = np.asarray(
            Image.fromarray(patch).resize((read_w, read_h), Image.BILINEAR),
            dtype=np.uint8,
        )
    out = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    out[:read_h, :read_w] = patch[..., :3]
    return out


def read_wsi_patch_level0(
    wsi_path: Path,
    x_level0: int,
    y_level0: int,
    patch_size: int,
    *,
    backend: str = "openslide",
) -> np.ndarray:
    """Read RGB patch at level-0 top-left (x, y); pad with white if clipped."""
    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    wsi = WholeSlideImage(str(wsi_path), backend=backend)
    try:
        return read_rgb_patch_from_open_wsi(wsi, x_level0, y_level0, patch_size)
    finally:
        wsi.close()


def pad_mask_to_tile(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Ensure P×P; clipped regions remain 255."""
    h, w = mask.shape[:2]
    if h == patch_size and w == patch_size:
        return mask
    out = np.full((patch_size, patch_size), IGNORE_LABEL, dtype=np.uint8)
    out[: min(h, patch_size), : min(w, patch_size)] = mask[: min(h, patch_size), : min(w, patch_size)]
    return out


def build_slide_patch_plan(
    *,
    slide_id: str,
    split: str,
    wsi_path: Path,
    xml_path: Path,
    patch_size: int = DEFAULT_PATCH_SIZE_L0,
    stride: int | None = None,
    backend: str = "openslide",
) -> SlidePatchPlan:
    """Plan all fixed-grid candidate patches for one slide (before split-level Rest subsampling)."""
    stride = int(stride if stride is not None else patch_size)
    if stride != patch_size:
        raise ValueError(f"stride must equal patch_size for non-overlap grid (got stride={stride}, P={patch_size})")

    wsa = load_official_wsa(xml_path, backend=backend)
    if wsa is None:
        raise RuntimeError(f"No official annotations in {xml_path}")

    candidate_cells = collect_candidate_cells(wsa, patch_size)
    records: list[PatchPlanRecord] = []
    patient = patient_id_from_slide(slide_id)
    wsi_path = Path(wsi_path).resolve()

    for ix, iy in sorted(candidate_cells):
        x0 = ix * patch_size
        y0 = iy * patch_size
        mask = rasterize_patch_mask(wsa, x0, y0, patch_size)
        valid, count_tls, count_gc, count_rest = _class_counts(mask)
        if valid <= 0:
            continue
        area = float(patch_size * patch_size)
        records.append(
            PatchPlanRecord(
                slide_id=slide_id,
                patient_id=patient,
                split=split,
                grid_ix=ix,
                grid_iy=iy,
                x_level0=x0,
                y_level0=y0,
                patch_size=patch_size,
                stride=stride,
                wsi_path=str(wsi_path),
                valid_pixel_count=valid,
                count_tls=count_tls,
                count_gc=count_gc,
                count_rest=count_rest,
                valid_fraction=valid / area,
            )
        )

    return SlidePatchPlan(
        slide_id=slide_id,
        patient_id=patient,
        split=split,
        wsi_path=wsi_path,
        xml_path=Path(xml_path),
        patch_size=patch_size,
        stride=stride,
        candidate_cell_count=len(candidate_cells),
        records=records,
    )


def plan_records_to_rows(records: Sequence[PatchPlanRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]


def write_selected_tiles_csv(path: Path, records: Sequence[PatchPlanRecord]) -> None:
    """Write TLS tiles in the shared ``patches.csv`` geometry columns."""
    from collections import defaultdict

    from data.patch_catalog import catalog_row_from_record, recover_slide_scale, write_tls_patches_csv

    grouped: dict[str, list[PatchPlanRecord]] = defaultdict(list)
    for record in records:
        grouped[record.slide_id].append(record)
    scales: dict[str, float] = {}
    for slide_id, group in grouped.items():
        if any(r.source_width_l0 > 0 for r in group):
            continue
        dummy = [
            {
                "grid_ix": r.grid_ix,
                "grid_iy": r.grid_iy,
                "x_he": r.x_level0,
                "y_he": r.y_level0,
            }
            for r in group
        ]
        scales[slide_id] = recover_slide_scale(dummy)
    rows = [catalog_row_from_record(r, scales.get(r.slide_id)) for r in records]
    write_tls_patches_csv(path, rows)


def load_selected_tiles_csv(path: Path) -> list[PatchPlanRecord]:
    """Load TLS tiles from a ``patches.csv`` catalog."""
    from data.patch_catalog import load_tls_patches_csv

    return load_tls_patches_csv(path)


def manifest_determinism_hash(records: Sequence[PatchPlanRecord]) -> str:
    payload = [
        {
            "slide_id": r.slide_id,
            "grid_ix": r.grid_ix,
            "grid_iy": r.grid_iy,
            "x_level0": r.x_level0,
            "y_level0": r.y_level0,
            "valid_pixel_count": r.valid_pixel_count,
            "count_tls": r.count_tls,
            "count_gc": r.count_gc,
            "count_rest": r.count_rest,
        }
        for r in sorted(records, key=lambda x: (x.grid_iy, x.grid_ix))
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_coordinate_uniqueness(records: Sequence[PatchPlanRecord]) -> dict[str, Any]:
    keys = [r.grid_key for r in records]
    unique = len(set(keys))
    return {
        "passed": unique == len(keys),
        "rows": len(keys),
        "unique_keys": unique,
        "duplicates": len(keys) - unique,
    }


def check_non_overlap(records: Sequence[PatchPlanRecord], patch_size: int) -> dict[str, Any]:
    issues: list[str] = []
    for r in records:
        if r.x_level0 % patch_size != 0 or r.y_level0 % patch_size != 0:
            issues.append(f"alignment:{r.grid_ix},{r.grid_iy}")
        if r.stride != patch_size:
            issues.append(f"stride:{r.grid_ix},{r.grid_iy}")
        if r.x_level0 != r.grid_ix * patch_size or r.y_level0 != r.grid_iy * patch_size:
            issues.append(f"grid_coord:{r.grid_ix},{r.grid_iy}")
    rects = [(r.x_level0, r.y_level0, r.x_level0 + patch_size, r.y_level0 + patch_size) for r in records]
    overlap_pairs = 0
    for i in range(len(rects)):
        x0a, y0a, x1a, y1a = rects[i]
        for j in range(i + 1, len(rects)):
            x0b, y0b, x1b, y1b = rects[j]
            iw = min(x1a, x1b) - max(x0a, x0b)
            ih = min(y1a, y1b) - max(y0a, y0b)
            if iw > 0 and ih > 0:
                overlap_pairs += 1
    return {
        "passed": not issues and overlap_pairs == 0,
        "alignment_issues": issues,
        "empirical_overlap_pairs": overlap_pairs,
    }


def valid_fraction_percentiles(records: Sequence[PatchPlanRecord]) -> dict[str, float]:
    if not records:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    fracs = np.array([r.valid_fraction for r in records], dtype=np.float64)
    return {
        "p50": float(np.percentile(fracs, 50)),
        "p90": float(np.percentile(fracs, 90)),
        "p95": float(np.percentile(fracs, 95)),
        "p99": float(np.percentile(fracs, 99)),
    }


def summarize_patch_plan(plan: SlidePatchPlan) -> dict[str, Any]:
    records = plan.records
    uniq = check_coordinate_uniqueness(records)
    overlap = check_non_overlap(records, plan.patch_size)
    det_hash = manifest_determinism_hash(records)
    return {
        "slide_id": plan.slide_id,
        "patient_id": plan.patient_id,
        "split": plan.split,
        "patch_size": plan.patch_size,
        "stride": plan.stride,
        "candidate_cell_count": plan.candidate_cell_count,
        "retained_cell_count": len(records),
        "tls_positive_patches": sum(1 for r in records if r.count_tls > 0),
        "gc_positive_patches": sum(1 for r in records if r.count_gc > 0),
        "rest_positive_patches": sum(1 for r in records if r.count_rest > 0),
        "valid_fraction_percentiles": valid_fraction_percentiles(records),
        "uniqueness_check": uniq,
        "non_overlap_check": overlap,
        "determinism_sha256": det_hash,
    }


def _annotation_polygons_for_draw(wsa) -> list[tuple[np.ndarray, int]]:
    """Return exterior coordinates (level-0) and class id for visualization."""
    polys: list[tuple[np.ndarray, int]] = []
    for ann in wsa.sampling_annotations:
        cls = int(ann.label.value)
        if cls not in SUPERVISED_CLASS_IDS:
            continue
        coords = np.asarray(ann.coordinates, dtype=np.float64)
        if coords.ndim == 2 and len(coords) >= 3:
            polys.append((coords, cls))
    return polys


def _mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Colorize supervised classes; ignore= dark gray."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == IGNORE_LABEL] = (40, 40, 40)
    rgb[mask == 1] = (0, 180, 0)
    rgb[mask == 2] = (220, 0, 180)
    rgb[mask == 3] = (240, 200, 0)
    return rgb


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = _mask_to_color(mask)
    base = rgb.astype(np.float32)
    over = color.astype(np.float32)
    supervised = mask != IGNORE_LABEL
    out = base.copy()
    out[supervised] = (1.0 - alpha) * base[supervised] + alpha * over[supervised]
    return np.clip(out, 0, 255).astype(np.uint8)


def select_zoom_example_records(
    records: Sequence[PatchPlanRecord],
    *,
    max_examples: int = 6,
) -> list[PatchPlanRecord]:
    """Pick diverse retained tiles for preview zoom panels."""
    if not records:
        return []
    chosen: list[PatchPlanRecord] = []
    seen: set[tuple[int, int]] = set()

    def _take(record: PatchPlanRecord | None) -> None:
        if record is None or len(chosen) >= max_examples:
            return
        key = (record.grid_ix, record.grid_iy)
        if key in seen:
            return
        seen.add(key)
        chosen.append(record)

    by_valid = sorted(records, key=lambda r: r.valid_pixel_count, reverse=True)
    tls_only = [r for r in records if r.count_tls > 0 and r.count_gc == 0 and r.count_rest == 0]
    gc_only = [r for r in records if r.count_gc > 0 and r.count_tls == 0 and r.count_rest == 0]
    rest_only = [r for r in records if r.count_rest > 0 and r.count_tls == 0 and r.count_gc == 0]
    multi = [r for r in records if sum((r.count_tls > 0, r.count_gc > 0, r.count_rest > 0)) >= 2]
    low_cov = sorted(records, key=lambda r: r.valid_fraction)

    _take(by_valid[0] if by_valid else None)
    _take(tls_only[0] if tls_only else None)
    _take(gc_only[0] if gc_only else None)
    _take(rest_only[0] if rest_only else None)
    _take(multi[0] if multi else None)
    _take(low_cov[0] if low_cov else None)
    for r in by_valid:
        if len(chosen) >= max_examples:
            break
        _take(r)
    return chosen


def render_slide_overview_png(
    plan: SlidePatchPlan,
    output_path: Path,
    *,
    backend: str = "openslide",
    max_thumb_px: int = 2048,
) -> None:
    """Overview: downsampled tissue + polygon contours + retained grid rectangles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    wsa = load_official_wsa(plan.xml_path, backend=backend)
    if wsa is None:
        raise RuntimeError(f"Cannot load annotations from {plan.xml_path}")

    polys = _annotation_polygons_for_draw(wsa)
    if not plan.records:
        xmin = ymin = 0
        xmax = ymax = plan.patch_size
    else:
        xs = [r.x_level0 for r in plan.records] + [r.x_level0 + plan.patch_size for r in plan.records]
        ys = [r.y_level0 for r in plan.records] + [r.y_level0 + plan.patch_size for r in plan.records]
        for coords, _ in polys:
            xs.extend(coords[:, 0].tolist())
            ys.extend(coords[:, 1].tolist())
        pad = plan.patch_size * 2
        xmin, xmax = min(xs) - pad, max(xs) + pad
        ymin, ymax = min(ys) - pad, max(ys) + pad

    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    wsi = WholeSlideImage(str(plan.wsi_path), backend=backend)
    try:
        wsi_w, wsi_h = wsi.shapes[0][:2]
        x0 = int(max(0, math.floor(xmin)))
        y0 = int(max(0, math.floor(ymin)))
        x1 = int(min(wsi_w, math.ceil(xmax)))
        y1 = int(min(wsi_h, math.ceil(ymax)))
        region_w = max(1, x1 - x0)
        region_h = max(1, y1 - y0)
        downsample = max(1.0, max(region_w, region_h) / float(max_thumb_px))
        import openslide

        slide = openslide.OpenSlide(str(plan.wsi_path))
        try:
            level = int(slide.get_best_level_for_downsample(downsample))
            level_ds = float(slide.level_downsamples[level])
            read_w = max(1, int(math.ceil(region_w / level_ds)))
            read_h = max(1, int(math.ceil(region_h / level_ds)))
            thumb = np.asarray(slide.read_region((x0, y0), level, (read_w, read_h)).convert("RGB"))
        finally:
            slide.close()
    finally:
        wsi.close()

    read_w, read_h = thumb.shape[1], thumb.shape[0]
    fig_w = max(6.0, read_w / 200)
    fig_h = max(6.0, read_h / 200)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(thumb, extent=(x0, x0 + region_w, y0 + region_h, y0))
    ax.set_xlim(x0, x0 + region_w)
    ax.set_ylim(y0 + region_h, y0)

    class_colors = {1: "#00b400", 2: "#dc00b4", 3: "#f0c800"}
    for coords, cls in polys:
        ax.plot(coords[:, 0], coords[:, 1], color=class_colors.get(cls, "white"), linewidth=1.2, alpha=0.95)
        ax.plot([coords[-1, 0], coords[0, 0]], [coords[-1, 1], coords[0, 1]], color=class_colors.get(cls, "white"), linewidth=1.2, alpha=0.95)

    rects = [
        Rectangle(
            (r.x_level0, r.y_level0),
            plan.patch_size,
            plan.patch_size,
            linewidth=0.8,
            edgecolor="cyan",
            facecolor="none",
            alpha=0.85,
        )
        for r in plan.records
    ]
    if rects:
        ax.add_collection(PatchCollection(rects, match_original=True))

    ax.set_title(f"{plan.slide_id}  retained={len(plan.records)}  candidates={plan.candidate_cell_count}")
    ax.set_xlabel("level-0 x")
    ax.set_ylabel("level-0 y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def materialize_patch_record(
    record: PatchPlanRecord,
    plan: SlidePatchPlan,
    *,
    images_dir: Path,
    masks_dir: Path,
    backend: str = "openslide",
    wsi=None,
    wsa=None,
    skip_existing: bool = False,
    png_compress_level: int = 1,
) -> PatchPlanRecord:
    """Write lossless uint8 PNG image + mask; return record with relative paths.

    Pass open ``wsi`` / ``wsa`` handles when materializing many patches from one
    slide — opening OpenSlide per patch is extremely slow.
    """
    stem = f"{record.slide_id}_ix{record.grid_ix}_iy{record.grid_iy}"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"{stem}.png"
    mask_path = masks_dir / f"{stem}.png"
    if skip_existing and image_path.is_file() and mask_path.is_file():
        return replace(
            record,
            image_relpath=str(image_path.name),
            mask_relpath=str(mask_path.name),
        )

    owns_wsi = wsi is None
    if wsa is None:
        wsa = load_official_wsa(plan.xml_path, backend=backend)
    if wsa is None:
        raise RuntimeError(f"Missing WSA for {plan.slide_id}")
    if wsi is None:
        from wholeslidedata.image.wholeslideimage import WholeSlideImage

        wsi = WholeSlideImage(str(plan.wsi_path), backend=backend)
    try:
        mask = rasterize_patch_mask(wsa, record.x_level0, record.y_level0, record.patch_size)
        rgb = read_rgb_patch_from_open_wsi(
            wsi, record.x_level0, record.y_level0, record.patch_size
        )
        Image.fromarray(rgb).save(image_path, compress_level=int(png_compress_level))
        Image.fromarray(mask, mode="L").save(mask_path, compress_level=int(png_compress_level))
    finally:
        if owns_wsi:
            wsi.close()
    return replace(
        record,
        image_relpath=str(image_path.name),
        mask_relpath=str(mask_path.name),
    )


def slide_plan_stub_for_materialize(
    *,
    slide_id: str,
    split: str,
    wsi_path: Path,
    xml_path: Path,
    patch_size: int,
) -> SlidePatchPlan:
    """Lightweight plan with paths only — do not rebuild the candidate grid."""
    return SlidePatchPlan(
        slide_id=slide_id,
        patient_id=patient_id_from_slide(slide_id),
        split=split,
        wsi_path=Path(wsi_path),
        xml_path=Path(xml_path),
        patch_size=int(patch_size),
        stride=int(patch_size),
        candidate_cell_count=0,
        records=[],
    )


def _materialize_one_slide_group(
    *,
    slide_id: str,
    indexed_records: Sequence[tuple[int, PatchPlanRecord]],
    plan: SlidePatchPlan,
    images_dir: Path,
    masks_dir: Path,
    backend: str,
    skip_existing: bool,
    png_compress_level: int,
) -> tuple[list[tuple[int, PatchPlanRecord]], int, int]:
    """Open one WSI+WSA, write all patches for that slide. Returns (results, written, skipped)."""
    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    wsa = load_official_wsa(plan.xml_path, backend=backend)
    if wsa is None:
        raise RuntimeError(f"Missing WSA for {slide_id}")
    wsi = WholeSlideImage(str(plan.wsi_path), backend=backend)
    written = 0
    skipped = 0
    results: list[tuple[int, PatchPlanRecord]] = []
    try:
        for idx, record in indexed_records:
            stem = f"{record.slide_id}_ix{record.grid_ix}_iy{record.grid_iy}"
            image_path = images_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"
            existed = skip_existing and image_path.is_file() and mask_path.is_file()
            if existed:
                results.append(
                    (
                        idx,
                        replace(
                            record,
                            image_relpath=str(image_path.name),
                            mask_relpath=str(mask_path.name),
                        ),
                    )
                )
                skipped += 1
                continue
            mask = rasterize_patch_mask(
                wsa, record.x_level0, record.y_level0, record.patch_size
            )
            rgb = read_rgb_patch_from_open_wsi(
                wsi, record.x_level0, record.y_level0, record.patch_size
            )
            Image.fromarray(rgb).save(image_path, compress_level=int(png_compress_level))
            Image.fromarray(mask, mode="L").save(mask_path, compress_level=int(png_compress_level))
            results.append(
                (
                    idx,
                    replace(
                        record,
                        image_relpath=str(image_path.name),
                        mask_relpath=str(mask_path.name),
                    ),
                )
            )
            written += 1
    finally:
        wsi.close()
    return results, written, skipped


def materialize_selected_records(
    records: Sequence[PatchPlanRecord],
    *,
    plan_by_slide: dict[str, SlidePatchPlan],
    images_dir: Path,
    masks_dir: Path,
    backend: str = "openslide",
    skip_existing: bool = True,
    png_compress_level: int = 1,
    progress_every: int = 100,
    workers: int = 1,
) -> list[PatchPlanRecord]:
    """Materialize patches with one OpenSlide + WSA open per slide.

    ``workers>1`` processes slides in parallel (each worker owns one slide at a time).
    """
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[int, PatchPlanRecord]]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[record.slide_id].append((idx, record))

    out: list[PatchPlanRecord | None] = [None] * len(records)
    written = 0
    skipped = 0
    total = len(records)
    slide_ids = sorted(grouped.keys())
    workers = max(1, int(workers))

    def _consume(slide_id: str, results, w, s, slide_i: int) -> None:
        nonlocal written, skipped
        for idx, rec in results:
            out[idx] = rec
        written += w
        skipped += s
        done = written + skipped
        if progress_every > 0 and (done % progress_every == 0 or done == total or w + s > 0):
            print(
                f"  materialize {done}/{total} "
                f"(written={written} skipped={skipped}) "
                f"slide {slide_i}/{len(slide_ids)} {slide_id}",
                flush=True,
            )

    if workers == 1:
        for slide_i, slide_id in enumerate(slide_ids, start=1):
            results, w, s = _materialize_one_slide_group(
                slide_id=slide_id,
                indexed_records=grouped[slide_id],
                plan=plan_by_slide[slide_id],
                images_dir=images_dir,
                masks_dir=masks_dir,
                backend=backend,
                skip_existing=skip_existing,
                png_compress_level=png_compress_level,
            )
            _consume(slide_id, results, w, s, slide_i)
    else:
        # ProcessPool needs top-level picklable worker; use inline path tuples.
        from multiprocessing import get_context

        jobs = []
        for slide_id in slide_ids:
            plan = plan_by_slide[slide_id]
            jobs.append(
                (
                    slide_id,
                    grouped[slide_id],
                    str(plan.wsi_path),
                    str(plan.xml_path),
                    plan.split,
                    plan.patch_size,
                    str(images_dir),
                    str(masks_dir),
                    backend,
                    skip_existing,
                    png_compress_level,
                )
            )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(_materialize_slide_job, job): (i, job[0])
                for i, job in enumerate(jobs, start=1)
            }
            for fut in as_completed(futures):
                slide_i, slide_id = futures[fut]
                results, w, s = fut.result()
                _consume(slide_id, results, w, s, slide_i)

    return [rec for rec in out if rec is not None]


def _materialize_slide_job(
    job: tuple,
) -> tuple[list[tuple[int, PatchPlanRecord]], int, int]:
    (
        slide_id,
        indexed_records,
        wsi_path,
        xml_path,
        split,
        patch_size,
        images_dir,
        masks_dir,
        backend,
        skip_existing,
        png_compress_level,
    ) = job
    plan = slide_plan_stub_for_materialize(
        slide_id=slide_id,
        split=split,
        wsi_path=Path(wsi_path),
        xml_path=Path(xml_path),
        patch_size=int(patch_size),
    )
    return _materialize_one_slide_group(
        slide_id=slide_id,
        indexed_records=indexed_records,
        plan=plan,
        images_dir=Path(images_dir),
        masks_dir=Path(masks_dir),
        backend=backend,
        skip_existing=bool(skip_existing),
        png_compress_level=int(png_compress_level),
    )


def load_slide_row_from_split(split_csv: Path, slide_id: str) -> dict[str, str]:
    with split_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("slide_id") == slide_id:
                return row
    raise KeyError(f"slide_id {slide_id!r} not found in {split_csv}")


def collect_split_candidate_records(
    *,
    split: str,
    slide_ids: Sequence[str],
    split_csv: Path,
    cfg: DatasetConfig,
    patch_size: int,
    backend: str,
    workers: int = 1,
) -> list[PatchPlanRecord]:
    """All grid candidates with supervised pixels, deterministic slide order."""
    workers = max(1, int(workers))
    jobs: list[tuple[str, str, str, str, int, str]] = []
    for slide_id in slide_ids:
        row = load_slide_row_from_split(split_csv, slide_id)
        wsi_path, xml_path, row_split = resolve_slide_paths(cfg, slide_id, row)
        if row_split != split:
            raise ValueError(
                f"slide {slide_id!r} has split={row_split!r} but expected {split!r} in {split_csv}"
            )
        jobs.append((slide_id, split, str(wsi_path), str(xml_path), int(patch_size), backend))

    if workers == 1:
        plans = [_build_slide_plan_job(job) for job in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        plans = []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            # Preserve deterministic slide order.
            for plan in pool.map(_build_slide_plan_job, jobs, chunksize=1):
                plans.append(plan)

    records: list[PatchPlanRecord] = []
    for plan in plans:
        records.extend(plan.records)
    return records


def _build_slide_plan_job(job: tuple[str, str, str, str, int, str]) -> SlidePatchPlan:
    slide_id, split, wsi_path, xml_path, patch_size, backend = job
    return build_slide_patch_plan(
        slide_id=slide_id,
        split=split,
        wsi_path=Path(wsi_path),
        xml_path=Path(xml_path),
        patch_size=patch_size,
        backend=backend,
    )


def resolve_slide_paths(
    cfg: DatasetConfig,
    slide_id: str,
    split_row: dict[str, str],
) -> tuple[Path, Path, str]:
    wsi_raw = split_row.get("wsi_path") or split_row.get("image") or ""
    wsi_path = Path(wsi_raw).expanduser() if wsi_raw else None
    if wsi_path is None or not wsi_path.is_file():
        raise FileNotFoundError(f"WSI not found for slide {slide_id}: {wsi_raw}")
    xml_path = xml_path_for_slide(cfg, slide_id)
    if xml_path is None or not xml_path.is_file():
        raise FileNotFoundError(f"XML not found for slide {slide_id}")
    split = split_row.get("split", "unknown")
    return wsi_path.resolve(), xml_path.resolve(), split
