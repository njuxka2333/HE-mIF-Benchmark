"""CLAM tissue-grid coordinates for WSI bags (prognosis).

Writes ``{dataset}/patches/{slide_id}.h5`` with a ``coords`` dataset of level-0
``(x, y)`` origins. Feature extraction reads these files; they are not in git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from preprocessing.cell_niches.clam_he import patch_in_tissue_contour, segment_tissue
from preprocessing.wsi.wsi_utils import save_hdf5
from utils.paths import CONFIG_DIR

PROTOCOL_PATH = CONFIG_DIR / "preprocessing" / "wsi_patches.yaml"


def load_wsi_patch_protocol(path: Path | None = None) -> dict[str, Any]:
    target = path or PROTOCOL_PATH
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _openslide(wsi_path: Path):
    import openslide

    return openslide.OpenSlide(str(wsi_path))


def _level_downsamples(slide) -> list[tuple[float, float]]:
    dim0 = slide.level_dimensions[0]
    out: list[tuple[float, float]] = []
    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        estimated = (dim0[0] / float(dim[0]), dim0[1] / float(dim[1]))
        if estimated != (downsample, downsample):
            out.append(estimated)
        else:
            out.append((float(downsample), float(downsample)))
    return out


def contour_grid_coords(
    contour: np.ndarray,
    holes: list[np.ndarray],
    *,
    img_w: int,
    img_h: int,
    patch_size: int,
    step_size: int,
    patch_downsample: tuple[float, float],
    use_padding: bool,
    contour_fn: str,
) -> np.ndarray:
    """CLAM ``process_contour`` candidates that pass ``four_pt`` tissue checks."""
    start_x, start_y, width, height = cv2.boundingRect(contour)
    ref_w = int(patch_size * patch_downsample[0])
    ref_h = int(patch_size * patch_downsample[1])
    if use_padding:
        stop_x, stop_y = start_x + width, start_y + height
    else:
        stop_x = min(start_x + width, img_w - ref_w + 1)
        stop_y = min(start_y + height, img_h - ref_h + 1)
    step_x = int(step_size * patch_downsample[0])
    step_y = int(step_size * patch_downsample[1])
    x_range = np.arange(start_x, stop_x, step=step_x)
    y_range = np.arange(start_y, stop_y, step=step_y)
    if x_range.size == 0 or y_range.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    xs, ys = np.meshgrid(x_range, y_range, indexing="ij")
    candidates = np.stack([xs.ravel(), ys.ravel()], axis=1)
    from preprocessing.cell_niches.clam_he import CLAMTissue

    tissue = CLAMTissue([contour], [holes], np.zeros((1, 1), np.uint8), (1, 1), patch_downsample)
    kept: list[np.ndarray] = []
    for x, y in candidates:
        if patch_in_tissue_contour(int(x), int(y), ref_w, tissue, contour_fn=contour_fn):
            kept.append((int(x), int(y)))
    if not kept:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(kept, dtype=np.int64)


def coords_for_slide(wsi_path: Path, protocol: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    proto = protocol or load_wsi_patch_protocol()
    seg = dict(proto["seg"])
    filt = dict(proto["filter"])
    patch = dict(proto["patch"])
    patch_size = int(proto["patch_size"])
    step_size = int(proto["step_size"])
    patch_level = int(proto["patch_level"])
    slide = _openslide(wsi_path)
    try:
        downsamples = _level_downsamples(slide)
        seg_level = int(seg["seg_level"])
        if seg_level < 0:
            seg_level = int(slide.get_best_level_for_downsample(64))
        thumb_w, thumb_h = slide.level_dimensions[seg_level]
        if thumb_w * thumb_h > 1e8:
            raise RuntimeError(f"seg_level {seg_level} is {thumb_w}x{thumb_h}, too large")
        rgb = np.asarray(slide.read_region((0, 0), seg_level, (thumb_w, thumb_h)).convert("RGB"))
        ds = downsamples[seg_level]
        tissue = segment_tissue(
            rgb,
            downsample=ds,
            sthresh=int(seg["sthresh"]),
            sthresh_up=int(seg.get("sthresh_up", 255)),
            mthresh=int(seg["mthresh"]),
            close=int(seg["close"]),
            use_otsu=bool(seg["use_otsu"]),
            filter_params={"a_t": int(filt["a_t"]), "a_h": int(filt["a_h"]), "max_n_holes": int(filt["max_n_holes"])},
            ref_patch_size=patch_size,
        )
        img_w, img_h = slide.level_dimensions[patch_level]
        patch_ds = downsamples[patch_level]
        blocks = [
            contour_grid_coords(
                contour,
                holes,
                img_w=img_w,
                img_h=img_h,
                patch_size=patch_size,
                step_size=step_size,
                patch_downsample=patch_ds,
                use_padding=bool(patch["use_padding"]),
                contour_fn=str(patch["contour_fn"]),
            )
            for contour, holes in zip(tissue.contours, tissue.holes)
        ]
        coords = (
            np.concatenate([b for b in blocks if len(b)], axis=0)
            if any(len(b) for b in blocks)
            else np.zeros((0, 2), dtype=np.int64)
        )
        attr = {
            "patch_size": patch_size,
            "patch_level": patch_level,
            "downsample": np.array(patch_ds),
            "downsampled_level_dim": np.array(slide.level_dimensions[patch_level]),
            "level_dim": np.array(slide.level_dimensions[patch_level]),
            "name": wsi_path.stem,
            "save_path": "",
        }
        return coords, attr
    finally:
        slide.close()


def write_coords_h5(path: Path, coords: np.ndarray, attr: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.unlink()
    attr = dict(attr)
    attr["save_path"] = str(path.parent)
    save_hdf5(str(path), {"coords": np.asarray(coords, dtype=np.int64)}, {"coords": attr}, mode="w")
