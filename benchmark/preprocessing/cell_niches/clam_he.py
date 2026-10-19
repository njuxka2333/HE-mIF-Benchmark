"""CLAM H&E tissue segmentation and per-patch white/black checks.

Port of ``WholeSlideImage.segmentTissue`` / ``isInContours`` from
``HE-mIF-Benchmark/clam`` so we can run it on numpy pyramid thumbnails
without OpenSlide (Cell-niches TIFFs are libvips pyramids).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from preprocessing.wsi.util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
)
from preprocessing.wsi.wsi_utils import isBlackPatch, isWhitePatch


@dataclass
class CLAMTissue:
    contours: list[np.ndarray]
    holes: list[list[np.ndarray]]
    tissue_mask: np.ndarray
    seg_hw: tuple[int, int]
    downsample: tuple[float, float]
    binary: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), np.uint8))


def _filter_contours(contours, hierarchy, filter_params: dict) -> tuple[list, list]:
    """CLAM ``segmentTissue._filter_contours`` (area + holes)."""
    filtered = []
    all_holes = []
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    for cont_idx in hierarchy_1:
        cont = contours[cont_idx]
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        a = cv2.contourArea(cont)
        hole_areas = [cv2.contourArea(contours[int(h)]) for h in holes]
        a = a - np.array(hole_areas).sum()
        if a == 0:
            continue
        if filter_params["a_t"] < a:
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[i] for i in filtered]
    hole_contours = []
    for hole_ids in all_holes:
        unfiltered_holes = [contours[int(idx)] for idx in hole_ids]
        unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        unfiltered_holes = unfiltered_holes[: filter_params["max_n_holes"]]
        filtered_holes = [
            hole for hole in unfiltered_holes if cv2.contourArea(hole) > filter_params["a_h"]
        ]
        hole_contours.append(filtered_holes)
    return foreground_contours, hole_contours


def segment_tissue(
    rgb: np.ndarray,
    *,
    downsample: tuple[float, float],
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 4,
    use_otsu: bool = False,
    filter_params: dict | None = None,
    ref_patch_size: int = 512,
) -> CLAMTissue:
    """Run CLAM HSV-S tissue segmentation on an RGB thumbnail.

    ``downsample`` is ``(level0_w / thumb_w, level0_h / thumb_h)``. Returned
    contours are in level-0 pixel coordinates, matching CLAM.
    """
    filter_params = dict(filter_params or {"a_t": 100, "a_h": 16, "max_n_holes": 8})
    img = np.ascontiguousarray(rgb[..., :3])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    scaled_ref_patch_area = int(ref_patch_size**2 / (downsample[0] * downsample[1]))
    fp = filter_params.copy()
    fp["a_t"] = fp["a_t"] * scaled_ref_patch_area
    fp["a_h"] = fp["a_h"] * scaled_ref_patch_area

    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        h, w = img_otsu.shape[:2]
        return CLAMTissue([], [], np.zeros((h, w), np.uint8), (h, w), downsample, img_otsu)

    hierarchy = np.squeeze(hierarchy, axis=(0,))
    if hierarchy.ndim == 1:
        hierarchy = hierarchy[None, :]
    hierarchy = hierarchy[:, 2:]
    foreground_contours, hole_contours = _filter_contours(contours, hierarchy, fp)

    contours_l0 = [np.array(cont * np.array(downsample), dtype=np.int32) for cont in foreground_contours]
    holes_l0 = [
        [np.array(hole * np.array(downsample), dtype=np.int32) for hole in holes]
        for holes in hole_contours
    ]
    mask = _draw_tissue_mask(img_otsu.shape[:2], contours_l0, holes_l0, downsample)
    h, w = img_otsu.shape[:2]
    return CLAMTissue(contours_l0, holes_l0, mask, (h, w), downsample, img_otsu)


def _draw_tissue_mask(
    seg_hw: tuple[int, int],
    contours_l0: list[np.ndarray],
    holes_l0: list[list[np.ndarray]],
    downsample: tuple[float, float],
) -> np.ndarray:
    h, w = seg_hw
    mask = np.zeros((h, w), np.uint8)
    inv = np.array([1.0 / downsample[0], 1.0 / downsample[1]])
    for cont, holes in zip(contours_l0, holes_l0):
        c = np.array(cont * inv, dtype=np.int32)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
        if holes:
            hs = [np.array(hole * inv, dtype=np.int32) for hole in holes]
            cv2.drawContours(mask, hs, -1, 0, thickness=-1)
    return mask


def _contour_fn(name: str, contour: np.ndarray, patch_size: int):
    if name == "four_pt":
        return isInContourV3_Easy(contour=contour, patch_size=patch_size, center_shift=0.5)
    if name == "four_pt_hard":
        return isInContourV3_Hard(contour=contour, patch_size=patch_size, center_shift=0.5)
    if name == "center":
        return isInContourV2(contour=contour, patch_size=patch_size)
    if name == "basic":
        return isInContourV1(contour=contour)
    raise ValueError(f"unknown contour_fn {name}")


def is_in_holes(holes: list[np.ndarray], pt: tuple[int, int], patch_size: int) -> bool:
    cx = pt[0] + patch_size / 2.0
    cy = pt[1] + patch_size / 2.0
    for hole in holes:
        if cv2.pointPolygonTest(hole, (float(cx), float(cy)), False) > 0:
            return True
    return False


def patch_in_tissue_contour(
    x: int,
    y: int,
    patch_size: int,
    tissue: CLAMTissue,
    contour_fn: str = "four_pt",
) -> bool:
    """CLAM ``isInContours`` over the union of tissue contours."""
    pt = (int(x), int(y))
    for cont, holes in zip(tissue.contours, tissue.holes):
        fn = _contour_fn(contour_fn, cont, patch_size)
        if fn(pt) and not is_in_holes(holes, pt, patch_size):
            return True
    return False


def tissue_fraction(
    x: int,
    y: int,
    w: int,
    h: int,
    tissue: CLAMTissue,
) -> float:
    dsx, dsy = tissue.downsample
    x0 = int(np.floor(x / dsx))
    y0 = int(np.floor(y / dsy))
    x1 = int(np.ceil((x + w) / dsx))
    y1 = int(np.ceil((y + h) / dsy))
    mask = tissue.tissue_mask
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(mask.shape[1], x1)
    y1 = min(mask.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float(crop.mean()) / 255.0


def is_white_patch(rgb: np.ndarray, sat_thresh: int = 5) -> bool:
    return bool(isWhitePatch(np.ascontiguousarray(rgb[..., :3]), satThresh=sat_thresh))


def is_black_patch(rgb: np.ndarray, rgb_thresh: int = 40) -> bool:
    return bool(isBlackPatch(np.ascontiguousarray(rgb[..., :3]), rgbThresh=rgb_thresh))
