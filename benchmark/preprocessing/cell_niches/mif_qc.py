"""mIF-side QC: Ultivue stitch lines, DAPI/CK foreground, dynamic range."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MIFQC:
    dapi_thumb: np.ndarray
    ck_thumb: np.ndarray | None
    stitch_mask: np.ndarray
    dapi_fg_mask: np.ndarray
    dapi_otsu_u8: np.ndarray
    downsample: tuple[float, float]


def to_u8_percentile(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(img.astype(np.float32), (p_lo, p_hi))
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros(img.shape[:2], np.uint8)
    x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _robust_hot_1d(profile: np.ndarray, z_thresh: float) -> np.ndarray:
    """MAD z-score of first differences; matches the Cell-niches FOV stitch scan."""
    d = np.abs(np.diff(profile.astype(np.float32)))
    hot = np.zeros(profile.shape[0], dtype=bool)
    if d.size == 0:
        return hot
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    z = (d - med) / (1.4826 * mad)
    hot[1:] = z > z_thresh
    hot[:-1] |= z > z_thresh
    return hot


def stitch_mask_2d(
    gray: np.ndarray,
    *,
    z_thresh: float = 10.0,
    dilate_px: int = 2,
) -> np.ndarray:
    """Flag Ultivue stitch lines from column/row-mean gradients.

    Use CK (not DAPI) when available: banding is much stronger on CK.
    Do not substitute mIF Otsu. Thresholds are MAD z-scores, not mean/std.
    """
    g = gray.astype(np.float32)
    col_hot = _robust_hot_1d(g.mean(axis=0), z_thresh)
    row_hot = _robust_hot_1d(g.mean(axis=1), z_thresh)
    if dilate_px > 0:
        k = np.ones(2 * dilate_px + 1, np.uint8)
        col_hot = cv2.dilate(col_hot.astype(np.uint8)[None, :], k.reshape(1, -1))[0].astype(bool)
        row_hot = cv2.dilate(row_hot.astype(np.uint8)[:, None], k.reshape(-1, 1))[:, 0].astype(bool)
    return (col_hot[None, :] | row_hot[:, None]).astype(np.uint8) * 255


def dapi_foreground_mask(dapi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u8 = to_u8_percentile(dapi)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return binary, u8


def window_stats(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[float, float, float]:
    """Return (mean_frac_of_255, p1, p99) on a thumbnail window."""
    h, w = arr.shape[:2]
    xa, xb = max(0, x0), min(w, x1)
    ya, yb = max(0, y0), min(h, y1)
    if xb <= xa or yb <= ya:
        return 0.0, 0.0, 0.0
    crop = arr[ya:yb, xa:xb]
    if crop.size == 0:
        return 0.0, 0.0, 0.0
    frac = float(crop.mean()) / 255.0 if crop.dtype == np.uint8 else float((crop > 0).mean())
    p1, p99 = np.percentile(crop.astype(np.float32), (1.0, 99.0))
    return frac, float(p1), float(p99)


def map_level0_window_to_thumb(
    x: int,
    y: int,
    w: int,
    h: int,
    downsample: tuple[float, float],
    thumb_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    dsx, dsy = downsample
    x0 = int(np.floor(x / dsx))
    y0 = int(np.floor(y / dsy))
    x1 = int(np.ceil((x + w) / dsx))
    y1 = int(np.ceil((y + h) / dsy))
    th, tw = thumb_hw
    return max(0, x0), max(0, y0), min(tw, x1), min(th, y1)


def saturation_fraction(arr: np.ndarray, sat_value: int = 65535) -> float:
    if arr.size == 0:
        return 0.0
    return float((arr >= sat_value).mean())


def build_mif_qc(
    dapi_thumb: np.ndarray,
    *,
    downsample: tuple[float, float],
    ck_thumb: np.ndarray | None = None,
    stitch_gray: np.ndarray | None = None,
    stitch_z_thresh: float = 10.0,
    stitch_dilate_px: int = 2,
) -> MIFQC:
    src = stitch_gray if stitch_gray is not None else (ck_thumb if ck_thumb is not None else dapi_thumb)
    stitch = stitch_mask_2d(src, z_thresh=stitch_z_thresh, dilate_px=stitch_dilate_px)
    fg, u8 = dapi_foreground_mask(dapi_thumb)
    return MIFQC(
        dapi_thumb=dapi_thumb,
        ck_thumb=ck_thumb,
        stitch_mask=stitch,
        dapi_fg_mask=fg,
        dapi_otsu_u8=u8,
        downsample=downsample,
    )
