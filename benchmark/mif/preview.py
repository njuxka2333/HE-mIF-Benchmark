"""Shared display for virtual-staining GT vs predictions.

Models emit in different intensity spaces (tanh, occupancy, IMC counts, IF).
Independent p1–p99 stretch then fills the colormap for every panel, so a dense
occupancy map looks orange while sparse GT stays black. Previews instead:

1. resample the prediction onto the GT grid
2. copy GT's brightness histogram onto the prediction (rank-preserving)
3. colorize GT and every model with the same LUT

The spatial pattern is unchanged (Spearman vs the raw pred is 1). Only the
paint job is shared, so a reader can judge whether hotspots sit on GT cells.
"""

from __future__ import annotations

import numpy as np

from utils.resize_ops import resize_continuous

__all__ = [
    "MARKER_CMAP_STOPS",
    "colorize_marker",
    "decode_marker_rgb",
    "display_like_gt",
    "match_histogram",
    "marker_lut",
    "percentile_01",
    "preview_triplet",
]

MARKER_CMAP_STOPS = (
    "#000000",
    "#3b0f70",
    "#8c2981",
    "#de4968",
    "#fe9f6d",
    "#fcfdbf",
)
P_LO = 1.0
P_HI = 99.0


def _hex_rgb(color: str) -> tuple[float, float, float]:
    raw = color.lstrip("#")
    return (int(raw[0:2], 16) / 255.0, int(raw[2:4], 16) / 255.0, int(raw[4:6], 16) / 255.0)


def marker_lut(n: int = 256) -> np.ndarray:
    stops = np.asarray([_hex_rgb(c) for c in MARKER_CMAP_STOPS], dtype=np.float64)
    x = np.linspace(0.0, 1.0, len(stops))
    xi = np.linspace(0.0, 1.0, int(n))
    return np.stack([np.interp(xi, x, stops[:, k]) for k in range(3)], axis=1).astype(np.float32)


_LUT = marker_lut()


def percentile_01(arr: np.ndarray, lo: float = P_LO, hi: float = P_HI) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x[finite], (lo, hi))
    if not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks so ties share one value (no raster-order wash)."""
    n = int(x.size)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[sorter]
    change = np.empty(n, dtype=bool)
    change[0] = True
    if n > 1:
        change[1:] = sorted_x[1:] != sorted_x[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty(starts.size, dtype=np.intp)
    if starts.size:
        ends[:-1] = starts[1:]
        ends[-1] = n
    avg = 0.5 * (starts + 1 + ends)
    ranks[sorter] = avg[np.cumsum(change) - 1]
    return ranks


def match_histogram(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Give ``source`` the intensity histogram of ``template``. Ranks stay put.

    Tied source pixels share one output value (the template value at their
    average rank). Plain argsort assignment would paint a raster-order
    gradient across a zero background, which is a display artifact.
    """
    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(template, dtype=np.float64)
    if src.shape != ref.shape:
        raise ValueError(f"shape mismatch {src.shape} vs {ref.shape}")
    flat_s = src.ravel()
    sorted_t = np.sort(ref.ravel(), kind="mergesort")
    n = int(flat_s.size)
    if n == 0:
        return np.asarray(source, dtype=np.float32)
    if n == 1:
        return np.full(src.shape, sorted_t[0], dtype=np.float32)
    pos = _average_ranks(flat_s) - 1.0
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    t = pos - lo
    out = (1.0 - t) * sorted_t[lo] + t * sorted_t[hi]
    return out.reshape(src.shape).astype(np.float32)


def display_like_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    size_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gt01, pred_matched)`` on the same grid, both in ``[0, 1]``."""
    gt_map = np.asarray(gt, dtype=np.float32)
    pred_map = np.asarray(pred, dtype=np.float32)
    if gt_map.ndim != 2 or pred_map.ndim != 2:
        raise ValueError(f"expected 2-D maps, got pred={pred_map.shape} gt={gt_map.shape}")
    if size_wh is not None:
        width, height = int(size_wh[0]), int(size_wh[1])
        gt_map = resize_continuous(gt_map, (width, height))
        pred_map = resize_continuous(pred_map, (width, height))
    elif pred_map.shape != gt_map.shape:
        height, width = gt_map.shape
        pred_map = resize_continuous(pred_map, (width, height))
    gt01 = percentile_01(gt_map)
    return gt01, match_histogram(pred_map, gt01)


def colorize_marker(arr01: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """Colorize a ``[0, 1]`` map with the shared marker LUT."""
    table = _LUT if lut is None else np.asarray(lut, dtype=np.float32)
    n = int(table.shape[0])
    idx = np.clip(np.rint(np.asarray(arr01, dtype=np.float32) * (n - 1)), 0, n - 1).astype(np.intp)
    return np.ascontiguousarray((table[idx] * 255.0).astype(np.uint8))


def decode_marker_rgb(rgb: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """Approximate inverse of :func:`colorize_marker` (nearest LUT bin)."""
    table = _LUT if lut is None else np.asarray(lut, dtype=np.float32)
    pix = np.asarray(rgb, dtype=np.float32)
    if pix.ndim != 3 or pix.shape[-1] < 3:
        raise ValueError(f"expected HxWx3, got {pix.shape}")
    flat = pix[..., :3].reshape(-1, 3) / 255.0
    n = int(table.shape[0])
    out = np.empty(flat.shape[0], dtype=np.float32)
    step = 65536
    for start in range(0, flat.shape[0], step):
        sl = flat[start : start + step]
        dist = ((sl[:, None, :] - table[None, :, :]) ** 2).sum(-1)
        out[start : start + step] = dist.argmin(1).astype(np.float32) / float(n - 1)
    return out.reshape(pix.shape[:2])


def preview_triplet(
    he: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    side: int = 256,
) -> np.ndarray:
    """H&E | GT | pred, same marker colormap, pred histogram-matched to GT."""
    import cv2

    he_s = cv2.resize(np.asarray(he), (side, side), interpolation=cv2.INTER_AREA)
    if he_s.ndim == 2:
        he_s = np.stack([he_s] * 3, axis=-1)
    gt01, pred01 = display_like_gt(pred, gt, size_wh=(side, side))
    return np.concatenate(
        [he_s[..., :3], colorize_marker(gt01), colorize_marker(pred01)],
        axis=1,
    )
