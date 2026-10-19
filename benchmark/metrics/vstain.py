"""Virtual-staining metrics.

One module for Cell-niches scoring:

* Table-4 pixel metrics (Pearson / PSNR / SSIM in MIPHEI IF space)
* shared ``cell_niches_shared`` protocol (Spearman / Otsu-AUPRC at 8 µm)
* CD3 FOV-bootstrap and shuffled-pairing null
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tifffile
from numpy.lib.stride_tricks import sliding_window_view

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from scipy.ndimage import correlate1d as _scipy_correlate1d
except ImportError:  # pragma: no cover
    _scipy_correlate1d = None

IF_LO = -0.9
IF_HI = 0.9
IF_SPAN = IF_HI - IF_LO  # 1.8
ORION_P99 = 0.99
UINT16_BINS = 65536

__all__ = [
    "IF_HI",
    "IF_LO",
    "IF_SPAN",
    "ORION_P99",
    "collect_fov_p99",
    "histogram_percentile",
    "mean_over_markers",
    "pred_to_if",
    "raw_to_if_orion",
    "tile_coords",
    "tile_pearson",
    "uint8_to_if",
    "update_u16_hist",
]


def uint8_to_if(x: np.ndarray) -> np.ndarray:
    """Official MIPHEI ``NormalizationLayer(mode='if')``."""
    return np.float32(x) / 255.0 * IF_SPAN + IF_LO


def pred_to_if(pred: np.ndarray) -> np.ndarray:
    """Official eval clip before PSNR / SSIM / Pearson."""
    return np.clip(np.asarray(pred, dtype=np.float32), IF_LO, IF_HI)


def raw_to_if_orion(gt: np.ndarray, p99: float) -> np.ndarray:
    """Orion-like uint8 encoding, then the official IF map, without AF subtraction."""
    scale = max(float(p99), 1.0)
    unit = np.clip(np.log1p(np.asarray(gt, dtype=np.float32) / scale), 0.0, 1.0)
    return uint8_to_if(np.uint8(np.round(unit * 255.0)))


def update_u16_hist(hist: np.ndarray, arr: np.ndarray) -> None:
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return
    clipped = np.clip(flat.astype(np.int64, copy=False), 0, UINT16_BINS - 1)
    hist += np.bincount(clipped, minlength=UINT16_BINS)


def histogram_percentile(hist: np.ndarray, q: float = ORION_P99) -> float:
    """Percentile from a histogram whose bins are intensity 0, 1, 2, ...

    ``q`` is a fraction in ``[0, 1]``, matching MIPHEI's ``percentile=0.99`` call.
    """
    total = float(np.asarray(hist, dtype=np.float64).sum())
    if total <= 0.0:
        return 1.0
    cdf = np.cumsum(hist, dtype=np.float64) / total
    return float(np.searchsorted(cdf, q, side="right"))


def collect_fov_p99(
    patches_dir: Path,
    records: list[dict],
    channel_indices: list[int],
    n_channels: int,
) -> dict[str, dict[int, float]]:
    """Per-FOV 99th percentiles on native uint16 GT, Orion histogram convention."""
    hists: dict[str, dict[int, np.ndarray]] = {}
    for rec in records:
        fov = str(rec["fov_id"])
        gt = tifffile.imread(patches_dir / rec["mif_relpath"])
        if gt.ndim != 3:
            raise RuntimeError(f"GT {rec['mif_relpath']} shape {gt.shape}")
        slot = hists.setdefault(fov, {})
        for ch in channel_indices:
            hist = slot.setdefault(ch, np.zeros(UINT16_BINS, dtype=np.int64))
            update_u16_hist(hist, gt[ch])
        del gt
    out: dict[str, dict[int, float]] = {}
    for fov, per_ch in hists.items():
        out[fov] = {ch: histogram_percentile(hist, ORION_P99) for ch, hist in per_ch.items()}
        for ch in range(n_channels):
            out[fov].setdefault(ch, 1.0)
    return out


def mean_over_markers(per_marker: Mapping[str, Mapping[str, float]], key: str) -> float:
    vals = [float(v[key]) for v in per_marker.values() if key in v]
    return float(np.nanmean(vals)) if vals else float("nan")


def tile_coords(phys_px: int, tile_px: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for y in range(0, phys_px, tile_px):
        for x in range(0, phys_px, tile_px):
            if y + tile_px <= phys_px and x + tile_px <= phys_px:
                coords.append((y, x))
    return coords


def tile_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson r on one tile (MIPHEI mean-of-tiles column, not Table-4 pooled r)."""
    x = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(true, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


__all__ = [
    "StreamingPearson",
    "StreamingPSNR",
    "StreamingSSIM",
    "MarkerAccumulator",
    "VStainMetrics",
    "ssim_tile",
    "normalized_mean_rank",
]


# --------------------------------------------------------------------------
# Pearson
# --------------------------------------------------------------------------
@dataclass
class StreamingPearson:
    """Pearson r from running sums; exact, not an online approximation."""

    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        x = np.asarray(pred, dtype=np.float64).ravel()
        y = np.asarray(true, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.all():
            x, y = x[finite], y[finite]
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())

    @property
    def value(self) -> float:
        if self.n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        denom = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
        if denom <= 0.0:
            # A constant channel (common for a marker a model cannot express)
            # has no correlation defined; report nan rather than 0.
            return float("nan")
        return float(cov / denom)


# --------------------------------------------------------------------------
# PSNR
# --------------------------------------------------------------------------
@dataclass
class StreamingPSNR:
    data_range: float = 1.0
    sse: float = 0.0
    n: float = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        x = np.asarray(pred, dtype=np.float64)
        y = np.asarray(true, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
        d = (x - y).ravel()
        finite = np.isfinite(d)
        d = d[finite]
        self.sse += float((d * d).sum())
        self.n += d.size

    @property
    def value(self) -> float:
        if self.n <= 0:
            return float("nan")
        mse = self.sse / self.n
        if mse <= 0.0:
            return float("inf")
        return float(10.0 * np.log10((self.data_range ** 2) / mse))


# --------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _sep_filter(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable convolution, numpy ``reflect`` / OpenCV ``BORDER_REFLECT_101``.

    Same padding as the original per-tap Python loop. OpenCV/scipy run the
    11-tap filter in C so a GigaTIME 1024² map is tens of milliseconds
    instead of seconds.
    """
    img = np.ascontiguousarray(img, dtype=np.float64)
    kf = np.ascontiguousarray(k, dtype=np.float64)
    if cv2 is not None:
        return cv2.sepFilter2D(
            img, cv2.CV_64F, kf, kf, borderType=cv2.BORDER_REFLECT_101
        )
    if _scipy_correlate1d is not None:
        rows = _scipy_correlate1d(img, kf, axis=0, mode="mirror")
        return _scipy_correlate1d(rows, kf, axis=1, mode="mirror")
    pad = len(kf) // 2
    x = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
    rows = np.tensordot(sliding_window_view(x, len(kf), axis=0), kf, axes=([2], [0]))
    x = np.pad(rows, ((0, 0), (pad, pad)), mode="reflect")
    return np.tensordot(sliding_window_view(x, len(kf), axis=1), kf, axes=([2], [0]))


def ssim_tile(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    data_range: float = 1.0,
    size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean SSIM of one 2-D tile (11x11 Gaussian window, Wang et al. defaults)."""
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(true, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"expected matching 2-D tiles, got {x.shape} vs {y.shape}")
    k = _gaussian_kernel(int(size), float(sigma))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = _sep_filter(x, k)
    mu_y = _sep_filter(y, k)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sxx = _sep_filter(x * x, k) - mu_xx
    syy = _sep_filter(y * y, k) - mu_yy
    sxy = _sep_filter(x * y, k) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sxy + c2)
    den = (mu_xx + mu_yy + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


@dataclass
class StreamingSSIM:
    data_range: float = 1.0
    total: float = 0.0
    count: int = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.total += ssim_tile(pred, true, data_range=self.data_range)
        self.count += 1

    @property
    def value(self) -> float:
        return float(self.total / self.count) if self.count else float("nan")


# --------------------------------------------------------------------------
# Bundles
# --------------------------------------------------------------------------
@dataclass
class MarkerAccumulator:
    marker: str
    data_range: float = 1.0
    pearson: StreamingPearson = field(default_factory=StreamingPearson)
    psnr: StreamingPSNR = field(init=False)
    ssim: StreamingSSIM = field(init=False)

    def __post_init__(self) -> None:
        self.psnr = StreamingPSNR(data_range=self.data_range)
        self.ssim = StreamingSSIM(data_range=self.data_range)

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.pearson.update(pred, true)
        self.psnr.update(pred, true)
        if np.ndim(pred) == 2:
            self.ssim.update(pred, true)

    def result(self) -> dict[str, float]:
        return {
            "pearson_r": self.pearson.value,
            "psnr": self.psnr.value,
            "ssim": self.ssim.value,
            "n_pixels": float(self.pearson.n),
            "n_tiles": float(self.ssim.count),
        }


class VStainMetrics:
    """Per-marker accumulators for one (model, cohort) run."""

    def __init__(self, markers: Sequence[str], *, data_range: float = 1.0) -> None:
        self.data_range = float(data_range)
        self.markers = list(markers)
        self._acc = {m: MarkerAccumulator(m, data_range=self.data_range) for m in self.markers}

    def update(self, marker: str, pred: np.ndarray, true: np.ndarray) -> None:
        if marker not in self._acc:
            raise KeyError(f"unknown marker {marker!r}; declared: {self.markers}")
        self._acc[marker].update(pred, true)

    def results(self) -> dict[str, dict[str, float]]:
        return {m: a.result() for m, a in self._acc.items()}


# --------------------------------------------------------------------------
# Cross-model aggregation
# --------------------------------------------------------------------------
def normalized_mean_rank(
    scores: Mapping[str, Mapping[str, float]],
    *,
    higher_is_better: bool = True,
) -> dict[str, dict[str, float]]:
    """Difficulty-corrected summary across a ragged marker x model table.

    ``scores[model][marker] = value``; missing markers are simply absent (never 0).
    Within each marker the competing models are ranked, converted to a percentile
    in ``[0, 1]`` where 0 is best, then averaged over the markers each model
    covers. Ranking inside a marker is fair by construction because every model
    there faced the same target, so no baseline anchors are needed.

    Returns ``{model: {"mean_rank": float, "coverage": int}}``.
    """
    markers: set[str] = set()
    for per_model in scores.values():
        markers.update(per_model)

    per_model_ranks: dict[str, list[float]] = {m: [] for m in scores}
    for marker in sorted(markers):
        present = [
            (model, float(vals[marker]))
            for model, vals in scores.items()
            if marker in vals and np.isfinite(vals[marker])
        ]
        if len(present) < 2:
            continue  # a single competitor carries no rank information
        present.sort(key=lambda kv: kv[1], reverse=higher_is_better)
        n = len(present)
        for position, (model, _v) in enumerate(present):
            per_model_ranks[model].append(position / (n - 1))

    return {
        model: {
            "mean_rank": float(np.mean(r)) if r else float("nan"),
            "coverage": len(r),
        }
        for model, r in per_model_ranks.items()
    }


PROTOCOL_ID = "cell_niches_shared"
PATCH_UM = 256.0
CACHE_BIN_UM = 4.0
REPORT_BIN_UM = 8.0
P_LO = 1.0
P_HI = 99.0
CORE_MARKERS = ("CD3", "CD8", "CD20", "PanCK")
#: Every scale-dependent metric here is computed on p1-p99 -> [0, 1] maps.
DATA_RANGE = 1.0
#: Cross-model ranking columns (SSIM/PSNR stay in per_marker as appendix).
HEADLINE_METRIC_KEYS = ("spearman", "pearson", "auprc")
APPENDIX_METRIC_KEYS = ("ssim", "psnr")
METRIC_KEYS = HEADLINE_METRIC_KEYS + APPENDIX_METRIC_KEYS

__all__ = [
    "APPENDIX_METRIC_KEYS",
    "CACHE_BIN_UM",
    "CORE_MARKERS",
    "DATA_RANGE",
    "HEADLINE_METRIC_KEYS",
    "METRIC_KEYS",
    "PATCH_UM",
    "PROTOCOL_ID",
    "REPORT_BIN_UM",
    "SharedVStainAccumulator",
    "average_precision",
    "describe_geometry",
    "expected_he_px",
    "mean_pool_grid",
    "n_bins_for",
    "nanmean",
    "otsu_threshold",
    "patch_auprc",
    "percentile_01",
    "pool_to_um",
    "ranking_block",
    "spearman_r",
    "write_per_patch_spearman",
    "write_pooled_4um_cache",
]


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def expected_he_px(patch_um: float, model_mpp: float) -> int:
    return max(1, int(round(float(patch_um) / float(model_mpp))))


def n_bins_for(patch_um: float, bin_um: float) -> int:
    n = int(round(float(patch_um) / float(bin_um)))
    if n < 1:
        raise ValueError(f"bin_um={bin_um} is larger than patch_um={patch_um}")
    return n


def mean_pool_grid(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-pool a 2-D map into an ``n_bins × n_bins`` grid.

    Exact block averages when ``H``/``W`` divide ``n_bins``. Native Cell-niches
    GT is 788 px for a 256 µm window, which is not a multiple of the 4 µm grid
    (64 bins), so those maps are area-resized onto the bin grid.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"mean_pool_grid expects 2-D, got {a.shape}")
    n_bins = int(n_bins)
    h, w = a.shape
    if n_bins < 1:
        raise ValueError(f"cannot mean-pool {h}x{w} into {n_bins} bins")
    if h == n_bins and w == n_bins:
        return a.astype(np.float32)
    if h % n_bins == 0 and w % n_bins == 0:
        bh, bw = h // n_bins, w // n_bins
        return a.reshape(n_bins, bh, n_bins, bw).mean(axis=(1, 3)).astype(np.float32)
    if cv2 is None:
        raise ValueError(f"cannot mean-pool {h}x{w} into {n_bins} bins")
    interp = cv2.INTER_AREA if (h > n_bins or w > n_bins) else cv2.INTER_CUBIC
    out = cv2.resize(
        np.ascontiguousarray(a, dtype=np.float32),
        (n_bins, n_bins),
        interpolation=interp,
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def pool_to_um(arr: np.ndarray, patch_um: float, bin_um: float) -> np.ndarray:
    return mean_pool_grid(arr, n_bins_for(patch_um, bin_um))


def otsu_threshold(values: np.ndarray) -> float:
    """Histogram Otsu on finite values. Returns the bin centre of the split."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(x[0]) if x.size else 0.0
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(x, bins=256, range=(lo, hi))
    total = int(hist.sum())
    if total == 0:
        return lo
    centers = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist).astype(np.float64)
    w2 = float(total) - w1
    s1 = np.cumsum(hist * centers)
    m1 = s1 / np.maximum(w1, 1.0)
    m2 = (s1[-1] - s1) / np.maximum(w2, 1.0)
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[:-1]) ** 2
    valid = (w1[:-1] > 0) & (w2[:-1] > 0)
    var = np.where(valid, var, -np.inf)
    best = float(np.max(var))
    if not np.isfinite(best):
        return 0.5 * (lo + hi)
    # Plateau (two-level 0/1 maps) → take the middle split, not the first bin.
    ks = np.flatnonzero(np.isclose(var, best))
    k = int(ks[len(ks) // 2])
    return float(centers[k])


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """sklearn-compatible binary average precision (no sklearn import)."""
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    finite = np.isfinite(ys) & np.isfinite(yt)
    yt = yt[finite].astype(np.int8)
    ys = ys[finite]
    if yt.size == 0:
        return float("nan")
    n_pos = int(yt.sum())
    if n_pos == 0 or n_pos == yt.size:
        return float("nan")
    order = np.argsort(ys, kind="mergesort")[::-1]
    yt = yt[order]
    ys = ys[order]
    distinct = np.where(np.diff(ys))[0]
    threshold_idxs = np.r_[distinct, yt.size - 1]
    tps = np.cumsum(yt, dtype=np.float64)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / float(n_pos)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall) * precision[1:]))


def patch_auprc(pred: np.ndarray, true: np.ndarray) -> float:
    """Per-patch AUPRC: Otsu on this patch's GT, then AP of pred vs that mask."""
    t = np.asarray(true)
    p = np.asarray(pred)
    y = (t.ravel() > otsu_threshold(t)).astype(np.int8)
    return average_precision(y, p.ravel())


def describe_geometry(
    *,
    model: str,
    model_mpp: float,
    input_size: int,
    feed: str,
    patch_um: float = PATCH_UM,
    pred_hw: tuple[int, int] | None = None,
    gt_hw: tuple[int, int] | None = None,
    source_he_px: int | None = None,
    source_he_mpp: float | None = None,
) -> dict:
    he_px = expected_he_px(patch_um, model_mpp)
    tile_px = int(input_size) if feed == "mosaic" else he_px
    tile_um = float(tile_px) * float(model_mpp)
    out = {
        "model": model,
        "patch_um": float(patch_um),
        "model_mpp": float(model_mpp),
        "he_px_at_model_mpp": he_px,
        "feed": feed,
        "network_tile_px": int(input_size),
        "fed_tile_px": int(tile_px),
        "fed_tile_um": tile_um,
        "source_he_px": source_he_px,
        "source_he_mpp": source_he_mpp,
        "pred_hw": list(pred_hw) if pred_hw is not None else None,
        "gt_hw": list(gt_hw) if gt_hw is not None else None,
        "cache_bin_um": CACHE_BIN_UM,
        "report_bin_um": REPORT_BIN_UM,
        "notes": (
            f"256 µm HE is resampled to {he_px} px at {model_mpp} mpp "
            f"({feed} {tile_px} px = {tile_um:.3f} µm), "
            f"pred resampled to GT grid (AREA down / CUBIC up); "
            f"score after {CACHE_BIN_UM:g} µm mean-pool cache, "
            f"headline {REPORT_BIN_UM:g} µm."
        ),
    }
    return out


def percentile_01(arr: np.ndarray, lo: float = P_LO, hi: float = P_HI) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x[finite], (lo, hi))
    if not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    y[~finite] = np.nan
    return y.astype(np.float32)


def _pearson(pred: np.ndarray, true: np.ndarray) -> float:
    x = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(true, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson of average ranks; ties share a rank."""
    x = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(true, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    return _pearson(_average_ranks(x), _average_ranks(y))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks; ties share a rank.

    Vectorized: the previous Python ``while`` over each pixel was the GigaTIME
    1024² bottleneck (~seconds per marker), not SSIM.
    """
    n = int(x.size)
    sorter = np.argsort(x, kind="mergesort")
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
    # 1-based inclusive average rank of each tie group.
    avg = 0.5 * (starts + 1 + ends)
    ranks_sorted = avg[np.cumsum(change) - 1]
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = ranks_sorted
    return ranks


@dataclass
class SharedVStainAccumulator:
    """Point-estimate accumulator on the pooled 4 µm / 8 µm grid.

    Default path mean-pools each pred/GT pair to ``cache_bin_um`` (4 µm),
    then to ``report_bin_um`` (8 µm). Spearman, AUPRC, Pearson, SSIM and PSNR
    all run on the 8 µm maps. The 4 µm arrays are always kept so they can be
    written to disk.

    ``cache_patches=True`` additionally keeps the 8 µm (pred, true) pair per
    cached marker plus the patch's ``group`` (FOV id) for the significance
    layer. Pass ``cache_markers`` to keep only those channels.

    Set ``cache_bin_um <= 0`` to score native pixels (unit tests).
    """

    markers: Sequence[str]
    cache_patches: bool = False
    cache_markers: Sequence[str] | None = None
    data_range: float = DATA_RANGE
    cache_bin_um: float = CACHE_BIN_UM
    report_bin_um: float = REPORT_BIN_UM
    patch_um: float = PATCH_UM
    pearson: dict[str, StreamingPearson] = field(init=False)
    psnr: dict[str, StreamingPSNR] = field(init=False)
    ssim: dict[str, StreamingSSIM] = field(init=False)
    spearman_patches: dict[str, list[float]] = field(init=False)
    spearman_4um_patches: dict[str, list[float]] = field(init=False)
    auprc_patches: dict[str, list[float]] = field(init=False)
    pos_frac_patches: dict[str, list[float]] = field(init=False)
    n_pos_pixels: dict[str, int] = field(init=False)
    n_pixels: dict[str, int] = field(init=False)
    pooled_4um: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(init=False)
    raw_patches: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(init=False)
    patch_groups: dict[str, list] = field(init=False)

    def __post_init__(self) -> None:
        self.pearson = {m: StreamingPearson() for m in self.markers}
        self.psnr = {m: StreamingPSNR(data_range=self.data_range) for m in self.markers}
        self.ssim = {m: StreamingSSIM(data_range=self.data_range) for m in self.markers}
        self.spearman_patches = {m: [] for m in self.markers}
        self.spearman_4um_patches = {m: [] for m in self.markers}
        self.auprc_patches = {m: [] for m in self.markers}
        self.pos_frac_patches = {m: [] for m in self.markers}
        self.n_pos_pixels = {m: 0 for m in self.markers}
        self.n_pixels = {m: 0 for m in self.markers}
        self.pooled_4um = {m: [] for m in self.markers}
        if self.cache_patches:
            cached = (
                list(self.markers)
                if self.cache_markers is None
                else [m for m in self.markers if m in set(self.cache_markers)]
            )
            self.raw_patches = {m: [] for m in cached}
            self.patch_groups = {m: [] for m in cached}
        else:
            self.raw_patches = {}
            self.patch_groups = {}

    def _reduce(self, pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return (pred_report, true_report, pred_4um_or_None, true_4um_or_None)."""
        p = np.asarray(pred, dtype=np.float32)
        t = np.asarray(true, dtype=np.float32)
        if self.cache_bin_um <= 0:
            return p, t, None, None
        p4 = pool_to_um(p, self.patch_um, self.cache_bin_um)
        t4 = pool_to_um(t, self.patch_um, self.cache_bin_um)
        if self.report_bin_um <= self.cache_bin_um:
            return p4, t4, p4, t4
        p8 = pool_to_um(p4, self.patch_um, self.report_bin_um)
        t8 = pool_to_um(t4, self.patch_um, self.report_bin_um)
        return p8, t8, p4, t4

    def update(self, marker: str, pred: np.ndarray, true: np.ndarray, group=None) -> None:
        p = np.asarray(pred)
        t = np.asarray(true)
        if p.shape != t.shape:
            raise ValueError(f"{marker}: pred {p.shape} vs gt {t.shape}")
        p_rep, t_rep, p4, t4 = self._reduce(p, t)
        if p4 is not None:
            self.pooled_4um[marker].append((np.asarray(p4, dtype=np.float32), np.asarray(t4, dtype=np.float32)))
            self.spearman_4um_patches[marker].append(spearman_r(p4, t4))
        else:
            self.spearman_4um_patches[marker].append(float("nan"))

        self.spearman_patches[marker].append(spearman_r(p_rep, t_rep))
        self.auprc_patches[marker].append(patch_auprc(p_rep, t_rep))
        gt_bin = (np.asarray(t_rep).ravel() > otsu_threshold(t_rep)).astype(np.int8)
        self.pos_frac_patches[marker].append(float(gt_bin.mean()) if gt_bin.size else float("nan"))
        self.n_pos_pixels[marker] += int(gt_bin.sum())

        pn = percentile_01(p_rep)
        tn = percentile_01(t_rep)
        self.pearson[marker].update(pn, tn)
        self.psnr[marker].update(pn, tn)
        if pn.ndim == 2:
            self.ssim[marker].update(
                np.nan_to_num(pn, nan=0.0), np.nan_to_num(tn, nan=0.0)
            )
        self.n_pixels[marker] += int(np.isfinite(p_rep).sum())

        if marker in self.raw_patches:
            self.raw_patches[marker].append(
                (np.asarray(p_rep, dtype=np.float32), np.asarray(t_rep, dtype=np.float32))
            )
            self.patch_groups[marker].append(group)

    def per_marker(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in self.markers:
            vals = np.asarray(self.spearman_patches[name], dtype=np.float64)
            ap = np.asarray(self.auprc_patches[name], dtype=np.float64)
            s4 = np.asarray(self.spearman_4um_patches[name], dtype=np.float64)
            pos = np.asarray(self.pos_frac_patches[name], dtype=np.float64)
            n_pix = int(self.n_pixels[name])
            out[name] = {
                "spearman_r_mean_patch": float(np.nanmean(vals)) if vals.size else float("nan"),
                "spearman_r_std_patch": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else float("nan"),
                "spearman_r_mean_patch_4um": float(np.nanmean(s4)) if np.isfinite(s4).any() else float("nan"),
                "auprc_mean_patch": float(np.nanmean(ap)) if ap.size else float("nan"),
                "auprc_std_patch": float(np.nanstd(ap, ddof=1)) if ap.size > 1 else float("nan"),
                "positive_frac_mean_patch": float(np.nanmean(pos)) if pos.size else float("nan"),
                "positive_frac_global_pixels": (
                    float(self.n_pos_pixels[name]) / n_pix if n_pix else float("nan")
                ),
                "pearson_r_calibrated_global": self.pearson[name].value,
                "ssim_mean_patch": self.ssim[name].value,
                "psnr_global_db": self.psnr[name].value,
                "n_patches": int(vals.size),
                "n_pixels": int(self.n_pixels[name]),
                "cache_bin_um": float(self.cache_bin_um) if self.cache_bin_um > 0 else float("nan"),
                "report_bin_um": float(self.report_bin_um) if self.cache_bin_um > 0 else float("nan"),
            }
        return out


#: per_marker() field name for each short metric key.
_FIELD = {
    "spearman": "spearman_r_mean_patch",
    "pearson": "pearson_r_calibrated_global",
    "auprc": "auprc_mean_patch",
    "ssim": "ssim_mean_patch",
    "psnr": "psnr_global_db",
}


def ranking_block(per_marker: Mapping[str, Mapping[str, float]]) -> dict:
    """Headline columns.

    ``cd3_spearman`` and ``cd3_auprc`` are the cross-model comparison (CD3 is
    the only marker every generator emits). SSIM/PSNR stay out of this block.
    ``overall_*`` averages the markers this model actually covers.
    ``core4_*`` is kept for backwards compatibility and flagged as not
    comparable across models.
    """
    vals = {
        key: {m: float(v[_FIELD[key]]) for m, v in per_marker.items()}
        for key in HEADLINE_METRIC_KEYS
    }
    present = sorted(per_marker)
    core_present = [m for m in CORE_MARKERS if m in per_marker]
    first = next(iter(per_marker.values()), {})

    out: dict = {
        "headline_metric": "cd3_spearman",
        "headline_detection": "cd3_auprc",
        "data_range": DATA_RANGE,
        "cache_bin_um": first.get("cache_bin_um", CACHE_BIN_UM),
        "report_bin_um": first.get("report_bin_um", REPORT_BIN_UM),
        "normalisation": (
            "mean-pool to 4 µm cache then 8 µm report; Spearman on pooled ranks; "
            "AUPRC vs per-patch Otsu on pooled GT; chance = mean Otsu-positive "
            "fraction; Pearson on p1-p99 of the 8 µm maps. Do not report P/R/F1@0.5."
        ),
        "ssim_psnr_in_ranking": False,
    }
    cd3 = per_marker.get("CD3") or {}
    out["cd3_positive_frac"] = float(cd3.get("positive_frac_mean_patch", float("nan")))
    out["cd3_positive_frac_global_pixels"] = float(
        cd3.get("positive_frac_global_pixels", float("nan"))
    )
    for key in HEADLINE_METRIC_KEYS:
        out[f"cd3_{key}"] = vals[key].get("CD3", float("nan"))
    for key in HEADLINE_METRIC_KEYS:
        out[f"overall_{key}_mean"] = nanmean(list(vals[key].values()))
    out["overall_markers"] = present
    out["n_overlap_markers"] = len(present)
    out["overall_comparable_across_models"] = False

    out["cd3_pearson_calibrated"] = out["cd3_pearson"]
    out["overlap_spearman_mean"] = out["overall_spearman_mean"]
    out["overlap_pearson_calibrated_mean"] = out["overall_pearson_mean"]

    out["core4_markers_present"] = core_present
    for key in HEADLINE_METRIC_KEYS:
        out[f"core4_{key}_mean"] = nanmean([vals[key][m] for m in core_present])
    out["core4_pearson_calibrated_mean"] = out["core4_pearson_mean"]
    out["core4_comparable"] = False
    out["core4_note"] = (
        "core-4 averages a different marker subset per model "
        "(e.g. HEMIT {CD3, PanCK} vs HistoPlexer {CD3, CD8, CD20}); "
        "use cd3_spearman and cd3_auprc to compare models."
    )
    return out


def write_per_patch_spearman(
    path: Path,
    patch_ids: Sequence[str],
    spearman_patches: Mapping[str, Sequence[float]],
    auprc_patches: Mapping[str, Sequence[float]] | None = None,
    spearman_4um_patches: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """Write one row per scored patch (completion order, not csv order)."""
    import pandas as pd

    n = len(patch_ids)
    rows = []
    for i, pid in enumerate(patch_ids):
        row = {"patch_id": str(pid)}
        for marker, vals in spearman_patches.items():
            row[f"{marker}_spearman"] = float(vals[i]) if i < len(vals) else float("nan")
        if auprc_patches:
            for marker, vals in auprc_patches.items():
                row[f"{marker}_auprc"] = float(vals[i]) if i < len(vals) else float("nan")
        if spearman_4um_patches:
            for marker, vals in spearman_4um_patches.items():
                row[f"{marker}_spearman_4um"] = float(vals[i]) if i < len(vals) else float("nan")
        rows.append(row)
    extras = [spearman_patches]
    if auprc_patches:
        extras.append(auprc_patches)
    if spearman_4um_patches:
        extras.append(spearman_4um_patches)
    if n and any(len(v) != n for extra in extras for v in extra.values()):
        raise RuntimeError(
            f"per-patch metric length mismatch: n_ids={n} "
            + ", ".join(f"{m}={len(v)}" for m, v in spearman_patches.items())
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_pooled_4um_cache(
    path: Path,
    patch_ids: Sequence[str],
    pooled_4um: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
    groups: Sequence | None = None,
) -> None:
    """Save the 4 µm (pred, gt) stacks so 8 µm can be re-derived later."""
    payload: dict[str, np.ndarray] = {
        "patch_ids": np.asarray(list(patch_ids), dtype=object),
        "protocol": np.asarray(PROTOCOL_ID),
        "cache_bin_um": np.asarray(CACHE_BIN_UM),
        "report_bin_um": np.asarray(REPORT_BIN_UM),
        "patch_um": np.asarray(PATCH_UM),
    }
    if groups is not None:
        payload["groups"] = np.asarray(list(groups), dtype=object)
    n = len(patch_ids)
    for marker, pairs in pooled_4um.items():
        if not pairs:
            continue
        if len(pairs) != n:
            raise RuntimeError(f"{marker} pooled cache length {len(pairs)} != {n} patch_ids")
        pred = np.stack([np.asarray(p, dtype=np.float32) for p, _t in pairs])
        true = np.stack([np.asarray(t, dtype=np.float32) for _p, t in pairs])
        payload[f"{marker}_pred"] = pred
        payload[f"{marker}_gt"] = true
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def filter_kept_manifest(manifest, patch_ids_path: Path | None, limit: int = 0):
    """Restrict a kept-patches frame to an optional id list, then optional head."""
    import pandas as pd

    out = manifest
    if patch_ids_path is not None:
        path = Path(patch_ids_path)
        if path.suffix.lower() == ".csv":
            wanted = set(pd.read_csv(path)["patch_id"].astype(str))
        else:
            wanted = {
                ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            }
        out = out.loc[out["patch_id"].astype(str).isin(wanted)].reset_index(drop=True)
        missing = wanted - set(out["patch_id"].astype(str))
        if missing:
            sample = ", ".join(sorted(missing)[:8])
            raise SystemExit(f"{len(missing)} patch-ids not in kept manifest, e.g. {sample}")
    if limit and limit > 0:
        out = out.head(int(limit)).reset_index(drop=True)
    return out


__all__ = [
    "DEFAULT_N_BOOT",
    "DEFAULT_N_PERM",
    "SIGNIFICANCE_MARKERS",
    "bootstrap_ci_from_patches",
    "compute_marker_significance",
    "patch_pearson",
    "permutation_floor",
    "shuffle_null_from_patches",
    "significance_block",
    "significance_ranking_block",
]

DEFAULT_N_BOOT = 2000
DEFAULT_N_PERM = 2000
SIGNIFICANCE_MARKERS = ("CD3",)
_CI = 0.95


def patch_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson r on independently p1-p99-normalised (pred, true); matches the
    "calibrated" Pearson already reported by ``SharedVStainAccumulator``."""
    x = np.asarray(percentile_01(pred), dtype=np.float64).ravel()
    y = np.asarray(percentile_01(true), dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _ci_bounds(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def _row_pearson(
    pred_c: np.ndarray,
    pred_ss: np.ndarray,
    true_c: np.ndarray,
    true_ss: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Pearson of each row ``pred[i]`` against ``true[idx[i]]``."""
    dots = np.einsum("ij,ij->i", pred_c, true_c[idx])
    denom = np.sqrt(pred_ss * true_ss[idx])
    out = np.full(pred_c.shape[0], np.nan, dtype=np.float64)
    ok = denom > 0.0
    out[ok] = dots[ok] / denom[ok]
    return out


def _center_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = rows - rows.mean(axis=1, keepdims=True)
    ss = np.einsum("ij,ij->i", centered, centered)
    return centered, ss


@dataclass
class _PairCache:
    """Ranks + p1-p99 maps, already centered, so a shuffle is just a gather."""

    sp_pred_c: np.ndarray
    sp_pred_ss: np.ndarray
    sp_true_c: np.ndarray
    sp_true_ss: np.ndarray
    pe_pred_c: np.ndarray
    pe_pred_ss: np.ndarray
    pe_true_c: np.ndarray
    pe_true_ss: np.ndarray

    def aligned_stats(self) -> tuple[np.ndarray, np.ndarray]:
        ident = np.arange(self.sp_pred_c.shape[0])
        return (
            _row_pearson(self.sp_pred_c, self.sp_pred_ss, self.sp_true_c, self.sp_true_ss, ident),
            _row_pearson(self.pe_pred_c, self.pe_pred_ss, self.pe_true_c, self.pe_true_ss, ident),
        )

    def perm_means(self, perm: np.ndarray) -> tuple[float, float]:
        sp = _row_pearson(self.sp_pred_c, self.sp_pred_ss, self.sp_true_c, self.sp_true_ss, perm)
        pe = _row_pearson(self.pe_pred_c, self.pe_pred_ss, self.pe_true_c, self.pe_true_ss, perm)
        return float(np.nanmean(sp)), float(np.nanmean(pe))


def _build_pair_cache(patches: Sequence[tuple[np.ndarray, np.ndarray]]) -> _PairCache | None:
    """Precompute per-patch ranks and p1-p99 maps.

    Only the all-finite, equal-shape case is vectorized. Mixed NaNs would
    change the joint finite mask on every shuffled pair, which forces a
    re-rank; Cell-niches generator outputs are finite, so the slow path is
    the exception.
    """
    if not patches:
        return None
    pred_rows = []
    true_rows = []
    pe_pred_rows = []
    pe_true_rows = []
    n_pix = None
    for pred, true in patches:
        p = np.asarray(pred)
        t = np.asarray(true)
        if p.shape != t.shape:
            return None
        pr = np.asarray(p, dtype=np.float64).ravel()
        tr = np.asarray(t, dtype=np.float64).ravel()
        if n_pix is None:
            n_pix = int(pr.size)
        if pr.size != n_pix or tr.size != n_pix:
            return None
        if not (np.isfinite(pr).all() and np.isfinite(tr).all()):
            return None
        pred_rows.append(pr)
        true_rows.append(tr)
        pe_pred_rows.append(np.asarray(percentile_01(p), dtype=np.float64).ravel())
        pe_true_rows.append(np.asarray(percentile_01(t), dtype=np.float64).ravel())
    sp_pred = np.stack([_average_ranks(row) for row in pred_rows])
    sp_true = np.stack([_average_ranks(row) for row in true_rows])
    pe_pred = np.stack(pe_pred_rows)
    pe_true = np.stack(pe_true_rows)
    if not (np.isfinite(pe_pred).all() and np.isfinite(pe_true).all()):
        return None
    sp_pred_c, sp_pred_ss = _center_rows(sp_pred)
    sp_true_c, sp_true_ss = _center_rows(sp_true)
    pe_pred_c, pe_pred_ss = _center_rows(pe_pred)
    pe_true_c, pe_true_ss = _center_rows(pe_true)
    return _PairCache(
        sp_pred_c, sp_pred_ss, sp_true_c, sp_true_ss,
        pe_pred_c, pe_pred_ss, pe_true_c, pe_true_ss,
    )


def _patch_stats(
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    cache: _PairCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is None:
        cache = _build_pair_cache(patches)
    if cache is not None:
        return cache.aligned_stats()
    sp = np.array([spearman_r(p, t) for p, t in patches], dtype=np.float64)
    pe = np.array([patch_pearson(p, t) for p, t in patches], dtype=np.float64)
    return sp, pe


def _resample_indices(n: int, groups, n_boot: int, rng) -> tuple[str, int | None, list]:
    """Draw ``n_boot`` bootstrap index sets. Resamples FOVs when ``groups`` is
    given, patches otherwise. Returns (unit, n_groups, list_of_index_arrays)."""
    if groups is None:
        return "patch", None, list(rng.integers(0, n, size=(n_boot, n)))
    groups = np.asarray(groups)
    if groups.shape[0] != n:
        raise ValueError(f"groups has {groups.shape[0]} entries for {n} patches")
    unique = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in unique}
    draws = [
        np.concatenate([rows[g] for g in rng.choice(unique, size=unique.size, replace=True)])
        for _ in range(n_boot)
    ]
    return "fov", int(unique.size), draws


def bootstrap_ci_from_patches(
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = _CI,
    seed: int = 0,
    pair_cache: _PairCache | None = None,
) -> dict:
    """Percentile bootstrap CI for this marker's mean Spearman and calibrated
    Pearson. Pairing (pred[i] with true[i]) is preserved - only which
    observations are included/repeated varies across resamples.

    ``groups`` is the per-patch FOV id. When given, whole FOVs are resampled
    (cluster bootstrap); when omitted, patches are, which understates the
    interval on this cohort. See the module docstring."""
    n = len(patches)
    nan_blk = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    if n < 2:
        return {
            "spearman": dict(nan_blk),
            "pearson_calibrated": dict(nan_blk),
            "auprc": dict(nan_blk),
            "n_patches": n,
            "n_boot": n_boot,
            "resample_unit": "fov" if groups is not None else "patch",
            "n_fovs": len(set(groups)) if groups is not None else None,
            "spearman_ci_used_for_ranking": False,
        }
    rng = np.random.default_rng(seed)
    sp_vals, pe_vals = _patch_stats(patches, cache=pair_cache)
    ap_vals = np.array([patch_auprc(p, t) for p, t in patches], dtype=np.float64)
    unit, n_groups, draws = _resample_indices(n, groups, n_boot, rng)
    sp_boot = np.array([np.nanmean(sp_vals[d]) for d in draws], dtype=np.float64)
    pe_boot = np.array([np.nanmean(pe_vals[d]) for d in draws], dtype=np.float64)
    ap_boot = np.array([np.nanmean(ap_vals[d]) for d in draws], dtype=np.float64)
    sp_lo, sp_hi = _ci_bounds(sp_boot, ci)
    pe_lo, pe_hi = _ci_bounds(pe_boot, ci)
    ap_lo, ap_hi = _ci_bounds(ap_boot, ci)
    return {
        "spearman": {"point": float(np.nanmean(sp_vals)), "ci_lo": sp_lo, "ci_hi": sp_hi},
        "pearson_calibrated": {"point": float(np.nanmean(pe_vals)), "ci_lo": pe_lo, "ci_hi": pe_hi},
        "auprc": {"point": float(np.nanmean(ap_vals)), "ci_lo": ap_lo, "ci_hi": ap_hi},
        "n_patches": n,
        "n_boot": n_boot,
        "resample_unit": unit,
        "n_fovs": n_groups,
        "ci": ci,
        "spearman_ci_used_for_ranking": False,
    }


def permutation_floor(n_perm: int) -> float:
    """Smallest one-sided p from ``(count + 1) / (n_perm + 1)``."""
    return 1.0 / (int(n_perm) + 1)


@dataclass
class _APIndex:
    """Per-patch pred sort order so a shuffle AUPRC is just a gather of labels."""

    orders: np.ndarray
    threshold_idxs: list[np.ndarray]

    @classmethod
    def build(cls, pred_rows: np.ndarray) -> "_APIndex":
        orders = np.argsort(-pred_rows, axis=1, kind="mergesort")
        idxs: list[np.ndarray] = []
        for i, order in enumerate(orders):
            ys = pred_rows[i, order]
            distinct = np.where(np.diff(ys))[0]
            idxs.append(np.r_[distinct, ys.size - 1])
        return cls(orders=orders, threshold_idxs=idxs)

    def mean_ap(self, y_rows: np.ndarray) -> float:
        n = int(y_rows.shape[0])
        aps = np.empty(n, dtype=np.float64)
        for i in range(n):
            yt = y_rows[i, self.orders[i]]
            n_pos = int(yt.sum())
            if n_pos == 0 or n_pos == yt.size:
                aps[i] = np.nan
                continue
            thr = self.threshold_idxs[i]
            tps = np.cumsum(yt, dtype=np.float64)[thr]
            fps = (1 + thr) - tps
            precision = tps / np.maximum(tps + fps, 1e-12)
            recall = tps / float(n_pos)
            precision = np.r_[1.0, precision]
            recall = np.r_[0.0, recall]
            aps[i] = float(np.sum(np.diff(recall) * precision[1:]))
        return float(np.nanmean(aps))


def _otsu_rows(patches: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_rows, otsu-binarised GT rows), both ``(n, n_pix)``."""
    pred_rows = np.stack([np.asarray(p, dtype=np.float64).ravel() for p, _t in patches])
    y_rows = np.stack(
        [
            (np.asarray(t).ravel() > otsu_threshold(t)).astype(np.int8)
            for _p, t in patches
        ]
    )
    return pred_rows, y_rows


def _derange(rng: np.random.Generator, n: int) -> np.ndarray:
    ar = np.arange(n)
    perm = rng.permutation(n)
    tries = 0
    while n > 1 and np.any(perm == ar) and tries < 20:
        perm = rng.permutation(n)
        tries += 1
    return perm


def _null_samples(
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_perm: int,
    seed: int,
    cache: _PairCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spearman / Pearson / AUPRC null draws. Ranks, p1-p99 and pred sorts once."""
    n = len(patches)
    sp_null = np.empty(n_perm, dtype=np.float64)
    pe_null = np.empty(n_perm, dtype=np.float64)
    ap_null = np.empty(n_perm, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if cache is None:
        cache = _build_pair_cache(patches)
    pred_rows, y_rows = _otsu_rows(patches)
    ap_index = _APIndex.build(pred_rows)
    preds = None if cache is not None else [p for p, _t in patches]
    trues = None if cache is not None else [t for _p, t in patches]
    for k in range(n_perm):
        perm = _derange(rng, n)
        if cache is not None:
            sp_null[k], pe_null[k] = cache.perm_means(perm)
        else:
            sp_null[k] = float(np.nanmean([spearman_r(preds[i], trues[perm[i]]) for i in range(n)]))
            pe_null[k] = float(np.nanmean([patch_pearson(preds[i], trues[perm[i]]) for i in range(n)]))
        ap_null[k] = ap_index.mean_ap(y_rows[perm])
        if n_perm >= 1000 and ((k + 1) % 200 == 0 or k + 1 == n_perm):
            print(f"  shuffle-null {k + 1}/{n_perm}", flush=True)
    return sp_null, pe_null, ap_null


def shuffle_null_from_patches(
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_perm: int = DEFAULT_N_PERM,
    ci: float = _CI,
    seed: int = 0,
) -> dict:
    """Empirical noise floor: derange the pred<->true patch pairing (no fixed
    points) and recompute the marker's mean metric, ``n_perm`` times."""
    n = len(patches)
    nan_blk = {"null_mean": float("nan"), "null_ci_lo": float("nan"), "null_ci_hi": float("nan")}
    if n < 2:
        return {
            "spearman": dict(nan_blk),
            "pearson_calibrated": dict(nan_blk),
            "auprc": dict(nan_blk),
            "n_patches": n,
            "n_perm": n_perm,
        }
    sp_null, pe_null, ap_null = _null_samples(patches, n_perm=n_perm, seed=seed)
    sp_lo, sp_hi = _ci_bounds(sp_null, ci)
    pe_lo, pe_hi = _ci_bounds(pe_null, ci)
    ap_lo, ap_hi = _ci_bounds(ap_null, ci)
    return {
        "spearman": {"null_mean": float(np.nanmean(sp_null)), "null_ci_lo": sp_lo, "null_ci_hi": sp_hi},
        "pearson_calibrated": {"null_mean": float(np.nanmean(pe_null)), "null_ci_lo": pe_lo, "null_ci_hi": pe_hi},
        "auprc": {"null_mean": float(np.nanmean(ap_null)), "null_ci_lo": ap_lo, "null_ci_hi": ap_hi},
        "n_patches": n,
        "n_perm": n_perm,
        "ci": ci,
    }


def _empirical_p(observed: float, null_samples: np.ndarray) -> float:
    """One-sided p: fraction of null draws at least as large as the real score."""
    if not np.isfinite(observed) or null_samples.size == 0:
        return float("nan")
    return float((np.sum(null_samples >= observed) + 1) / (null_samples.size + 1))


def significance_block(
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    ci: float = _CI,
    seed: int = 0,
) -> dict:
    """Bootstrap CI + shuffled-pairing null for one marker, plus the verdict
    that actually matters: is the real score detectable above chance."""
    n = len(patches)
    cache = _build_pair_cache(patches) if n >= 2 else None
    boot = bootstrap_ci_from_patches(
        patches, groups=groups, n_boot=n_boot, ci=ci, seed=seed, pair_cache=cache
    )
    if n < 2:
        nan = float("nan")
        nan_blk = {"null_mean": nan, "null_ci_lo": nan, "null_ci_hi": nan}
        null = {
            "spearman": dict(nan_blk),
            "pearson_calibrated": dict(nan_blk),
            "auprc": dict(nan_blk),
            "n_patches": n,
            "n_perm": n_perm,
        }
        sp_null = np.empty(0)
        pe_null = np.empty(0)
        ap_null = np.empty(0)
        pos_mean = pos_global = nan
    else:
        sp_null, pe_null, ap_null = _null_samples(
            patches, n_perm=n_perm, seed=seed + 1, cache=cache
        )
        sp_lo, sp_hi = _ci_bounds(sp_null, ci)
        pe_lo, pe_hi = _ci_bounds(pe_null, ci)
        ap_lo, ap_hi = _ci_bounds(ap_null, ci)
        null = {
            "spearman": {
                "null_mean": float(np.nanmean(sp_null)),
                "null_ci_lo": sp_lo,
                "null_ci_hi": sp_hi,
            },
            "pearson_calibrated": {
                "null_mean": float(np.nanmean(pe_null)),
                "null_ci_lo": pe_lo,
                "null_ci_hi": pe_hi,
            },
            "auprc": {
                "null_mean": float(np.nanmean(ap_null)),
                "null_ci_lo": ap_lo,
                "null_ci_hi": ap_hi,
            },
            "n_patches": n,
            "n_perm": n_perm,
            "ci": ci,
        }
        _pred_rows, y_rows = _otsu_rows(patches)
        pos_mean = float(y_rows.mean(axis=1).mean())
        pos_global = float(y_rows.mean())

    sp_point = boot["spearman"]["point"]
    pe_point = boot["pearson_calibrated"]["point"]
    ap_point = float((boot.get("auprc") or {}).get("point", float("nan")))
    sp_p = _empirical_p(sp_point, sp_null)
    pe_p = _empirical_p(pe_point, pe_null)
    ap_p = _empirical_p(ap_point, ap_null)
    # Spearman CI is not emitted. Detectable = shuffle p<0.05.
    sp_above = bool(np.isfinite(sp_p) and sp_p < 0.05)
    pe_above = bool(np.isfinite(pe_p) and pe_p < 0.05)
    ap_above = bool(np.isfinite(ap_p) and ap_p < 0.05)
    floor = permutation_floor(n_perm) if n >= 2 else float("nan")

    boot_pub = {
        "spearman": {"point": boot["spearman"]["point"]},
        "pearson_calibrated": {"point": boot["pearson_calibrated"]["point"]},
        "auprc": boot.get("auprc") or {
            "point": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        },
        "n_patches": boot.get("n_patches"),
        "n_boot": boot.get("n_boot"),
        "resample_unit": boot.get("resample_unit"),
        "n_fovs": boot.get("n_fovs"),
        "ci": boot.get("ci"),
        "auprc_ci_only": True,
    }

    return {
        "bootstrap": boot_pub,
        "null": null,
        "positive_frac": {
            "mean_patch": pos_mean,
            "global_pixels": pos_global,
            "note": "chance AUPRC for a non-informative ranker is mean_patch under per-patch Otsu",
        },
        "spearman_p": sp_p,
        "pearson_calibrated_p": pe_p,
        "auprc_p": ap_p,
        "spearman_above_null_95": sp_above,
        "pearson_calibrated_above_null_95": pe_above,
        "auprc_above_null_95": ap_above,
        "spearman_p_is_permutation_floor": bool(
            np.isfinite(sp_p) and np.isfinite(floor) and abs(sp_p - floor) < 1e-12
        ),
        "auprc_p_is_permutation_floor": bool(
            np.isfinite(ap_p) and np.isfinite(floor) and abs(ap_p - floor) < 1e-12
        ),
        "permutation_floor": floor,
        "resample_unit": boot.get("resample_unit"),
        "spearman_ci_used_for_ranking": False,
        "verdict": "detectable" if sp_above else "indistinguishable_from_floor",
        "auprc_verdict": "detectable" if ap_above else "indistinguishable_from_floor",
        "verdict_note": (
            "verdict is Spearman vs shuffle-null; auprc_verdict is AUPRC vs shuffle-null. "
            "A p at permutation_floor only says above chance, not that models differ."
        ),
    }


def compute_marker_significance(
    raw_patches: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
    patch_groups: Mapping[str, Sequence] | None = None,
    *,
    markers: Sequence[str] = SIGNIFICANCE_MARKERS,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
) -> dict:
    """Run :func:`significance_block` only for ``markers`` that were cached."""
    groups_map = patch_groups or {}
    out: dict = {}
    for name in markers:
        patches = raw_patches.get(name) or ()
        if len(patches) < 2:
            continue
        shape = tuple(int(s) for s in np.asarray(patches[0][0]).shape)
        print(
            f"significance {name} n_patches={len(patches)} shape={shape} "
            f"n_boot={n_boot} n_perm={n_perm}",
            flush=True,
        )
        out[name] = significance_block(
            patches,
            groups=groups_map.get(name),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
        )
        print(f"significance {name} verdict={out[name]['verdict']}", flush=True)
    return out


def significance_ranking_block(per_marker_significance: Mapping[str, Mapping[str, object]]) -> dict:
    """Same shape as ``shared_vstain_protocol.ranking_block`` but for the
    significance layer: which markers cleared the null, restricted to the
    fixed CD3 / core-4 / overlap columns so it lines up with the point-
    estimate ranking columns already in the sheet."""

    def _detectable(name: str) -> bool | None:
        blk = per_marker_significance.get(name)
        if blk is None:
            return None
        return bool(blk["verdict"] == "detectable")

    core_flags = [f for f in (_detectable(m) for m in CORE_MARKERS) if f is not None]
    all_flags = [f for f in (_detectable(m) for m in per_marker_significance) if f is not None]
    return {
        "cd3_detectable": _detectable("CD3"),
        "core4_n_detectable": int(sum(core_flags)),
        "core4_n_markers": len(core_flags),
        "overlap_n_detectable": int(sum(all_flags)),
        "overlap_n_markers": len(all_flags),
        "markers_tested": list(per_marker_significance),
    }
