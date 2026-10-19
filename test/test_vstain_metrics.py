"""Numeric tests for the virtual-staining metrics (numpy only, no torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics.vstain import (  # noqa: E402
    IF_LO,
    IF_SPAN,
    histogram_percentile,
    pred_to_if,
    raw_to_if_orion,
    uint8_to_if,
)
from metrics.vstain import (  # noqa: E402
    StreamingPSNR,
    StreamingPearson,
    VStainMetrics,
    normalized_mean_rank,
    ssim_tile,
)


def test_streaming_pearson_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 64, 64))
    y = 0.7 * x + 0.3 * rng.normal(size=x.shape)
    acc = StreamingPearson()
    for i in range(x.shape[0]):          # fed tile by tile
        acc.update(x[i], y[i])
    expected = np.corrcoef(x.ravel(), y.ravel())[0, 1]
    assert abs(acc.value - expected) < 1e-10, (acc.value, expected)


def test_pearson_is_affine_invariant():
    """Intensity normalisation must not move the primary metric."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 32))
    y = rng.normal(size=(32, 32))
    a, b = StreamingPearson(), StreamingPearson()
    a.update(x, y)
    b.update(3.7 * x - 2.0, 0.25 * y + 11.0)
    assert abs(a.value - b.value) < 1e-10


def test_pearson_constant_channel_is_nan_not_zero():
    acc = StreamingPearson()
    acc.update(np.zeros((16, 16)), np.random.default_rng(2).normal(size=(16, 16)))
    assert np.isnan(acc.value)


def test_streaming_pearson_ignores_non_finite():
    x = np.array([1.0, 2.0, 3.0, np.nan])
    y = np.array([2.0, 4.0, 6.0, 1.0])
    acc = StreamingPearson()
    acc.update(x, y)
    assert abs(acc.value - 1.0) < 1e-12


def test_psnr_matches_closed_form():
    rng = np.random.default_rng(3)
    t = rng.random((2, 32, 32))
    p = t + 0.1
    acc = StreamingPSNR(data_range=1.0)
    for i in range(t.shape[0]):
        acc.update(p[i], t[i])
    assert abs(acc.value - 10 * np.log10(1.0 / 0.01)) < 1e-8


def test_psnr_identical_is_inf():
    t = np.random.default_rng(4).random((8, 8))
    acc = StreamingPSNR()
    acc.update(t, t)
    assert np.isinf(acc.value)


def _sep_filter_python_loop(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    pad = len(k) // 2
    x = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
    rows = sum(k[i] * x[i : i + img.shape[0], :] for i in range(len(k)))
    x = np.pad(rows, ((0, 0), (pad, pad)), mode="reflect")
    return sum(k[i] * x[:, i : i + img.shape[1]] for i in range(len(k)))


def test_sep_filter_matches_python_loop():
    from metrics.vstain import _gaussian_kernel, _sep_filter

    rng = np.random.default_rng(11)
    img = rng.random((96, 80))
    k = _gaussian_kernel(11, 1.5)
    got = _sep_filter(img, k)
    expected = _sep_filter_python_loop(img, k)
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_ssim_identity_and_bounds():
    t = np.random.default_rng(5).random((64, 64))
    assert abs(ssim_tile(t, t) - 1.0) < 1e-8
    noisy = np.clip(t + 0.4 * np.random.default_rng(6).normal(size=t.shape), 0, 1)
    s = ssim_tile(noisy, t)
    assert -1.0 <= s < 1.0


def test_ssim_rewards_empty_prediction_on_sparse_marker():
    """The reference benchmark's warning, as an executable check.

    A near-empty prediction can beat a structured one on a sparse channel, which
    is why SSIM must not be the ranking metric.
    """
    truth = np.zeros((64, 64))
    truth[30:34, 30:34] = 1.0                      # sparse marker
    empty = np.zeros_like(truth)
    rng = np.random.default_rng(7)
    structured = np.clip(0.25 * rng.random(truth.shape), 0, 1)
    assert ssim_tile(empty, truth) > ssim_tile(structured, truth)


def test_vstain_metrics_rejects_undeclared_marker():
    m = VStainMetrics(["CD3"])
    try:
        m.update("CD20", np.zeros((4, 4)), np.zeros((4, 4)))
    except KeyError:
        return
    raise AssertionError("undeclared marker did not raise")


def test_normalized_mean_rank_ragged_coverage():
    scores = {
        "A": {"CD3": 0.9, "CD20": 0.8, "Ki67": 0.7},   # best everywhere
        "B": {"CD3": 0.5, "CD20": 0.4, "Ki67": 0.3},   # worst everywhere
        "C": {"CD3": 0.7},                             # middle, 1 marker only
    }
    out = normalized_mean_rank(scores)
    assert out["A"]["mean_rank"] == 0.0 and out["A"]["coverage"] == 3
    assert out["B"]["mean_rank"] == 1.0 and out["B"]["coverage"] == 3
    assert abs(out["C"]["mean_rank"] - 0.5) < 1e-12 and out["C"]["coverage"] == 1


def test_normalized_mean_rank_skips_single_competitor_markers():
    scores = {"A": {"CD3": 0.9, "solo": 0.1}, "B": {"CD3": 0.5}}
    out = normalized_mean_rank(scores)
    assert out["A"]["coverage"] == 1  # 'solo' carries no rank information


def test_normalized_mean_rank_ignores_nan():
    scores = {"A": {"CD3": 0.9, "CD20": float("nan")}, "B": {"CD3": 0.5, "CD20": 0.4}}
    out = normalized_mean_rank(scores)
    assert out["A"]["coverage"] == 1 and out["B"]["coverage"] == 1


def test_miphei_uint8_to_if_matches_official_affine():
    x = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    y = uint8_to_if(x)
    assert abs(y[0] - IF_LO) < 1e-6
    assert abs(y[-1] - (IF_LO + IF_SPAN)) < 1e-6
    assert abs(float(y[1]) - 0.0) < 1e-5


def test_miphei_pred_clip_matches_evaluator():
    x = np.array([-2.0, -0.9, 0.0, 0.9, 3.0], dtype=np.float32)
    y = pred_to_if(x)
    assert np.all(y >= IF_LO - 1e-6) and np.all(y <= IF_LO + IF_SPAN + 1e-6)
    assert abs(y[0] - IF_LO) < 1e-6 and abs(y[-1] - (IF_LO + IF_SPAN)) < 1e-6


def test_miphei_orion_log_maps_zero_to_if_floor():
    gt = np.zeros((8, 8), dtype=np.float32)
    y = raw_to_if_orion(gt, p99=100.0)
    assert np.allclose(y, IF_LO)


def test_histogram_percentile_matches_numpy_on_uint16():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4000, size=50_000, dtype=np.int64)
    hist = np.bincount(x, minlength=65536)
    got = histogram_percentile(hist, 0.99)
    expected = float(np.percentile(x, 99))
    assert abs(got - expected) <= 2.0
    assert 0.0 <= got <= 4000.0


def test_psnr_if_space_uses_span_1_8():
    rng = np.random.default_rng(8)
    t = rng.uniform(-0.9, 0.9, size=(2, 32, 32))
    p = t + 0.1
    acc = StreamingPSNR(data_range=IF_SPAN)
    for i in range(t.shape[0]):
        acc.update(p[i], t[i])
    assert abs(acc.value - 10 * np.log10((IF_SPAN ** 2) / 0.01)) < 1e-8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} TESTS PASSED")
