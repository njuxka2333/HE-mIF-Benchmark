"""Tests for the shared Cell-niches vstain protocol (numpy only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics.vstain import (  # noqa: E402
    CACHE_BIN_UM,
    CORE_MARKERS,
    HEADLINE_METRIC_KEYS,
    PROTOCOL_ID,
    REPORT_BIN_UM,
    SharedVStainAccumulator,
    average_precision,
    describe_geometry,
    expected_he_px,
    mean_pool_grid,
    otsu_threshold,
    percentile_01,
    ranking_block,
    spearman_r,
)


def test_expected_he_px_matches_operating_points():
    assert expected_he_px(256.0, 0.5) == 512
    assert expected_he_px(256.0, 0.25) == 1024


def test_gigatime_generator_is_whole_512_at_0p5():
    geo = describe_geometry(
        model="gigatime", model_mpp=0.5, input_size=256, feed="whole"
    )
    assert geo["he_px_at_model_mpp"] == 512
    assert geo["fed_tile_px"] == 512
    assert abs(geo["fed_tile_um"] - 256.0) < 1e-6


def test_rosie_generator_is_whole_1024_at_0p25():
    geo = describe_geometry(
        model="rosie", model_mpp=0.25, input_size=128, feed="whole"
    )
    assert geo["he_px_at_model_mpp"] == 1024
    assert geo["fed_tile_px"] == 1024
    assert abs(geo["fed_tile_um"] - 256.0) < 1e-6


def test_hemit_whole_window_is_256um():
    geo = describe_geometry(
        model="hemit", model_mpp=0.25, input_size=1024, feed="whole"
    )
    assert geo["he_px_at_model_mpp"] == 1024
    assert geo["fed_tile_px"] == 1024
    assert abs(geo["fed_tile_um"] - 256.0) < 1e-6
    assert "pred resampled to GT grid" in geo["notes"]


def test_spearman_is_monotonic_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=256)
    y = rng.normal(size=256)
    a = spearman_r(x, y)
    b = spearman_r(np.exp(x), y)
    assert abs(a - b) < 1e-10


def test_spearman_perfect_rank():
    x = np.arange(100.0)
    assert abs(spearman_r(x, x * 3 + 2) - 1.0) < 1e-12
    assert abs(spearman_r(x, -x) + 1.0) < 1e-12


def test_average_ranks_matches_python_loop_with_ties():
    from metrics.vstain import _average_ranks

    def python_loop(x):
        n = int(x.size)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and x[order[j]] == x[order[i]]:
                j += 1
            avg = 0.5 * ((i + 1) + j)
            ranks[order[i:j]] = avg
            i = j
        return ranks

    rng = np.random.default_rng(3)
    x = rng.integers(0, 12, size=4096).astype(np.float64)
    x[:8] = 7.0
    got = _average_ranks(x)
    expected = python_loop(x)
    assert np.allclose(got, expected)


def test_percentile_01_independent_affine():
    rng = np.random.default_rng(1)
    x = rng.random((32, 32)) * 50 + 10
    y = percentile_01(x)
    assert 0.0 <= float(np.nanmin(y)) <= float(np.nanmax(y)) <= 1.0


def test_accumulator_and_core4_nan_not_zero():
    acc = SharedVStainAccumulator(["CD3", "PanCK"], cache_bin_um=0)
    rng = np.random.default_rng(2)
    gt = rng.random((16, 16))
    pred = 0.8 * gt + 0.2 * rng.random((16, 16))
    acc.update("CD3", pred, gt)
    acc.update("PanCK", pred, gt)
    per = acc.per_marker()
    rank = ranking_block(per)
    assert rank["core4_markers_present"] == ["CD3", "PanCK"]
    assert "CD8" not in rank["core4_markers_present"]
    assert np.isfinite(rank["cd3_spearman"])
    assert np.isfinite(rank["core4_spearman_mean"])
    # missing CD8/CD20 must not be filled with 0
    assert set(CORE_MARKERS) - set(rank["core4_markers_present"]) == {"CD8", "CD20"}


# ---------------------------------------------------------------------------
# per-marker SSIM / PSNR (appendix) and overall-vs-CD3 aggregation
# ---------------------------------------------------------------------------
from metrics.vstain import DATA_RANGE  # noqa: E402


def _acc_with(markers, seed=0, shape=(24, 24)):
    acc = SharedVStainAccumulator(markers, cache_bin_um=0)
    rng = np.random.default_rng(seed)
    for i, m in enumerate(markers):
        gt = rng.random(shape)
        pred = (0.9 - 0.1 * i) * gt + (0.1 + 0.1 * i) * rng.random(shape)
        acc.update(m, pred, gt)
    return acc


def test_per_marker_emits_all_four_metrics():
    per = _acc_with(["CD3", "PanCK"]).per_marker()
    for m in ("CD3", "PanCK"):
        for field in ("spearman_r_mean_patch", "pearson_r_calibrated_global",
                      "auprc_mean_patch", "ssim_mean_patch", "psnr_global_db"):
            assert field in per[m], f"{m} missing {field}"
            assert np.isfinite(per[m][field]), f"{m}/{field} not finite"


def test_ssim_and_psnr_are_perfect_on_identical_input():
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=0)
    rng = np.random.default_rng(3)
    gt = rng.random((24, 24))
    acc.update("CD3", gt, gt)
    per = acc.per_marker()["CD3"]
    assert per["ssim_mean_patch"] > 0.999
    assert np.isinf(per["psnr_global_db"])          # MSE == 0
    assert abs(per["pearson_r_calibrated_global"] - 1.0) < 1e-9


def test_ssim_psnr_are_scale_invariant_via_normalisation():
    """Same structure, different output range -> same SSIM/PSNR."""
    rng = np.random.default_rng(4)
    gt = rng.random((24, 24))
    pred = 0.8 * gt + 0.2 * rng.random((24, 24))
    a = SharedVStainAccumulator(["CD3"], cache_bin_um=0)
    a.update("CD3", pred, gt)
    b = SharedVStainAccumulator(["CD3"], cache_bin_um=0)
    b.update("CD3", pred * 30.0 - 1.0, gt)          # HistoPlexer-like range
    pa, pb = a.per_marker()["CD3"], b.per_marker()["CD3"]
    assert abs(pa["ssim_mean_patch"] - pb["ssim_mean_patch"]) < 1e-6
    assert abs(pa["psnr_global_db"] - pb["psnr_global_db"]) < 1e-6


def test_ranking_block_exposes_cd3_for_every_metric():
    rank = ranking_block(_acc_with(["CD3", "CD8", "PanCK"]).per_marker())
    assert rank["headline_metric"] == "cd3_spearman"
    assert rank["headline_detection"] == "cd3_auprc"
    assert rank["ssim_psnr_in_ranking"] is False
    assert rank["data_range"] == DATA_RANGE
    for key in HEADLINE_METRIC_KEYS:
        assert np.isfinite(rank[f"cd3_{key}"]), key
    assert "cd3_ssim" not in rank
    assert "cd3_psnr" not in rank


def test_overall_is_mean_over_this_models_markers_only():
    per = _acc_with(["CD3", "CD8", "PanCK"]).per_marker()
    rank = ranking_block(per)
    assert rank["overall_markers"] == ["CD3", "CD8", "PanCK"]
    assert rank["n_overlap_markers"] == 3
    assert rank["overall_comparable_across_models"] is False
    expected = np.mean([per[m]["auprc_mean_patch"] for m in per])
    assert abs(rank["overall_auprc_mean"] - expected) < 1e-12


def test_overall_composition_differs_between_coverages():
    """A 2-marker model and a 3-marker model must not look like one scale."""
    hemit = ranking_block(_acc_with(["CD3", "PanCK"]).per_marker())
    histo = ranking_block(_acc_with(["CD3", "CD8", "CD20"]).per_marker())
    assert hemit["overall_markers"] != histo["overall_markers"]
    assert hemit["core4_comparable"] is False
    # CD3 is the marker they share, so that column is the fair one.
    assert np.isfinite(hemit["cd3_spearman"]) and np.isfinite(histo["cd3_spearman"])


def test_backwards_compatible_keys_still_present():
    rank = ranking_block(_acc_with(["CD3", "PanCK"]).per_marker())
    assert rank["cd3_pearson_calibrated"] == rank["cd3_pearson"]
    assert rank["overlap_spearman_mean"] == rank["overall_spearman_mean"]
    assert rank["overlap_pearson_calibrated_mean"] == rank["overall_pearson_mean"]
    assert rank["core4_pearson_calibrated_mean"] == rank["core4_pearson_mean"]


def test_nan_pixels_do_not_poison_ssim():
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=0)
    rng = np.random.default_rng(5)
    gt = rng.random((24, 24))
    pred = 0.8 * gt + 0.2 * rng.random((24, 24))
    pred[0, 0] = np.nan
    acc.update("CD3", pred, gt)
    assert np.isfinite(acc.per_marker()["CD3"]["ssim_mean_patch"])


# ---------------------------------------------------------------------------
# 4 µm cache / 8 µm report, Otsu-AUPRC
# ---------------------------------------------------------------------------

def test_protocol_id():
    assert PROTOCOL_ID == "cell_niches_shared"
    assert CACHE_BIN_UM == 4.0
    assert REPORT_BIN_UM == 8.0


def test_mean_pool_grid_averages_blocks():
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    got = mean_pool_grid(x, 2)
    assert got.shape == (2, 2)
    assert abs(float(got[0, 0]) - np.mean(x[:2, :2])) < 1e-6
    assert abs(float(got[1, 1]) - np.mean(x[2:, 2:])) < 1e-6


def test_mean_pool_grid_native_gt_size():
    """Cell-niches GT is 788 px / 256 µm; 4 µm bins are 64, 788 % 64 != 0."""
    rng = np.random.default_rng(9)
    gt = rng.random((788, 788)).astype(np.float32)
    got = mean_pool_grid(gt, 64)
    assert got.shape == (64, 64)
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=4.0, report_bin_um=8.0, patch_um=256.0)
    pred = (0.8 * gt + 0.2 * rng.random((788, 788))).astype(np.float32)
    acc.update("CD3", pred, gt)
    p4, t4 = acc.pooled_4um["CD3"][0]
    assert p4.shape == (64, 64)
    assert t4.shape == (64, 64)
    assert np.isfinite(acc.per_marker()["CD3"]["spearman_r_mean_patch"])


def test_otsu_splits_two_well_separated_blobs():
    x = np.concatenate([np.zeros(200), np.ones(200)])
    thr = otsu_threshold(x)
    assert 0.2 < thr < 0.8


def test_average_precision_matches_sklearn_example():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    # sklearn.metrics.average_precision_score on this pair is 0.8333...
    assert abs(average_precision(y_true, y_score) - 5.0 / 6.0) < 1e-12


def test_identical_maps_give_perfect_auprc_after_otsu():
    rng = np.random.default_rng(6)
    gt = rng.random((32, 32))
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=0)
    acc.update("CD3", gt, gt)
    assert abs(acc.per_marker()["CD3"]["auprc_mean_patch"] - 1.0) < 1e-12


def test_pool_64px_is_identity_at_4um_then_2x2_at_8um():
    """64×64 on a 256 µm patch is already 4 µm/px, so the cache is a copy."""
    rng = np.random.default_rng(7)
    gt = rng.random((64, 64)).astype(np.float32)
    pred = (0.8 * gt + 0.2 * rng.random((64, 64))).astype(np.float32)
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=4.0, report_bin_um=8.0, patch_um=256.0)
    acc.update("CD3", pred, gt)
    p4, t4 = acc.pooled_4um["CD3"][0]
    assert p4.shape == (64, 64)
    np.testing.assert_allclose(p4, pred, atol=1e-5)
    from metrics.vstain import mean_pool_grid, spearman_r

    sp8 = acc.spearman_patches["CD3"][0]
    assert abs(sp8 - spearman_r(mean_pool_grid(pred, 32), mean_pool_grid(gt, 32))) < 1e-6


def test_ranking_has_positive_frac_and_omits_thresholded_prf():
    acc = _acc_with(["CD3", "PanCK"])
    per = acc.per_marker()
    assert "precision_at_0p5_mean_patch" not in per["CD3"]
    assert "recall_at_0p5_mean_patch" not in per["CD3"]
    assert "f1_at_0p5_mean_patch" not in per["CD3"]
    assert 0.0 <= per["CD3"]["positive_frac_mean_patch"] <= 1.0
    rank = ranking_block(per)
    assert 0.0 <= rank["cd3_positive_frac"] <= 1.0
    assert "cd3_positive_frac_global_pixels" in rank


def test_ranking_exposes_report_bin():
    rng = np.random.default_rng(8)
    acc = SharedVStainAccumulator(["CD3"], cache_bin_um=4.0, report_bin_um=8.0, patch_um=256.0)
    gt = rng.random((64, 64))
    acc.update("CD3", 0.7 * gt + 0.3 * rng.random((64, 64)), gt)
    rank = ranking_block(acc.per_marker())
    assert rank["report_bin_um"] == 8.0
    assert rank["cache_bin_um"] == 4.0
    assert np.isfinite(rank["cd3_spearman"])
    assert np.isfinite(rank["cd3_auprc"])
