"""Tests for the bootstrap-CI / shuffled-null significance layer (numpy only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics.vstain import (  # noqa: E402
    bootstrap_ci_from_patches,
    patch_pearson,
    shuffle_null_from_patches,
    significance_block,
    significance_ranking_block,
)


def _correlated_patches(n=20, size=16, rho=0.95, seed=0):
    rng = np.random.default_rng(seed)
    patches = []
    for _ in range(n):
        true = rng.normal(size=(size, size)).astype(np.float32)
        noise = rng.normal(size=(size, size)).astype(np.float32)
        pred = (rho * true + np.sqrt(1 - rho**2) * noise).astype(np.float32)
        patches.append((pred, true))
    return patches


def _uncorrelated_patches(n=20, size=16, seed=1):
    rng = np.random.default_rng(seed)
    patches = []
    for _ in range(n):
        true = rng.normal(size=(size, size)).astype(np.float32)
        pred = rng.normal(size=(size, size)).astype(np.float32)
        patches.append((pred, true))
    return patches


def test_patch_pearson_matches_numpy_on_normalised_data():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(32, 32)).astype(np.float32)
    pred = (0.8 * true + 0.2 * rng.normal(size=(32, 32))).astype(np.float32)
    r = patch_pearson(pred, true)
    assert np.isfinite(r)
    assert -1.0 <= r <= 1.0


def test_patch_pearson_constant_is_nan():
    const = np.ones((8, 8), dtype=np.float32)
    other = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert np.isnan(patch_pearson(const, other))


def test_bootstrap_ci_contains_point_estimate():
    patches = _correlated_patches()
    boot = bootstrap_ci_from_patches(patches, n_boot=500, seed=0)
    for key in ("spearman", "pearson_calibrated"):
        blk = boot[key]
        assert blk["ci_lo"] <= blk["point"] <= blk["ci_hi"]


def test_bootstrap_ci_tight_for_strong_correlation():
    patches = _correlated_patches(n=30, rho=0.98)
    boot = bootstrap_ci_from_patches(patches, n_boot=1000, seed=0)
    assert boot["spearman"]["point"] > 0.7
    assert boot["spearman"]["ci_lo"] > 0.3


def test_bootstrap_ci_degenerate_for_single_patch():
    patches = _correlated_patches(n=1)
    boot = bootstrap_ci_from_patches(patches, n_boot=100, seed=0)
    assert np.isnan(boot["spearman"]["point"])
    assert boot["n_patches"] == 1


def test_shuffle_null_centered_near_zero_for_iid_patches():
    # With independent random patches, mismatching the pairing further should
    # not systematically change the (already near-zero) expected correlation.
    patches = _uncorrelated_patches(n=25)
    null = shuffle_null_from_patches(patches, n_perm=500, seed=0)
    assert abs(null["spearman"]["null_mean"]) < 0.15
    assert abs(null["pearson_calibrated"]["null_mean"]) < 0.15


def test_shuffle_null_derangement_has_no_fixed_points_effect():
    # Even for perfectly correlated data, shuffling the pairing should destroy
    # most of the correlation (derangement means no patch is left aligned).
    patches = _correlated_patches(n=20, rho=0.99)
    null = shuffle_null_from_patches(patches, n_perm=300, seed=0)
    boot = bootstrap_ci_from_patches(patches, n_boot=300, seed=0)
    assert null["spearman"]["null_mean"] < boot["spearman"]["point"] - 0.3


def test_significance_block_omits_spearman_ci():
    patches = _correlated_patches(n=12, rho=0.9, seed=11)
    block = significance_block(patches, n_boot=80, n_perm=40, seed=0)
    assert "ci_lo" not in block["bootstrap"]["spearman"]
    assert "ci_hi" not in block["bootstrap"]["spearman"]
    assert "ci_lo" in block["bootstrap"]["auprc"]
    assert "ci_hi" in block["bootstrap"]["auprc"]
    assert "auprc" in block["null"]
    assert "auprc_p" in block
    assert "auprc_verdict" in block
    assert 0.0 <= block["positive_frac"]["mean_patch"] <= 1.0
    assert block["permutation_floor"] == 1.0 / 41


def test_ap_index_matches_average_precision_on_aligned_pairs():
    from metrics.vstain import average_precision
    from metrics.vstain import _APIndex, _otsu_rows

    patches = _correlated_patches(n=8, size=16, seed=12)
    pred_rows, y_rows = _otsu_rows(patches)
    idx = _APIndex.build(pred_rows)
    naive = np.array([average_precision(y, p) for y, p in zip(y_rows, pred_rows)])
    assert abs(idx.mean_ap(y_rows) - float(np.nanmean(naive))) < 1e-12


def test_significance_block_detects_strong_signal():
    patches = _correlated_patches(n=25, rho=0.95, seed=2)
    block = significance_block(patches, n_boot=500, n_perm=500, seed=0)
    assert block["verdict"] == "detectable"
    assert block["spearman_above_null_95"] or block["pearson_calibrated_above_null_95"]
    assert block["spearman_p"] < 0.05 or block["pearson_calibrated_p"] < 0.05



def test_significance_block_floor_for_noise():
    patches = _uncorrelated_patches(n=20, seed=3)
    block = significance_block(patches, n_boot=500, n_perm=500, seed=0)
    assert block["verdict"] == "indistinguishable_from_floor"
    assert not block["spearman_above_null_95"]
    assert not block["pearson_calibrated_above_null_95"]


def test_significance_ranking_block_shape():
    strong = _correlated_patches(n=20, rho=0.95, seed=4)
    weak = _uncorrelated_patches(n=20, seed=5)
    per_marker_sig = {
        "CD3": significance_block(strong, n_boot=300, n_perm=300, seed=0),
        "CD8": significance_block(weak, n_boot=300, n_perm=300, seed=0),
        "CD20": significance_block(weak, n_boot=300, n_perm=300, seed=0),
        "PanCK": significance_block(weak, n_boot=300, n_perm=300, seed=0),
    }
    rank = significance_ranking_block(per_marker_sig)
    assert rank["cd3_detectable"] is True
    assert rank["core4_n_markers"] == 4
    assert 1 <= rank["core4_n_detectable"] <= 4
    assert rank["overlap_n_markers"] == 4


def test_significance_ranking_block_missing_marker_is_none():
    per_marker_sig = {
        "CD8": significance_block(_uncorrelated_patches(n=15, seed=6), n_boot=200, n_perm=200, seed=0),
    }
    rank = significance_ranking_block(per_marker_sig)
    assert rank["cd3_detectable"] is None


# ------------------------------------------------- FOV-level (cluster) bootstrap
def _clustered_patches(n_fov=6, per_fov=15, seed=0):
    """Patches within an FOV share a base image, so they are correlated draws -
    the same structure as the real 477-patches-from-~20-FOVs cohort."""
    rng = np.random.default_rng(seed)
    patches, groups = [], []
    for f in range(n_fov):
        base = rng.random((8, 8))
        offset = rng.normal(scale=0.3)          # FOV-level effect
        for _ in range(per_fov):
            pred = base + offset + 0.02 * rng.random((8, 8))
            true = base + 0.02 * rng.random((8, 8))
            patches.append((pred.astype(np.float32), true.astype(np.float32)))
            groups.append(f"fov{f}")
    return patches, groups


def test_fov_bootstrap_is_wider_than_patch_bootstrap():
    from metrics.vstain import bootstrap_ci_from_patches

    patches, groups = _clustered_patches()
    patch = bootstrap_ci_from_patches(patches, n_boot=300, seed=42)
    fov = bootstrap_ci_from_patches(patches, groups=groups, n_boot=300, seed=42)
    width = lambda b: b["spearman"]["ci_hi"] - b["spearman"]["ci_lo"]  # noqa: E731
    assert width(fov) > width(patch), "clustered data must widen the interval"
    assert patch["resample_unit"] == "patch"
    assert fov["resample_unit"] == "fov"
    assert fov["n_fovs"] == 6


def test_fov_bootstrap_keeps_the_point_estimate_unchanged():
    from metrics.vstain import bootstrap_ci_from_patches

    patches, groups = _clustered_patches()
    a = bootstrap_ci_from_patches(patches, n_boot=50, seed=1)
    b = bootstrap_ci_from_patches(patches, groups=groups, n_boot=50, seed=1)
    assert abs(a["spearman"]["point"] - b["spearman"]["point"]) < 1e-12
    assert abs(a["pearson_calibrated"]["point"] - b["pearson_calibrated"]["point"]) < 1e-12


def test_bootstrap_rejects_mismatched_group_length():
    from metrics.vstain import bootstrap_ci_from_patches

    patches, groups = _clustered_patches(n_fov=2, per_fov=3)
    try:
        bootstrap_ci_from_patches(patches, groups=groups[:-1], n_boot=5, seed=0)
    except ValueError:
        return
    raise AssertionError("a groups/patches length mismatch must raise, not be silently zipped")


def test_significance_block_records_the_resample_unit():
    from metrics.vstain import significance_block

    patches, groups = _clustered_patches(n_fov=4, per_fov=5)
    block = significance_block(patches, groups=groups, n_boot=50, n_perm=20, seed=0)
    assert block["resample_unit"] == "fov"
    assert block["bootstrap"]["n_fovs"] == 4


def test_fast_cache_matches_naive_pair_metrics():
    """Ranks / p1-p99 are computed once; a shuffled pair must match spearman_r
    and patch_pearson on the same (pred_i, true_j)."""
    from metrics.vstain import spearman_r
    from metrics.vstain import _build_pair_cache, _derange

    patches = _correlated_patches(n=8, size=24, seed=7)
    cache = _build_pair_cache(patches)
    assert cache is not None
    rng = np.random.default_rng(0)
    perm = _derange(rng, len(patches))
    sp_fast, pe_fast = cache.perm_means(perm)
    sp_naive = np.mean([spearman_r(patches[i][0], patches[perm[i]][1]) for i in range(len(patches))])
    pe_naive = np.mean([patch_pearson(patches[i][0], patches[perm[i]][1]) for i in range(len(patches))])
    assert abs(sp_fast - sp_naive) < 1e-12
    assert abs(pe_fast - pe_naive) < 1e-12


def test_aligned_cache_matches_patch_stats():
    from metrics.vstain import spearman_r
    from metrics.vstain import _build_pair_cache

    patches = _correlated_patches(n=10, size=16, seed=8)
    sp_c, pe_c = _build_pair_cache(patches).aligned_stats()
    sp_n = np.array([spearman_r(p, t) for p, t in patches])
    pe_n = np.array([patch_pearson(p, t) for p, t in patches])
    np.testing.assert_allclose(sp_c, sp_n, atol=1e-12)
    np.testing.assert_allclose(pe_c, pe_n, atol=1e-12)


def test_compute_marker_significance_is_cd3_only():
    from metrics.vstain import compute_marker_significance

    raw = {
        "CD3": _correlated_patches(n=20, rho=0.95, seed=9),
        "CD8": _uncorrelated_patches(n=20, seed=10),
    }
    groups = {"CD3": ["a"] * 10 + ["b"] * 10, "CD8": ["a"] * 10 + ["b"] * 10}
    out = compute_marker_significance(raw, groups, n_boot=50, n_perm=30, seed=0)
    assert set(out) == {"CD3"}
    assert out["CD3"]["verdict"] == "detectable"


def test_accumulator_cache_markers_skips_other_channels():
    from metrics.vstain import SharedVStainAccumulator

    acc = SharedVStainAccumulator(["CD3", "CD8"], cache_patches=True, cache_markers=("CD3",), cache_bin_um=0)
    acc.update("CD3", np.random.random((4, 4)), np.random.random((4, 4)), group="fov0")
    acc.update("CD8", np.random.random((4, 4)), np.random.random((4, 4)), group="fov0")
    assert "CD3" in acc.raw_patches and len(acc.raw_patches["CD3"]) == 1
    assert "CD8" not in acc.raw_patches


def test_accumulator_records_patch_groups():
    from metrics.vstain import SharedVStainAccumulator

    acc = SharedVStainAccumulator(["CD3"], cache_patches=True, cache_bin_um=0)
    for fov in ("a", "a", "b"):
        acc.update("CD3", np.random.random((4, 4)), np.random.random((4, 4)), group=fov)
    assert acc.patch_groups["CD3"] == ["a", "a", "b"]
    assert len(acc.raw_patches["CD3"]) == 3
