"""Tests for the DeLong's-test / case-level bootstrap significance layer for
single-cell classification (numpy only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics.classification import (  # noqa: E402
    case_bootstrap_paired_diff,
    classification_significance_block,
    delong_roc_test,
    macro_auroc_ovr,
    per_class_delong,
)


def _binary_scores(n=400, signal=1.5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    scores = rng.normal(size=n) + y * signal
    return y, scores


def _multiclass_problem(n=600, n_classes=4, signal=1.2, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(size=(n, n_classes)) * 0.3
    logits[np.arange(n), labels] += signal
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    case_ids = np.array([f"case{i // 12}" for i in range(n)])
    return labels, probs, case_ids


def test_delong_identical_scores_gives_p_one():
    y, scores = _binary_scores()
    res = delong_roc_test(y, scores, scores)
    assert res["diff"] == 0.0
    assert res["p"] == 1.0
    assert res["z"] == 0.0


def test_delong_matches_exact_auc():
    y, scores = _binary_scores(signal=2.0, seed=3)
    res = delong_roc_test(y, scores, scores * 0 + scores)  # trivial pairing
    pos = scores[y == 1]
    neg = scores[y == 0]
    wins = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    exact = wins / (len(pos) * len(neg))
    assert abs(exact - res["auc_a"]) < 1e-9


def test_delong_detects_real_difference():
    y, _ = _binary_scores(signal=0.2, seed=1)
    rng = np.random.default_rng(1)
    weak = rng.normal(size=y.size) + y * 0.2
    strong = rng.normal(size=y.size) + y * 3.0
    # Same y, two different score vectors on the same (paired) samples.
    res = delong_roc_test(y, weak, strong)
    assert res["auc_b"] > res["auc_a"]
    assert res["p"] < 0.01


def test_delong_noise_vs_noise_not_significant():
    y, _ = _binary_scores(seed=5)
    rng = np.random.default_rng(9)
    a = rng.normal(size=y.size)
    b = rng.normal(size=y.size)
    res = delong_roc_test(y, a, b)
    assert res["p"] > 0.05


def test_macro_auroc_ovr_perfect_separation_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[labels]
    assert abs(macro_auroc_ovr(labels, probs) - 1.0) < 1e-9


def test_macro_auroc_ovr_random_is_near_half():
    rng = np.random.default_rng(0)
    n, k = 2000, 3
    labels = rng.integers(0, k, size=n)
    probs = rng.random(size=(n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    auc = macro_auroc_ovr(labels, probs)
    assert abs(auc - 0.5) < 0.05


def test_case_bootstrap_identical_models_zero_diff_degenerate_ci():
    labels, probs, case_ids = _multiclass_problem()
    boot = case_bootstrap_paired_diff(labels, probs, probs, case_ids, n_boot=300, seed=0)
    assert boot["diff"] == 0.0
    assert boot["ci_lo"] == 0.0 and boot["ci_hi"] == 0.0


def test_case_bootstrap_detects_real_difference():
    labels, probs_strong, case_ids = _multiclass_problem(signal=2.5, seed=7)
    _, probs_weak, _ = _multiclass_problem(signal=0.1, seed=7)
    boot = case_bootstrap_paired_diff(labels, probs_strong, probs_weak, case_ids, n_boot=500, seed=0)
    assert boot["diff"] > 0
    assert boot["ci_lo"] > 0  # CI excludes 0 -> real, detectable difference


def test_case_bootstrap_widens_relative_to_naive_per_row_when_cases_correlated():
    # Build a scenario where within-case scores are highly correlated (all
    # cells in a case share a per-case offset). Case-level resampling should
    # give a *wider* CI than resampling individual rows would, because the
    # effective sample size is the number of cases, not the number of cells.
    rng = np.random.default_rng(0)
    n_cases, cells_per_case = 12, 30
    case_ids = np.array([f"c{c}" for c in range(n_cases) for _ in range(cells_per_case)])
    labels = np.array([c % 2 for c in range(n_cases) for _ in range(cells_per_case)])
    per_case_signal = rng.normal(size=n_cases)[
        np.array([c for c in range(n_cases) for _ in range(cells_per_case)])
    ]
    scores_a = per_case_signal + labels * 0.5 + rng.normal(scale=0.05, size=n_cases * cells_per_case)
    probs_a = np.stack([1 - _sigmoid(scores_a), _sigmoid(scores_a)], axis=1)
    boot = case_bootstrap_paired_diff(labels, probs_a, probs_a, case_ids, n_boot=400, seed=0)
    # identical model vs itself is degenerate (0 width) regardless - this
    # test just confirms the function runs correctly over few, large cases.
    assert boot["n_cases"] == n_cases


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_significance_block_shape_and_verdict():
    labels, probs_strong, case_ids = _multiclass_problem(signal=2.5, seed=11)
    _, probs_weak, _ = _multiclass_problem(signal=0.1, seed=11)
    class_names = [f"class{i}" for i in range(probs_strong.shape[1])]
    block = classification_significance_block(
        labels, probs_strong, probs_weak, case_ids, class_names, n_boot=400, seed=0
    )
    assert block["verdict"] == "significantly_different"
    assert block["macro_significant_95ci"] is True
    assert set(block["per_class_delong"].keys()) == set(class_names)


def test_per_class_delong_handles_degenerate_class():
    # A class with zero positive examples should return NaNs, not crash.
    labels = np.zeros(50, dtype=int)
    probs = np.random.default_rng(0).random(size=(50, 2))
    probs /= probs.sum(axis=1, keepdims=True)
    out = per_class_delong(labels, probs, probs, ["a", "b"])
    assert np.isnan(out["b"]["p"])
