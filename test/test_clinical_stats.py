"""Tests for training.clinical_stats (numpy + stdlib only)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics.clinical import (  # noqa: E402
    fold_metric_summary,
    fold_t_interval,
    harrell_c_index,
    patient_bootstrap_interval,
    t_critical,
    verdict_against_chance,
)


def test_c_index_perfect_ranking_is_one():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # highest risk dies first
    assert abs(harrell_c_index(time, event, risk) - 1.0) < 1e-9


def test_c_index_inverted_ranking_is_zero():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(harrell_c_index(time, event, risk)) < 1e-9


def test_c_index_tied_risks_give_half():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.ones(4)
    assert abs(harrell_c_index(time, event, risk) - 0.5) < 1e-9


def test_c_index_ignores_censored_as_index_case():
    # only patient 0 has an event; censored patients cannot start a pair
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 0])
    risk = np.array([9.0, 1.0, 0.5])
    assert abs(harrell_c_index(time, event, risk) - 1.0) < 1e-9


def test_c_index_all_censored_is_nan():
    out = harrell_c_index(np.array([1.0, 2.0]), np.array([0, 0]), np.array([1.0, 2.0]))
    assert np.isnan(out)


def test_t_critical_matches_shazam_df4():
    # shazam_survival_analysis.py uses stats.t.ppf(0.975, df=4)
    assert abs(t_critical(5) - 2.776445) < 1e-5


def test_t_interval_is_wider_than_normal_approximation():
    vals = [0.60, 0.65, 0.70, 0.62, 0.68]
    out = fold_t_interval(vals)
    sd = float(np.std(vals, ddof=1))
    normal_half = 1.959964 * sd / np.sqrt(5)
    t_half = (out["ci_hi"] - out["ci_lo"]) / 2
    assert t_half > normal_half
    assert abs(t_half / normal_half - 2.776445 / 1.959964) < 1e-6


def test_t_interval_contains_mean_and_reports_folds():
    out = fold_t_interval([0.5, 0.6, 0.7])
    assert out["ci_lo"] < out["mean"] < out["ci_hi"]
    assert out["n_folds"] == 3


def test_t_interval_single_fold_has_no_interval():
    out = fold_t_interval([0.7])
    assert out["mean"] == 0.7
    assert np.isnan(out["ci_lo"])


def test_t_interval_ignores_nans():
    out = fold_t_interval([0.6, float("nan"), 0.8])
    assert out["n_folds"] == 2


def test_verdict_against_chance():
    assert verdict_against_chance(0.55) == "above chance"
    assert verdict_against_chance(0.45) == "not distinguishable from chance"
    assert verdict_against_chance(float("nan")) == "undetermined"


def test_fold_metric_summary_best_last_and_inflation():
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for fold, curve in enumerate([[0.5, 0.8, 0.6], [0.4, 0.7, 0.5]]):
            p = Path(tmp) / f"h{fold}.csv"
            pd.DataFrame({"epoch": range(len(curve)), "valid_c_index": curve}).to_csv(p, index=False)
            paths.append(p)
        out = fold_metric_summary(paths, "valid_c_index")
        assert abs(out["best"]["mean"] - 0.75) < 1e-9   # (0.8 + 0.7) / 2
        assert abs(out["last"]["mean"] - 0.55) < 1e-9   # (0.6 + 0.5) / 2
        assert abs(out["inflation"] - 0.20) < 1e-9


def test_fold_metric_summary_missing_column_is_empty():
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "h.csv"
        pd.DataFrame({"epoch": [0, 1], "other": [1.0, 2.0]}).to_csv(p, index=False)
        out = fold_metric_summary([p], "valid_c_index")
        assert out["best"]["n_folds"] == 0


def test_patient_bootstrap_recovers_a_strong_signal():
    rng = np.random.default_rng(0)
    n = 200
    risk = rng.normal(size=n)
    time = 100.0 - 10.0 * risk + rng.normal(scale=1.0, size=n)
    event = np.ones(n, dtype=int)
    out = patient_bootstrap_interval(harrell_c_index, time, event, risk,
                                     n_boot=200, seed=1, event_index=1)
    assert out["point"] > 0.9
    assert out["ci_lo"] > 0.8


def test_patient_bootstrap_random_risk_straddles_chance():
    rng = np.random.default_rng(3)
    n = 150
    time = rng.uniform(1, 100, size=n)
    event = rng.integers(0, 2, size=n)
    risk = rng.normal(size=n)
    out = patient_bootstrap_interval(harrell_c_index, time, event, risk,
                                     n_boot=300, seed=2, event_index=1)
    assert out["ci_lo"] < 0.5 < out["ci_hi"]


# --------------------------------------------------- postprocess case bootstrap
def test_case_bootstrap_is_deterministic_and_case_level():
    """The row cap must be drawn once, so the same seed gives the same interval,
    and resampling must be over cases (n_cases equals the number of groups)."""
    from training.postprocess import _case_bootstrap

    rng = np.random.default_rng(0)
    n_cases, per_case, n_class = 6, 400, 3
    case_ids = np.repeat(np.arange(n_cases), per_case)
    labels = rng.integers(0, n_class, case_ids.size)
    probs_a = rng.random((case_ids.size, n_class))
    probs_a[np.arange(case_ids.size), labels] += 1.2       # informative
    probs_b = rng.random((case_ids.size, n_class))         # chance
    probs_a /= probs_a.sum(1, keepdims=True)
    probs_b /= probs_b.sum(1, keepdims=True)

    kw = dict(n_boot=60, seed=42, max_rows=600)
    one = _case_bootstrap(labels, probs_a, probs_b, case_ids, **kw)
    two = _case_bootstrap(labels, probs_a, probs_b, case_ids, **kw)
    assert one["ci_lo"] == two["ci_lo"] and one["ci_hi"] == two["ci_hi"]
    assert one["n_cases"] == n_cases
    assert one["diff"] > 0                       # informative model wins
    assert one["ci_lo"] > 0                      # and the interval excludes zero


def test_case_bootstrap_point_estimate_uses_every_row():
    """The cap applies to resamples only; the point estimate is the full-data one."""
    from metrics.classification import macro_auroc_ovr
    from training.postprocess import _case_bootstrap

    rng = np.random.default_rng(1)
    case_ids = np.repeat(np.arange(4), 250)
    labels = rng.integers(0, 2, case_ids.size)
    probs_a = rng.random((case_ids.size, 2)); probs_a /= probs_a.sum(1, keepdims=True)
    probs_b = rng.random((case_ids.size, 2)); probs_b /= probs_b.sum(1, keepdims=True)
    out = _case_bootstrap(labels, probs_a, probs_b, case_ids, n_boot=20, seed=7, max_rows=100)
    assert abs(out["metric_a"] - macro_auroc_ovr(labels, probs_a)) < 1e-12
    assert abs(out["metric_b"] - macro_auroc_ovr(labels, probs_b)) < 1e-12
