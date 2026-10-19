"""Statistics for the patient-level clinical tasks (prognosis).

Companion to :mod:`metrics.classification`, which covers the
cell-level tasks (DeLong's test, case-level bootstrap). This module holds what
the survival endpoint needs:

* **Harrell's C-index**, implemented directly so the package carries no
  scikit-survival dependency.
* **Fold-level confidence intervals** using a t-distribution over the K
  cross-validation folds. This follows the Shazam reference implementation
  (``shazam_survival_analysis.py``), which uses ``t.ppf(0.975, df=K-1)``. For
  K=5 that is 2.776, not the normal 1.96 - a normal approximation is ~42% too
  narrow at this fold count.
* **Patient-level bootstrap** over pooled cross-validated predictions
  (PathBench-style), as a second, independent uncertainty source.
* **Last-epoch extraction** from a run's ``history.csv``. When a task was run
  with ``valid_split == test``, the reported max-over-epochs metric is
  selection-biased; the final-epoch value is not, provided no epoch was selected
  by the training loop. :func:`fold_metric_summary` returns both plus the
  inflation between them.

Caveat carried from the audit: pooling risk scores across folds assumes the
per-fold models produce comparable scales. Where fold-to-fold risk scale drifts
(GigaTIME's risk median varies ~3x more across folds than UNI2's), the pooled
bootstrap is distorted and the fold-level t-CI should be preferred.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "CHANCE_LEVEL",
    "harrell_c_index",
    "t_critical",
    "fold_t_interval",
    "fold_metric_summary",
    "patient_bootstrap_interval",
    "verdict_against_chance",
    "cohort_of",
    "site_identity_auroc",
    "within_cohort_auroc",
]

CHANCE_LEVEL = 0.5


# --------------------------------------------------------------------- metrics
def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index. Higher risk must mean shorter survival.

    ``event`` is 1 for an observed event, 0 for censored.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event).astype(int)
    risk = np.asarray(risk, dtype=np.float64)
    order = np.argsort(time)
    time, event, risk = time[order], event[order], risk[order]

    concordant = permissible = 0.0
    for i in range(time.size):
        if event[i] != 1:
            continue
        later = time > time[i]
        if not later.any():
            continue
        other = risk[later]
        permissible += other.size
        concordant += np.sum(risk[i] > other) + 0.5 * np.sum(risk[i] == other)
    return concordant / permissible if permissible else float("nan")


# ------------------------------------------------------- fold-level intervals
def t_critical(n_folds: int, ci: float = 0.95) -> float:
    """Two-sided t critical value with df = n_folds - 1.

    Hard-coded for the common fold counts so the package needs no scipy; falls
    back to the normal approximation above df=30, where the difference is <2%.
    """
    # keyed by DEGREES OF FREEDOM (df = n_folds - 1)
    table = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
             6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
             14: 2.144787, 19: 2.093024, 29: 2.045230}
    if ci != 0.95:
        raise ValueError("only the 95% interval is tabulated")
    df = max(1, int(n_folds) - 1)
    if df in table:
        return table[df]
    if df > 30:
        return 1.959964
    nearest = min(table, key=lambda k: abs(k - df))
    return table[nearest]


def fold_t_interval(values: Sequence[float], ci: float = 0.95) -> dict:
    """Mean and t-distribution CI across cross-validation folds (Shazam method)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = arr.size
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ci_lo": nan, "ci_hi": nan, "n_folds": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "std": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_folds": 1}
    std = float(arr.std(ddof=1))
    half = t_critical(n, ci) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half,
            "n_folds": n, "t_crit": t_critical(n, ci)}


def fold_metric_summary(history_paths: Sequence[Path], metric_column: str) -> dict:
    """Best-epoch vs last-epoch metric across folds, and the inflation between.

    ``best`` is the max-over-epochs value (what a run selecting on this split
    would report); ``last`` is the final-epoch value, which is unbiased provided
    the training loop selected no epoch. ``inflation = best - last``.
    """
    import pandas as pd

    best, last = [], []
    for path in history_paths:
        frame = pd.read_csv(path)
        if metric_column not in frame.columns:
            continue
        series = frame[metric_column].dropna()
        if series.empty:
            continue
        best.append(float(series.max()))
        last.append(float(series.iloc[-1]))
    return {
        "best": fold_t_interval(best),
        "last": fold_t_interval(last),
        "inflation": (float(np.mean(best)) - float(np.mean(last))) if best and last else float("nan"),
    }


# --------------------------------------------------- patient-level bootstrap
def patient_bootstrap_interval(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    min_events: int = 2,
    event_index: int | None = None,
) -> dict:
    """Percentile bootstrap over patients (resampled with replacement).

    ``metric_fn(*arrays[idx])`` is recomputed on each resample. When
    ``event_index`` is given, resamples containing fewer than ``min_events``
    events are skipped, which keeps a degenerate C-index out of the distribution.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    point = metric_fn(*arrays)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event_index is not None and int(np.sum(arrays[event_index][idx])) < min_events:
            continue
        value = metric_fn(*[a[idx] for a in arrays])
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        nan = float("nan")
        return {"point": point, "ci_lo": nan, "ci_hi": nan, "n_boot": 0}
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": len(samples), "ci": ci}


def verdict_against_chance(ci_lo: float, chance: float = CHANCE_LEVEL) -> str:
    if not np.isfinite(ci_lo):
        return "undetermined"
    return "above chance" if ci_lo > chance else "not distinguishable from chance"

# ------------------------------------------------- multi-cohort confounding
DEFAULT_COHORT = "other"


def cohort_of(case_id: str, patterns: dict | None = None, default: str = DEFAULT_COHORT) -> str:
    """Source cohort for a patient id, by id shape."""
    import re
    for name, pattern in (patterns or {}).items():
        if re.match(pattern, str(case_id)):
            return name
    return default


def site_identity_auroc(labels, cohorts) -> float:
    """AUROC of a score that knows only which cohort a patient came from.

    Pooling cohorts whose outcome base rates differ lets a model score above
    chance by recognising the institution and predicting its prevalence. This
    is the value to beat: any model at or below it has not been shown to read
    anything beyond provenance.
    """
    labels = np.asarray(labels, dtype=float)
    rates: dict = {}
    for c in set(cohorts):
        mask = np.array([x == c for x in cohorts])
        rates[c] = float(labels[mask].mean())
    return _rank_auroc(labels, np.array([rates[c] for c in cohorts], dtype=float))


def within_cohort_auroc(labels, scores, cohorts) -> dict:
    """AUROC computed inside each cohort, and the pair-weighted pooled average.

    ``gap = pooled - within`` is the part of the pooled figure attributable to
    between-cohort separation rather than to discrimination among comparable
    patients.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    per, num, den = {}, 0.0, 0.0
    for c in sorted(set(cohorts)):
        mask = np.array([x == c for x in cohorts])
        y, s = labels[mask], scores[mask]
        n1, n0 = float(y.sum()), float((1 - y).sum())
        value = _rank_auroc(y, s) if (n1 and n0) else float("nan")
        per[c] = {"auroc": value, "n": int(mask.sum()), "positives": int(n1)}
        if n1 and n0 and np.isfinite(value):
            num += value * n1 * n0
            den += n1 * n0
    pooled = _rank_auroc(labels, scores)
    within = num / den if den else float("nan")
    return {"pooled": pooled, "within": within, "gap": pooled - within, "per_cohort": per}


def _rank_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = y.size - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
