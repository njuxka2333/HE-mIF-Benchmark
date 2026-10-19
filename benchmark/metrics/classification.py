"""DeLong's test + case-level bootstrap for the single-cell classification task.

Motivation
----------
Route A ("frozen encoder features") point estimates currently show MIPHEI-ViT
nearly tying UNI2 on Immunocto (0.862 vs 0.861 AUROC) while UNI2 clearly leads
on NuCLS (0.782 vs 0.761). Before treating either gap as a real finding, we
need a number that says whether it's distinguishable from sampling noise on
this specific test set - especially given Immunocto is built from the same
Orion CRC data MIPHEI-ViT was trained on, so a "tie" there needs real
statistical backing, not just two point estimates.

Two complementary tools, both computed from data every run already has
(``test_result/predictions.csv``: per-cell ``label`` + full
``probability_class_*`` vector + ``case_id``) - no retraining required:

* **DeLong's test** (Sun & Xu, 2014 fast algorithm) - a closed-form,
  non-bootstrap test comparing two *correlated* ROC curves (same samples,
  two models' scores). Applied one-vs-rest per class, since DeLong's test is
  natively binary. This is the standard method in clinical-ML papers for
  "is model A significantly different from model B on this test set"
  (e.g. Neidlinger et al., Nat. Biomed. Eng. 2025).
* **Case-level paired bootstrap** on the macro-AUROC (or any other metric)
  difference between two models. Resampling is done at the *case* (slide/
  patient) level, not the cell level, because cells from the same slide are
  correlated - a naive per-cell bootstrap would understate the true
  variance. This also gives a CI for macro-AUROC, which has no closed-form
  multiclass DeLong generalisation.

Neither of these requires re-training anything: both operate purely on the
cached (label, probabilities, case_id) columns already written by
``training/test.py`` / the underlying ``evaluate()`` pass for every
already-trained checkpoint.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_N_BOOT",
    "delong_roc_variance",
    "delong_roc_test",
    "per_class_delong",
    "case_bootstrap_paired_diff",
    "macro_auroc_ovr",
    "classification_significance_block",
]

DEFAULT_N_BOOT = 2000
_CI = 0.95


def _binary_auc_and_cov(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """Fast DeLong: per-sample placement values + the resulting AUC.

    Returns (auc, positive_placements, negative_placements) where the
    placement arrays are what DeLong's covariance formula is built from.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y_true]
    neg = scores[~y_true]
    m, n = pos.size, neg.size
    if m == 0 or n == 0:
        nan = float("nan")
        return nan, np.zeros(0), np.zeros(0)

    # Midrank-based placement (handles ties), O((m+n) log(m+n)), fully
    # vectorised (no per-element Python loop - this is called on every
    # bootstrap iteration, so a Python-level tie loop is a major bottleneck
    # at real dataset scale, e.g. ~1M rows x thousands of iterations).
    def _midrank(x: np.ndarray) -> np.ndarray:
        n_ = len(x)
        if n_ == 0:
            return np.zeros(0, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        ranked = x[order]
        is_new_group = np.empty(n_, dtype=bool)
        is_new_group[0] = True
        is_new_group[1:] = ranked[1:] != ranked[:-1]
        group_id = np.cumsum(is_new_group) - 1
        one_based_pos = np.arange(1, n_ + 1, dtype=np.float64)
        group_sum = np.bincount(group_id, weights=one_based_pos)
        group_count = np.bincount(group_id).astype(np.float64)
        avg_rank_per_group = group_sum / group_count
        ranks_sorted = avg_rank_per_group[group_id]
        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks_sorted
        return out

    combined = np.concatenate([pos, neg])
    all_ranks = _midrank(combined)
    pos_ranks = all_ranks[:m]
    tx = _midrank(pos)
    ty = _midrank(neg)

    v01 = (pos_ranks - tx) / n  # per-positive placement (prob a random negative scores lower)
    v10 = 1.0 - (all_ranks[m:] - ty) / m  # per-negative placement
    auc = float(np.mean(v01))
    return auc, v01, v10


def delong_roc_variance(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """AUC and its DeLong variance for one binary score vector."""
    auc, v01, v10 = _binary_auc_and_cov(y_true, scores)
    if v01.size < 2 and v10.size < 2:
        return auc, float("nan")
    var = (np.var(v01, ddof=1) / v01.size if v01.size > 1 else 0.0) + (
        np.var(v10, ddof=1) / v10.size if v10.size > 1 else 0.0
    )
    return auc, float(var)


def delong_roc_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Two-sided DeLong test comparing two *correlated* ROC curves (same
    samples, two models). Returns AUCs, their difference, z, and p."""
    y_true = np.asarray(y_true).astype(bool)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    auc_a, v01_a, v10_a = _binary_auc_and_cov(y_true, scores_a)
    auc_b, v01_b, v10_b = _binary_auc_and_cov(y_true, scores_b)
    if not (np.isfinite(auc_a) and np.isfinite(auc_b)):
        return {"auc_a": auc_a, "auc_b": auc_b, "diff": float("nan"), "z": float("nan"), "p": float("nan")}

    m, n = v01_a.size, v10_a.size
    if m < 2 or n < 2:
        return {"auc_a": auc_a, "auc_b": auc_b, "diff": auc_a - auc_b, "z": float("nan"), "p": float("nan")}

    # 2x2 covariance matrix of (auc_a, auc_b) via the positive- and
    # negative-side placement covariances (Sun & Xu, 2014).
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1) / m
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1) / n
    cov = s01 + s10

    diff = auc_a - auc_b
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if not np.isfinite(var_diff) or var_diff <= 0:
        # Degenerate variance (e.g. identical score vectors): the two AUCs
        # either are exactly equal (a real, certain non-difference -> p=1)
        # or the placement covariance couldn't be estimated (report NaN
        # rather than a misleadingly precise p-value).
        if abs(diff) < 1e-12:
            return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "z": 0.0, "p": 1.0}
        return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "z": float("nan"), "p": float("nan")}

    z = diff / math.sqrt(var_diff)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "z": float(z), "p": float(p)}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def per_class_delong(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """One-vs-rest DeLong's test per class between two models' probability
    matrices, evaluated on the *same* samples (paired)."""
    labels = np.asarray(labels)
    out: dict = {}
    for idx, name in enumerate(class_names):
        y_bin = labels == idx
        if y_bin.sum() == 0 or (~y_bin).sum() == 0:
            out[name] = {"auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan"), "z": float("nan"), "p": float("nan")}
            continue
        out[name] = delong_roc_test(y_bin, probs_a[:, idx], probs_b[:, idx])
    return out


def macro_auroc_ovr(labels: np.ndarray, probs: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC (classes with only one label value
    present are skipped, matching sklearn's behaviour for degenerate
    classes)."""
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    aucs = []
    for idx in range(n_classes):
        y_bin = labels == idx
        if y_bin.sum() == 0 or (~y_bin).sum() == 0:
            continue
        auc, _, _ = _binary_auc_and_cov(y_bin, probs[:, idx])
        if np.isfinite(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def _ci_bounds(samples: np.ndarray, ci: float) -> tuple:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def case_bootstrap_paired_diff(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    case_ids: Sequence[str],
    *,
    metric_fn=macro_auroc_ovr,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = _CI,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI for metric_fn(model_a) - metric_fn(model_b),
    resampling whole *cases* (slides/patients) with replacement so that
    correlated cells from the same case move together, not independently.
    """
    labels = np.asarray(labels)
    case_ids = np.asarray(case_ids)
    unique_cases = np.unique(case_ids)
    n_cases = unique_cases.size
    case_to_rows = {c: np.where(case_ids == c)[0] for c in unique_cases}

    point_a = metric_fn(labels, probs_a)
    point_b = metric_fn(labels, probs_b)
    point_diff = point_a - point_b

    if n_cases < 2:
        nan = float("nan")
        return {
            "metric_a": point_a,
            "metric_b": point_b,
            "diff": point_diff,
            "ci_lo": nan,
            "ci_hi": nan,
            "n_cases": n_cases,
            "n_boot": n_boot,
        }

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sampled_cases = rng.choice(unique_cases, size=n_cases, replace=True)
        idx = np.concatenate([case_to_rows[c] for c in sampled_cases])
        a = metric_fn(labels[idx], probs_a[idx])
        b = metric_fn(labels[idx], probs_b[idx])
        diffs[k] = a - b
    lo, hi = _ci_bounds(diffs, ci)
    return {
        "metric_a": point_a,
        "metric_b": point_b,
        "diff": point_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_cases": int(n_cases),
        "n_boot": n_boot,
        "ci": ci,
    }


def classification_significance_block(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    case_ids: Sequence[str],
    class_names: Sequence[str],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = _CI,
    seed: int = 0,
) -> dict:
    """Combine per-class DeLong's tests with a case-level bootstrap CI on
    the macro-AUROC difference, and a verdict for whether model A is
    significantly different from model B on this test set."""
    per_class = per_class_delong(labels, probs_a, probs_b, class_names)
    macro = case_bootstrap_paired_diff(
        labels, probs_a, probs_b, case_ids, metric_fn=macro_auroc_ovr, n_boot=n_boot, ci=ci, seed=seed
    )
    macro_significant = bool(np.isfinite(macro["ci_lo"]) and (macro["ci_lo"] > 0 or macro["ci_hi"] < 0))
    class_p = [v["p"] for v in per_class.values() if np.isfinite(v.get("p", float("nan")))]
    any_class_significant = bool(any(p < 0.05 for p in class_p))
    return {
        "per_class_delong": per_class,
        "macro_auroc_bootstrap": macro,
        "macro_significant_95ci": macro_significant,
        "any_class_delong_p_below_0p05": any_class_significant,
        "verdict": "significantly_different" if macro_significant else "not_distinguishable_at_this_n",
    }
