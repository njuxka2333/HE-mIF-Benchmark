"""Post-training analysis for the benchmark's downstream tasks.

**Library, not an entrance.** ``training.pipeline`` calls these functions as the
second half of a task: training produces per-run artifacts, this turns them into
the reported statistics. There is deliberately no CLI here - a run of the
benchmark starts from training, so the statistics are always derived from runs
this repo just produced rather than from whatever happens to be on disk.

Per task:

``single_cell_classification``
    Rewrites Immunocto rows in ``single_cell_classification.csv`` from each
    arm's ``test_metrics.json``, then per-class DeLong + case-level paired
    bootstrap on macro-AUROC against UNI2 (or the top model on the output
    route). Reads ``test_result/predictions.csv``; writes
    ``significance_summary.json`` beside each route.

``prognosis``
    Best-epoch vs last-epoch C-index per fold, the selection inflation between
    them, a Shazam-style t(df=K-1) fold interval, and a patient-level bootstrap
    over pooled ``patient_risk.csv``. Writes ``prognosis_uncertainty.csv`` and
    augments ``prognosis_os.csv``.

``virtual_staining``
    Collects the per-model JSON into ``virtual_staining.csv`` (locked model
    order, not a score ranking), ``virtual_staining_significance.csv``
    (Spearman and AUPRC shuffle-null plus Otsu-positive prevalence), and
    ``virtual_staining_per_marker.csv``. Drops P/R/F1 at the cutoff 0.5.
    Flags any block still resampled by patch rather than by FOV.

``cell_semantic_segmentation`` / ``tls_region_segmentation``
    No confidence intervals - no published benchmark for these tasks reports
    them. Instead an evaluation-grid comparability audit: every arm within a
    dataset x route must share one patch count, slide count, instance count and
    set of class supports, or the macro F1 columns are not comparing the same
    thing. Writes ``comparability_report.json``.

Routing mirrors ``training.train``: a ``_POSTPROCESSORS`` table keyed by task.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.classification import (
    DEFAULT_N_BOOT,
    macro_auroc_ovr,
    per_class_delong,
)
from metrics.clinical import (
    fold_metric_summary,
    fold_t_interval,
    harrell_c_index,
    patient_bootstrap_interval,
    verdict_against_chance,
)

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
DEFAULT_SEED = 42


# ------------------------------------------------------------------ helpers
def _load_predictions(path: Path) -> dict:
    frame = pd.read_csv(path)
    prob_cols = sorted(
        (c for c in frame.columns if c.startswith("probability_class_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    frame = frame.sort_values("sample_id", kind="mergesort")
    labels = frame["label"].to_numpy(int)
    names = dict(zip(frame["label"].to_numpy(int), frame["label_name"].astype(str)))
    return {
        "sample_id": frame["sample_id"].to_numpy(),
        "labels": labels,
        "probs": frame[prob_cols].to_numpy(float),
        "case_ids": frame["case_id"].to_numpy(),
        "class_names": [names.get(i, str(i)) for i in range(len(prob_cols))],
    }


def _case_bootstrap(labels, probs_a, probs_b, case_ids, *, n_boot, seed, max_rows=None):
    """Paired bootstrap of the macro-AUROC difference, resampling whole cases.

    ``max_rows`` caps how many cells each case contributes, to bound runtime on
    the ~1M-row routes. The cap is drawn once, before the loop, so every
    resample sees the same cells for both models and the result is
    reproducible from ``seed``; the resampling unit stays the *case*. Capping
    cells within a case widens the interval slightly; it does not bias it. The
    point estimate always uses every row.
    """
    case_ids = np.asarray(case_ids)
    unique = np.unique(case_ids)
    rows = {c: np.where(case_ids == c)[0] for c in unique}
    point_a, point_b = macro_auroc_ovr(labels, probs_a), macro_auroc_ovr(labels, probs_b)
    rng = np.random.default_rng(seed)
    if max_rows is not None and labels.size > max_rows:
        per_case = max(2, int(max_rows // max(1, unique.size)))
        rows = {c: (rng.choice(r, size=per_case, replace=False) if r.size > per_case else r)
                for c, r in rows.items()}
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = np.concatenate([rows[c] for c in rng.choice(unique, size=unique.size, replace=True)])
        diffs[k] = macro_auroc_ovr(labels[idx], probs_a[idx]) - macro_auroc_ovr(labels[idx], probs_b[idx])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"metric_a": point_a, "metric_b": point_b, "diff": point_a - point_b,
            "ci_lo": float(lo), "ci_hi": float(hi), "n_cases": int(unique.size),
            "n_boot": n_boot, "max_rows": max_rows}


# ------------------------------------------------- single-cell classification
BASELINE_MODEL = "uni2"
LORA_REF_MODEL = "mipheivit"
SCC_CSV_MODEL = {"mipheivit": "miphei-vit", "hoptimus0": "h-optimus-0"}
SCC_LAYER = {"encoder_layer": "encoder", "output_layer": "virtual staining"}
IMMUNOCTO_CLASS_MAP = {
    "bcell": "B cell",
    "cd4": "CD4",
    "cd8": "CD8",
    "macrophage": "Macrophage",
    "other": "Other",
}


def _immunocto_csv_row(layer: str, model: str, blob: dict) -> dict:
    report = blob.get("lin_report") or {}
    row = {
        "dataset": "Immunocto",
        "layer": layer,
        "model": SCC_CSV_MODEL.get(model, model),
        "AUROC": blob.get("lin_auroc"),
        "AUPRC": blob.get("lin_auprc"),
        "Macro F1": blob.get("lin_macro_f1"),
        "Weighted F1": blob.get("lin_weighted_f1"),
        "Balanced Accuracy": blob.get("lin_bacc"),
        "Accuracy": blob.get("lin_acc"),
        "Loss": blob.get("loss"),
    }
    for src, dst in IMMUNOCTO_CLASS_MAP.items():
        block = report.get(src) or {}
        row[f"{dst} F1"] = block.get("f1-score")
        row[f"{dst} Precision"] = block.get("precision")
        row[f"{dst} Recall"] = block.get("recall")
        row[f"{dst} Support"] = block.get("support")
    return row


def _official_scc_run(dataset: str, route: str, model: str) -> Path:
    return DEFAULT_RESULTS_ROOT / "single_cell_cls" / dataset / route / model


def _load_official_scc(dataset: str, route: str, model: str) -> dict | None:
    path = _official_scc_run(dataset, route, model) / "test_result" / "predictions.csv"
    if path.is_file():
        return _load_predictions(path)
    return None


def _rewrite_immunocto_csv(root: Path) -> None:
    """Replace Immunocto rows in the task CSV from on-disk test_metrics.json."""
    immuno_root = root / "immunocto"
    if not immuno_root.is_dir():
        return
    new_rows = []
    for route_dir, layer in SCC_LAYER.items():
        base = immuno_root / route_dir
        if not base.is_dir():
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            path = model_dir / "test_metrics.json"
            if path.is_file():
                new_rows.append(_immunocto_csv_row(
                    layer, model_dir.name, json.loads(path.read_text())))
    if not new_rows:
        return
    immuno = pd.DataFrame(new_rows)
    csv_path = root / "single_cell_classification.csv"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        keep = frame[frame["dataset"] != "Immunocto"].copy()
        out = pd.concat([immuno, keep], ignore_index=True)
        for col in frame.columns:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[list(frame.columns)]
    else:
        out = immuno
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} immunocto_rows={len(immuno)}")


def _compare_models(loaded: dict, ref: str, others: list[str], args, *, max_rows=None) -> dict:
    report = {}
    for model in others:
        a, b = loaded[model], loaded[ref]
        if not np.array_equal(a["sample_id"], b["sample_id"]):
            raise SystemExit(f"sample_id mismatch: {model} vs {ref}")
        macro = _case_bootstrap(
            a["labels"], a["probs"], b["probs"], a["case_ids"],
            n_boot=args.n_boot, seed=args.seed, max_rows=max_rows,
        )
        block = {
            "per_class_delong": per_class_delong(a["labels"], a["probs"], b["probs"], a["class_names"]),
            "macro_auroc_bootstrap": macro,
            "macro_significant_95ci": bool(macro["ci_lo"] > 0 or macro["ci_hi"] < 0),
            "verdict": "significantly_different" if (macro["ci_lo"] > 0 or macro["ci_hi"] < 0)
                       else "not_distinguishable_at_this_n",
        }
        report[f"{model}_vs_{ref}"] = block
        print(
            f"{model:>12} vs {ref}: {macro['metric_a']:.4f} vs {macro['metric_b']:.4f} "
            f"(diff {macro['diff']:+.4f}, 95% CI [{macro['ci_lo']:+.4f}, {macro['ci_hi']:+.4f}], "
            f"n_cases={macro['n_cases']}) -> {block['verdict']}"
        )
    return report


def _compare_encoder_vs_output(root: Path, dataset: str, args) -> None:
    enc_base = root / dataset / "encoder_layer"
    out_base = root / dataset / "output_layer"
    if not enc_base.is_dir():
        return
    if not out_base.is_dir():
        official_out = DEFAULT_RESULTS_ROOT / "single_cell_cls" / dataset / "output_layer"
        out_base = official_out if official_out.is_dir() else out_base
    if not out_base.is_dir():
        return
    pca_models = sorted(
        m.name for m in enc_base.iterdir()
        if m.is_dir() and (m / "pca.json").is_file()
        and (m / "test_result" / "predictions.csv").is_file()
    )
    if not pca_models:
        return
    report = {}
    print(f"\n=== {dataset}: encoder@C vs output ===")
    for model in pca_models:
        out_pred = out_base / model / "test_result" / "predictions.csv"
        if not out_pred.is_file():
            print(f"  skip {model}: no output-route predictions")
            continue
        a = _load_predictions(enc_base / model / "test_result" / "predictions.csv")
        b = _load_predictions(out_pred)
        if not np.array_equal(a["sample_id"], b["sample_id"]):
            raise SystemExit(f"sample_id mismatch: {model} encoder@C vs output")
        n_rows = a["labels"].size
        max_rows = args.max_boot_rows if n_rows > (args.max_boot_rows or np.inf) else None
        macro = _case_bootstrap(
            a["labels"], a["probs"], b["probs"], a["case_ids"],
            n_boot=args.n_boot, seed=args.seed, max_rows=max_rows,
        )
        block = {
            "per_class_delong": per_class_delong(a["labels"], a["probs"], b["probs"], a["class_names"]),
            "macro_auroc_bootstrap": macro,
            "macro_significant_95ci": bool(macro["ci_lo"] > 0 or macro["ci_hi"] < 0),
            "verdict": "significantly_different" if (macro["ci_lo"] > 0 or macro["ci_hi"] < 0)
                       else "not_distinguishable_at_this_n",
        }
        report[f"{model}_encoder_pca_vs_output"] = block
        print(
            f"{model:>12} encoder@C vs output: {macro['metric_a']:.4f} vs {macro['metric_b']:.4f} "
            f"(diff {macro['diff']:+.4f}, 95% CI [{macro['ci_lo']:+.4f}, {macro['ci_hi']:+.4f}]) "
            f"-> {block['verdict']}"
        )
    if report:
        out = enc_base / "encoder_vs_output_significance.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"  wrote {out}")


def postprocess_single_cell_classification(args) -> None:
    root = args.results_root / "single_cell_cls"
    if not root.is_dir():
        print(f"skip single_cell_classification: {root} not found")
        return
    _rewrite_immunocto_csv(root)
    only = getattr(args, "only", None)
    for dataset in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for route in sorted(p.name for p in (root / dataset).iterdir() if p.is_dir()):
            if only and f"{dataset}/{route}" not in only:
                continue
            base = root / dataset / route
            models = sorted(
                m.name for m in base.iterdir()
                if (m / "test_result" / "predictions.csv").is_file()
            )
            if not models:
                continue
            loaded = {m: _load_predictions(base / m / "test_result" / "predictions.csv") for m in models}
            if root.resolve() != (DEFAULT_RESULTS_ROOT / "single_cell_cls").resolve():
                for model in (BASELINE_MODEL, LORA_REF_MODEL):
                    if model not in loaded:
                        extra = _load_official_scc(dataset, route, model)
                        if extra is not None:
                            loaded[model] = extra
            n_rows = loaded[models[0]]["labels"].size
            max_rows = args.max_boot_rows if n_rows > (args.max_boot_rows or np.inf) else None
            report = {}
            if BASELINE_MODEL in loaded:
                ref, others = BASELINE_MODEL, [m for m in models if m != BASELINE_MODEL]
                print(f"\n=== {dataset}/{route}: vs {ref} ===")
            else:
                scores = {m: macro_auroc_ovr(loaded[m]["labels"], loaded[m]["probs"]) for m in models}
                ref = max(scores, key=scores.get)
                others = [m for m in models if m != ref]
                print(f"\n=== {dataset}/{route}: no baseline in route; vs top model ({ref}) ===")
            report.update(_compare_models(loaded, ref, others, args, max_rows=max_rows))
            if (
                LORA_REF_MODEL in loaded
                and "hoptimus0" in loaded
                and LORA_REF_MODEL != ref
            ):
                print(f"\n=== {dataset}/{route}: LoRA effect (hoptimus0 vs {LORA_REF_MODEL}) ===")
                report.update(
                    _compare_models(loaded, LORA_REF_MODEL, ["hoptimus0"], args, max_rows=max_rows)
                )
            out = base / "significance_summary.json"
            out.write_text(json.dumps(report, indent=2))
            print(f"  wrote {out}")
        _compare_encoder_vs_output(root, dataset, args)


# ------------------------------------------------------------------ prognosis
def postprocess_prognosis(args) -> None:
    root = args.results_root / "prognosis_os"
    csv_path = root / "prognosis_os.csv"
    if not csv_path.is_file():
        print(f"skip prognosis: {csv_path} not found")
        return
    frame = pd.read_csv(csv_path)
    ds_dir = {"TCGA-COAD": "tcga_coad", "TCGA-LUAD": "tcga_luad"}
    route_dir = {"encoder": "encoder_layer", "virtual staining": "output_layer"}
    alias = {"miphei-vit": "mipheivit"}

    rows = []
    for _, rec in frame.iterrows():
        d = ds_dir.get(rec["dataset"])
        r = route_dir.get(rec["layer"])
        m = alias.get(rec["model"], rec["model"])
        if not (d and r):
            continue
        run = root / d / r / m
        summary = fold_metric_summary(sorted(run.glob("fold_*/history.csv")), "valid_c_index")
        last = summary["last"]

        boot = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        risk_files = sorted(run.glob("fold_*/test_result/patient_risk.csv"))
        if risk_files:
            pooled = pd.concat([pd.read_csv(f) for f in risk_files], ignore_index=True)
            pooled = pooled.drop_duplicates("case_id")
            if len(pooled) > 20:
                t = pooled["event_time"].to_numpy(float)
                e = (pooled["censorship"].to_numpy(float) == 0).astype(int)
                risk = pooled["risk"].to_numpy(float)
                boot = patient_bootstrap_interval(harrell_c_index, t, e, risk,
                                                  n_boot=args.n_boot, seed=args.seed,
                                                  event_index=1)
        rows.append({
            "dataset": rec["dataset"], "layer": rec["layer"], "model": rec["model"],
            "c_index_reported_best_epoch": rec.get("C-index"),
            "c_index_last_epoch": last["mean"], "c_index_last_epoch_std": last["std"],
            "selection_inflation": summary["inflation"],
            "fold_t_ci_lo": last["ci_lo"], "fold_t_ci_hi": last["ci_hi"],
            "n_folds": last["n_folds"],
            "verdict": verdict_against_chance(last["ci_lo"]),
            "patient_boot_point": boot["point"],
            "patient_boot_ci_lo": boot["ci_lo"], "patient_boot_ci_hi": boot["ci_hi"],
        })

    out = pd.DataFrame(rows)
    dest = root / "prognosis_uncertainty.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out)} runs)")
    for _, r in out.sort_values(["dataset", "layer", "c_index_last_epoch"],
                                ascending=[True, True, False]).iterrows():
        print(f"  {r['dataset']:<10}{r['layer']:<17}{r['model']:<13}"
              f"last={r['c_index_last_epoch']:.4f} "
              f"CI[{r['fold_t_ci_lo']:.3f},{r['fold_t_ci_hi']:.3f}] "
              f"infl={r['selection_inflation']:+.4f}  {r['verdict']}")


# ------------------------------------------------------------ virtual staining
VSTAIN_MODELS_DIR = "cell_niches"
# Locked model order, not a score ranking. Do not sort the paper table by Spearman.
VSTAIN_DISPLAY = {
    "mipheivit": "MIPHEI-ViT",
    "hemit": "HEMIT",
    "histoplexer": "HistoPlexer",
    "gigatime": "GigaTIME",
    "rosie": "ROSIE",
}
VSTAIN_MODEL_ORDER = list(VSTAIN_DISPLAY)
_PRF_KEYS = (
    "precision_at_0p5_mean_patch",
    "recall_at_0p5_mean_patch",
    "f1_at_0p5_mean_patch",
)


def _vstain_metrics_path(model_dir: Path) -> Path | None:
    return next(
        (
            c
            for c in (
                model_dir / "vstain_shared_metrics.json",
                model_dir / f"{model_dir.name}_vstain_shared_metrics.json",
            )
            if c.is_file()
        ),
        None,
    )


def _strip_prf(per_marker: dict) -> dict:
    return {
        name: {k: v for k, v in vals.items() if k not in _PRF_KEYS}
        for name, vals in per_marker.items()
    }


def postprocess_virtual_staining(args) -> None:
    """Collect per-model JSON into the three paper tables.

    Significance itself is computed during scoring (or from the 4 µm cache).
    This pass only gathers those blocks, strips the invalid P/R/F1@0.5 columns,
    and writes the official CSVs. Rows are in locked model order, not score order.
    """
    vs_root = args.results_root / "virtual_staining"
    root = vs_root / VSTAIN_MODELS_DIR
    if not root.is_dir():
        print(f"skip virtual_staining: {root} not found")
        return
    blobs: dict[str, dict] = {}
    missing, stale = [], []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics = _vstain_metrics_path(model_dir)
        if metrics is None:
            missing.append(model_dir.name)
            continue
        blob = json.loads(metrics.read_text())
        if not blob.get("significance"):
            missing.append(model_dir.name)
            continue
        blobs[model_dir.name] = blob

    if missing:
        print("  no significance block (re-run with --significance): " + ", ".join(missing))
    if not blobs:
        print("  nothing to tabulate")
        return

    main_rows, sig_rows, marker_rows = [], [], []
    for model in VSTAIN_MODEL_ORDER:
        if model not in blobs:
            continue
        blob = blobs[model]
        ranking = blob.get("ranking") or {}
        geo = blob.get("geometry") or {}
        sig = blob.get("significance") or {}
        cd3 = sig.get("CD3") or {}
        boot = cd3.get("bootstrap") or {}
        auprc_blk = boot.get("auprc") or {}
        null = cd3.get("null") or {}
        pos = cd3.get("positive_frac") or {}
        unit = cd3.get("resample_unit") or boot.get("resample_unit") or "patch"
        if unit != "fov":
            stale.append(f"{model}/CD3")
        pred_hw = geo.get("pred_hw")
        if isinstance(pred_hw, (list, tuple)) and len(pred_hw) == 2:
            pred_hw_s = f"{pred_hw[0]}x{pred_hw[1]}"
        else:
            pred_hw_s = blob.get("pred_hw") or ranking.get("pred_hw")
        overlap = ranking.get("overall_markers") or []
        main_rows.append({
            "model": model,
            "display_name": VSTAIN_DISPLAY.get(model, model),
            "protocol": blob.get("protocol"),
            "n_patches": blob.get("n_patches") or boot.get("n_patches"),
            "n_fovs": boot.get("n_fovs"),
            "pred_hw": pred_hw_s,
            "model_mpp": geo.get("model_mpp"),
            "network_tile_px": geo.get("network_tile_px"),
            "feed": geo.get("feed"),
            "fed_tile_px": geo.get("fed_tile_px"),
            "fed_tile_um": geo.get("fed_tile_um"),
            "n_overlap_markers": ranking.get("n_overlap_markers"),
            "overlap_markers": "|".join(overlap) if isinstance(overlap, list) else overlap,
            "cd3_spearman": ranking.get("cd3_spearman"),
            "cd3_spearman_4um": (blob.get("per_marker") or {}).get("CD3", {}).get(
                "spearman_r_mean_patch_4um"
            ),
            "cd3_auprc": ranking.get("cd3_auprc") or auprc_blk.get("point"),
            "cd3_auprc_ci_lo": auprc_blk.get("ci_lo"),
            "cd3_auprc_ci_hi": auprc_blk.get("ci_hi"),
            "cd3_positive_frac": ranking.get("cd3_positive_frac") or pos.get("mean_patch"),
            "cd3_auprc_null_mean": (null.get("auprc") or {}).get("null_mean"),
            "cd3_auprc_p_vs_null": cd3.get("auprc_p"),
            "cd3_auprc_verdict": cd3.get("auprc_verdict"),
            "cd3_pearson": ranking.get("cd3_pearson"),
            "cd3_verdict": cd3.get("verdict"),
            "cd3_spearman_p_vs_null": cd3.get("spearman_p"),
            "cd3_spearman_p_is_floor": cd3.get("spearman_p_is_permutation_floor"),
            "shuffle_null_spearman_mean": (null.get("spearman") or {}).get("null_mean"),
            "permutation_floor": cd3.get("permutation_floor"),
            "overall_spearman_mean": ranking.get("overall_spearman_mean"),
            "overall_pearson_mean": ranking.get("overall_pearson_mean"),
            "overall_comparable_across_models": ranking.get("overall_comparable_across_models"),
            "core4_comparable": ranking.get("core4_comparable"),
            "headline_metric": ranking.get("headline_metric"),
            "headline_detection": ranking.get("headline_detection"),
            "report_bin_um": ranking.get("report_bin_um"),
            "cache_bin_um": ranking.get("cache_bin_um"),
            "normalisation": ranking.get("normalisation"),
        })
        for marker, block in sig.items():
            boot_m = block.get("bootstrap") or {}
            unit_m = block.get("resample_unit") or boot_m.get("resample_unit") or "patch"
            if unit_m != "fov":
                stale.append(f"{model}/{marker}")
            ap = boot_m.get("auprc") or {}
            sig_rows.append({
                "model": model,
                "marker": marker,
                "spearman": (boot_m.get("spearman") or {}).get("point"),
                "auprc": ap.get("point"),
                "auprc_ci_lo": ap.get("ci_lo"),
                "auprc_ci_hi": ap.get("ci_hi"),
                "positive_frac": (block.get("positive_frac") or {}).get("mean_patch"),
                "auprc_null_mean": (block.get("null") or {}).get("auprc", {}).get("null_mean"),
                "auprc_p_vs_null": block.get("auprc_p"),
                "auprc_verdict": block.get("auprc_verdict"),
                "null_mean": (block.get("null") or {}).get("spearman", {}).get("null_mean"),
                "p_vs_null": block.get("spearman_p"),
                "p_is_permutation_floor": block.get("spearman_p_is_permutation_floor"),
                "resample_unit": unit_m,
                "n_fovs": boot_m.get("n_fovs"),
                "n_patches": boot_m.get("n_patches"),
                "verdict": block.get("verdict"),
            })
        for marker, vals in _strip_prf(blob.get("per_marker") or {}).items():
            marker_rows.append({
                "model": model,
                "display_name": VSTAIN_DISPLAY.get(model, model),
                "marker": marker,
                "is_headline_cd3": marker == "CD3",
                **vals,
            })

    extra = [m for m in blobs if m not in VSTAIN_MODEL_ORDER]
    if extra:
        print(f"  WARNING: models not in locked order left out of main table: {extra}")
    if stale:
        print(f"  WARNING: {len(stale)} blocks resampled PATCHES, not FOVs - intervals are too narrow.")

    vs_root.mkdir(parents=True, exist_ok=True)
    main = pd.DataFrame(main_rows)
    sig = pd.DataFrame(sig_rows)
    per = pd.DataFrame(marker_rows)
    main.to_csv(vs_root / "virtual_staining.csv", index=False)
    sig.to_csv(vs_root / "virtual_staining_significance.csv", index=False)
    per.to_csv(vs_root / "virtual_staining_per_marker.csv", index=False)
    print(f"wrote {vs_root / 'virtual_staining.csv'}  ({len(main)} models, locked order)")
    print(f"wrote {vs_root / 'virtual_staining_significance.csv'}  ({len(sig)} marker x model)")
    print(f"wrote {vs_root / 'virtual_staining_per_marker.csv'}  ({len(per)} rows)")
    for _, r in main.iterrows():
        print(
            f"  {r['display_name']:<13} Spearman {r['cd3_spearman']:.3f}  "
            f"AUPRC {r['cd3_auprc']:.3f} vs chance {r['cd3_positive_frac']:.3f}  "
            f"Spearman-null {r['cd3_verdict']}  AUPRC-null {r['cd3_auprc_verdict']}"
        )


# ------------------------------------------------------------- segmentation
SEGMENTATION_SPECS = {
    "cell_semantic_segmentation": {
        "dir": "cell_semantic_seg", "csv": "cell_semantic_segmentation.csv",
    },
    "tls_region_segmentation": {
        "dir": "tls_seg", "csv": "tls_segmentation.csv",
    },
}
AREA_TOLERANCE = 0.02  # 2% - resampling a fixed region onto a different grid never lands exactly


def _run_geometry(run_dir: Path, fallback: dict) -> dict:
    """``target_mpp`` and ``target_patch_size`` for a run.

    The output-layer runs carry no ``resolved_config.json``; a model's geometry
    is a property of the model, so its encoder run supplies the fallback.
    """
    cfg = run_dir / "resolved_config.json"
    if cfg.is_file():
        blob = json.loads(cfg.read_text())
        experiment = blob.get("experiment") or {}
        mpp = experiment.get("target_mpp") or (blob.get("run") or {}).get("fallback_mpp")
        return {"mpp": mpp, "patch_px": experiment.get("target_patch_size")}
    return dict(fallback)


def _evaluated_area_mm2(metrics: dict, mpp: float | None) -> float | None:
    """Physical area of the evaluated ground truth, from the class supports.

    Supports are pixel counts, so a model working at 0.25 mpp reports 4x the
    pixels of one at 0.5 mpp *for the same tissue*. Comparing the raw numbers
    says the arms were scored on different data when they were not; converting
    to mm^2 is the comparison that means something.
    """
    if not mpp:
        return None
    supports = [v for k, v in metrics.items() if k.endswith("/support")]
    if not supports:
        return None
    return float(sum(supports)) * float(mpp) ** 2 / 1e6


def _postprocess_segmentation(args, task: str) -> None:
    """Evaluation-grid comparability audit, in physical units. No confidence
    intervals - no published benchmark for these tasks reports them.

    The arms tile the same slides differently because each model has its own
    native input size and mpp (rosie 64 um tiles, uni2 112, mipheivit/gigatime
    128, hemit/histoplexer 256). Different patch counts and different support
    *pixel* counts are therefore expected and are not evidence of different
    data. What must match is the physical extent actually scored, so that is
    what this checks, to ``AREA_TOLERANCE``.

    Instance counts are resolution-independent and are reported separately.
    They still vary with tile size (a cell straddling a tile boundary is scored
    as two fragments), which is an inherent consequence of each model's native
    resolution rather than a protocol defect - but it is a systematic
    disadvantage for finely-tiled models and belongs in the paper's limitations,
    so the spread and its tile sizes are recorded rather than hidden.
    """
    spec = SEGMENTATION_SPECS[task]
    root = args.results_root / spec["dir"]
    csv_path = root / spec["csv"]
    if not csv_path.is_file():
        print(f"skip {task}: {csv_path} not found")
        return

    # geometry per model, from whichever run recorded it
    geometry: dict[str, dict] = {}
    for cfg in root.glob("**/resolved_config.json"):
        geometry.setdefault(cfg.parent.name, _run_geometry(cfg.parent, {}))

    # keyed by (dataset, route, model); ``dataset`` is None for tasks whose
    # results are not split by dataset (TLS), and must be part of the key for
    # those that are (PathoCell and Lizard sit side by side under cellseg).
    measured: dict[tuple[str | None, str, str], dict] = {}
    for metrics_path in root.glob("**/test_metrics.json"):
        run_dir = metrics_path.parent
        model = run_dir.name
        route = run_dir.parent.name
        parent = run_dir.parent.parent
        dataset_dir = None if parent == root else parent.name
        blob = json.loads(metrics_path.read_text())
        metrics = blob.get("metrics", blob)
        geo = _run_geometry(run_dir, geometry.get(model, {}))
        mpp, px = geo.get("mpp"), geo.get("patch_px")
        measured[(dataset_dir, route, model)] = {
            "mpp": mpp,
            "tile_um": (float(px) * float(mpp)) if (px and mpp) else None,
            "patches": metrics.get("patches"),
            "instances": metrics.get("instance_count"),
            "area_mm2": _evaluated_area_mm2(metrics, mpp),
        }

    report, flags = {}, {}
    frame = pd.read_csv(csv_path)
    route_dir = {"encoder": "encoder_layer", "virtual staining": "output_layer"}
    for (dataset, layer), sub in frame.groupby(["dataset", "layer"], dropna=False):
        key = f"{dataset}/{layer}"
        route = route_dir.get(layer, layer)
        def _arm(model: str) -> dict:
            slug = str(model).replace("-", "")
            for dataset_key in (str(dataset).lower(), None):
                hit = measured.get((dataset_key, route, slug))
                if hit is not None:
                    return hit
            return {}

        arms = {m: _arm(m) for m in sub["model"]}

        areas = {m: a["area_mm2"] for m, a in arms.items() if a.get("area_mm2")}
        comparable, reason = True, "evaluated area matches across arms"
        outliers: dict[str, float] = {}
        if len(areas) > 1:
            reference = float(np.median(list(areas.values())))
            outliers = {m: v for m, v in areas.items()
                        if abs(v - reference) / reference > AREA_TOLERANCE}
            comparable = not outliers
            if outliers:
                reason = "arms scored a different physical extent"
        elif not areas:
            reason = "no class supports recorded; area could not be checked"

        instances = {m: a["instances"] for m, a in arms.items() if a.get("instances")}
        spread = (max(instances.values()) / min(instances.values())) if len(instances) > 1 else None
        report[key] = {
            "comparable": comparable,
            "reason": reason,
            "evaluated_area_mm2": areas,
            "area_outliers_mm2": outliers,
            "tile_um": {m: a.get("tile_um") for m, a in arms.items()},
            "instance_counts": instances,
            "instance_spread": spread,
        }
        for idx in sub.index:
            flags[idx] = comparable
        status = "OK" if comparable else f"NOT COMPARABLE - {reason}"
        print(f"  {key:<28}{len(sub)} arms  {status}")
        if areas:
            print(f"      evaluated area {min(areas.values()):.1f}-{max(areas.values()):.1f} mm2 "
                  f"across tiles of {sorted({a['tile_um'] for a in arms.values() if a.get('tile_um')})} um")
        for model, value in outliers.items():
            print(f"      {model}: {value:.1f} mm2 vs {np.median(list(areas.values())):.1f} mm2 median")
        if spread and spread > 1.02:
            fine = min(instances, key=lambda m: arms[m].get("tile_um") or 1e9)
            coarse = max(instances, key=lambda m: arms[m].get("tile_um") or 0)
            print(f"      instance fragmentation {spread:.2f}x "
                  f"({fine} at {arms[fine].get('tile_um'):.0f} um -> {instances[fine]:,}; "
                  f"{coarse} at {arms[coarse].get('tile_um'):.0f} um -> {instances[coarse]:,}) "
                  f"- boundary-split cells, report as a limitation")

    frame["grid_comparable"] = [flags.get(i, False) for i in frame.index]
    shutil.copy(csv_path, csv_path.with_suffix(".csv.bak"))
    frame.to_csv(csv_path, index=False)
    out = root / "comparability_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {csv_path} (backup .csv.bak) and {out}")
    bad = [k for k, v in report.items() if not v["comparable"]]
    if bad:
        print(f"  {len(bad)} of {len(report)} route(s) NOT comparable: {bad}")
        print("  Re-run only the outlier arms named above; the rest share one evaluated extent.")


def postprocess_cell_semantic_segmentation(args) -> None:
    _postprocess_segmentation(args, "cell_semantic_segmentation")


def postprocess_tls_region_segmentation(args) -> None:
    _postprocess_segmentation(args, "tls_region_segmentation")


_POSTPROCESSORS = {
    "single_cell_classification": postprocess_single_cell_classification,
    "prognosis": postprocess_prognosis,
    "virtual_staining": postprocess_virtual_staining,
    "cell_semantic_segmentation": postprocess_cell_semantic_segmentation,
    "tls_region_segmentation": postprocess_tls_region_segmentation,
}
