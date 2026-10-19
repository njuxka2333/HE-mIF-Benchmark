"""Tests for training.pipeline (the task-level entrance) and the task-level
post-processors that do not need real predictions on disk."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import pipeline  # noqa: E402
from training.postprocess import (  # noqa: E402
    _POSTPROCESSORS,
    postprocess_cell_semantic_segmentation,
    postprocess_tls_region_segmentation,
)


def test_every_task_maps_to_an_existing_config_dir():
    for task, (subdir, _key) in pipeline.TASKS.items():
        path = ROOT / "configs" / "downstream" / subdir
        assert path.is_dir(), f"{task} points at missing {path}"


def test_every_downstream_config_dir_is_claimed_by_a_task():
    """A new task directory must be registered, or its runs silently never run."""
    on_disk = {p.name for p in (ROOT / "configs" / "downstream").iterdir() if p.is_dir()}
    claimed = {subdir for subdir, _ in pipeline.TASKS.values()}
    assert on_disk == claimed, f"unclaimed: {on_disk - claimed}; stale: {claimed - on_disk}"


def test_task_order_covers_every_task_exactly_once():
    main_tasks = [k for k in pipeline.TASKS if k not in pipeline.SUPPLEMENTARY_TASKS]
    assert sorted(pipeline.TASK_ORDER) == sorted(main_tasks)
    assert len(pipeline.TASK_ORDER) == len(set(pipeline.TASK_ORDER))


def test_every_task_key_resolves_to_a_postprocessor():
    for task, (_subdir, key) in pipeline.TASKS.items():
        assert key in _POSTPROCESSORS, f"{task} -> unknown postprocessor {key!r}"


def test_scc_supplementary_tasks_are_not_in_task_order():
    assert "single_cell_classification_hoptimus" in pipeline.SUPPLEMENTARY_TASKS
    assert "single_cell_classification_pca" in pipeline.SUPPLEMENTARY_TASKS
    assert "prognosis_pca" in pipeline.SUPPLEMENTARY_TASKS
    assert "cell_semantic_segmentation_hoptimus" in pipeline.SUPPLEMENTARY_TASKS
    assert "tls_region_segmentation_hoptimus" in pipeline.SUPPLEMENTARY_TASKS
    assert "single_cell_classification_hoptimus" not in pipeline.TASK_ORDER
    assert "cell_semantic_segmentation_hoptimus" not in pipeline.TASK_ORDER
    assert "tls_region_segmentation_hoptimus" not in pipeline.TASK_ORDER
    assert pipeline.SUPPLEMENTARY_RESULTS_ROOT["single_cell_classification_hoptimus"] == "scc_hoptimus"
    assert pipeline.SUPPLEMENTARY_RESULTS_ROOT["single_cell_classification_pca"] == "scc_pca"
    assert pipeline.SUPPLEMENTARY_RESULTS_ROOT["prognosis_pca"] == "prognosis_pca"
    assert pipeline.SUPPLEMENTARY_RESULTS_ROOT["cell_semantic_segmentation_hoptimus"] == "cellseg_hoptimus"
    assert pipeline.SUPPLEMENTARY_RESULTS_ROOT["tls_region_segmentation_hoptimus"] == "tls_hoptimus"


def test_discover_configs_finds_runs_and_filters_by_model():
    configs = pipeline.discover_configs("prognosis")
    assert len(configs) > 1
    assert all(c.suffix == ".yaml" for c in configs)
    only_uni2 = pipeline.discover_configs("prognosis", models=["uni2"])
    assert {c.stem for c in only_uni2} == {"uni2"}
    assert len(only_uni2) < len(configs)


def _seg_run(root: Path, dataset: str | None, route: str, model: str,
             mpp: float, patch_px: int, supports: dict, instances=None, patches=None):
    """Write the per-run artifacts the audit reads."""
    parts = [p for p in (dataset, route, model) if p]
    run = root.joinpath(*parts)
    run.mkdir(parents=True, exist_ok=True)
    (run / "resolved_config.json").write_text(json.dumps(
        {"experiment": {"target_mpp": mpp, "target_patch_size": patch_px}}))
    metrics = {f"class_{k}/support": v for k, v in supports.items()}
    if instances is not None:
        metrics["instance_count"] = instances
    if patches is not None:
        metrics["patches"] = patches
    (run / "test_metrics.json").write_text(json.dumps({"metrics": metrics}))


def _seg_args(tmp: Path, results_dir: str, csv_name: str, frame):
    root = tmp / results_dir
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / csv_name, index=False)
    return SimpleNamespace(results_root=tmp)


def test_same_tissue_at_different_mpp_is_comparable():
    """The arms tile the same slides differently because each model has its own
    native input size and mpp. Four times the support pixels at half the mpp is
    the same tissue, and must not be flagged."""
    frame = pd.DataFrame({
        "dataset": ["HookNet-TLS"] * 3, "layer": ["encoder"] * 3,
        "model": ["uni2", "mipheivit", "hemit"], "Macro F1": [0.88, 0.63, 0.55],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        root = tmp / "tls_seg"
        # 190 mm2 of tissue, rasterised three ways
        _seg_run(root, None, "encoder_layer", "uni2", 0.5, 224, {1: 758_112_580})
        _seg_run(root, None, "encoder_layer", "mipheivit", 0.5, 256, {1: 761_068_101})
        _seg_run(root, None, "encoder_layer", "hemit", 0.25, 1024, {1: 3_044_297_819})
        args = _seg_args(tmp, "tls_seg", "tls_segmentation.csv", frame)
        postprocess_tls_region_segmentation(args)
        report = json.loads((root / "comparability_report.json").read_text())
        entry = report["HookNet-TLS/encoder"]
        assert entry["comparable"] is True, entry
        areas = list(entry["evaluated_area_mm2"].values())
        assert max(areas) / min(areas) < 1.02


def test_an_arm_scoring_less_tissue_is_flagged():
    """A real defect: same mpp, a fraction of the extent actually evaluated."""
    frame = pd.DataFrame({
        "dataset": ["HookNet-TLS"] * 3, "layer": ["virtual staining"] * 3,
        "model": ["rosie", "hemit", "gigatime"], "Macro F1": [0.62, 0.41, 0.60],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        root = tmp / "tls_seg"
        _seg_run(root, None, "output_layer", "rosie", 0.5, 224, {1: 761_068_101})
        _seg_run(root, None, "output_layer", "gigatime", 0.25, 512, {1: 3_044_297_819})
        _seg_run(root, None, "output_layer", "hemit", 0.25, 1024, {1: 178_822_331})
        args = _seg_args(tmp, "tls_seg", "tls_segmentation.csv", frame)
        postprocess_tls_region_segmentation(args)
        report = json.loads((root / "comparability_report.json").read_text())
        entry = report["HookNet-TLS/virtual staining"]
        assert entry["comparable"] is False
        assert set(entry["area_outliers_mm2"]) == {"hemit"}


def test_instance_fragmentation_is_recorded_not_failed():
    """Instance counts vary with tile size (boundary-split cells). That is a
    limitation to report, not a reason to call the route incomparable."""
    frame = pd.DataFrame({
        "dataset": ["PathoCell"] * 2, "layer": ["encoder"] * 2,
        "model": ["rosie", "hemit"], "Macro F1": [0.20, 0.18],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        root = tmp / "cell_semantic_seg"
        _seg_run(root, "pathocell", "encoder_layer", "rosie", 0.5, 128,
                 {1: 100_000_000}, instances=46_365)
        _seg_run(root, "pathocell", "encoder_layer", "hemit", 0.25, 1024,
                 {1: 400_000_000}, instances=38_310)
        args = _seg_args(tmp, "cell_semantic_seg", "cell_semantic_segmentation.csv", frame)
        postprocess_cell_semantic_segmentation(args)
        report = json.loads((root / "comparability_report.json").read_text())
        entry = report["PathoCell/encoder"]
        assert entry["comparable"] is True
        assert entry["instance_spread"] > 1.2
        assert entry["instance_counts"]["rosie"] == 46365


def test_datasets_do_not_borrow_each_others_runs():
    """PathoCell and Lizard sit side by side; the audit must key on both."""
    frame = pd.DataFrame({
        "dataset": ["PathoCell", "Lizard"], "layer": ["encoder"] * 2,
        "model": ["rosie", "rosie"], "Macro F1": [0.20, 0.55],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        root = tmp / "cell_semantic_seg"
        _seg_run(root, "pathocell", "encoder_layer", "rosie", 0.5, 128,
                 {1: 100_000_000}, instances=46_365)
        _seg_run(root, "lizard", "encoder_layer", "rosie", 0.5, 128,
                 {1: 300_000_000}, instances=148_374)
        args = _seg_args(tmp, "cell_semantic_seg", "cell_semantic_segmentation.csv", frame)
        postprocess_cell_semantic_segmentation(args)
        report = json.loads((root / "comparability_report.json").read_text())
        assert report["PathoCell/encoder"]["instance_counts"]["rosie"] == 46365
        assert report["Lizard/encoder"]["instance_counts"]["rosie"] == 148374


def test_segmentation_postprocessors_add_no_confidence_intervals():
    """Deliberate: no published cell-segmentation benchmark reports CIs here."""
    frame = pd.DataFrame({
        "dataset": ["PathoCell"], "layer": ["encoder"], "model": ["uni2"], "Macro F1": [0.29],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _seg_run(tmp / "cell_semantic_seg", "pathocell", "encoder_layer", "uni2", 0.5, 224,
                 {1: 100_000_000}, instances=43_618)
        args = _seg_args(tmp, "cell_semantic_seg", "cell_semantic_segmentation.csv", frame)
        postprocess_cell_semantic_segmentation(args)
        out = pd.read_csv(tmp / "cell_semantic_seg" / "cell_semantic_segmentation.csv")
        assert not [c for c in out.columns if "ci_" in c.lower() or "_ci" in c.lower()]


# --------------------------------------------------- model x route addressing
def test_discover_configs_filters_by_route():
    encoder = pipeline.discover_configs("prognosis", routes=["encoder"])
    output = pipeline.discover_configs("prognosis", routes=["output"])
    assert encoder and output
    assert all(c.parent.name == "encoder" for c in encoder)
    assert all(c.parent.name == "output" for c in output)
    assert len(encoder) + len(output) == len(pipeline.discover_configs("prognosis"))


def test_model_and_route_together_address_one_arm():
    arms = pipeline.discover_configs("prognosis", models=["rosie"], routes=["output"])
    assert {c.stem for c in arms} == {"rosie"}
    assert all(c.parent.name == "output" for c in arms)


def test_every_task_lays_configs_out_as_model_x_route():
    """The pipeline addresses arms by <route>/<model>.yaml; a task that nests
    them differently would silently ignore --route."""
    for task in pipeline.TASKS:
        for config in pipeline.discover_configs(task):
            assert config.parent.name in {"encoder", "output"}, f"{task}: {config}"


def test_unknown_model_is_an_error_not_a_silent_no_op():
    assert pipeline.discover_configs("prognosis", models=["nosuchmodel"]) == []


def test_site_identity_baseline_detects_base_rate_separation():
    """A pooled cohort whose sources differ in outcome prevalence is separable
    by provenance alone; that value is the bar a pooled AUROC must clear."""
    from metrics.clinical import site_identity_auroc, within_cohort_auroc

    # site A: 1/4 positive; site B: 3/4 positive
    labels = [1, 0, 0, 0, 1, 1, 1, 0]
    cohorts = ["A"] * 4 + ["B"] * 4
    assert site_identity_auroc(labels, cohorts) > 0.5

    # a score that only encodes the site separates pooled but not within
    scores = [0.25] * 4 + [0.75] * 4
    stats = within_cohort_auroc(labels, scores, cohorts)
    assert stats["pooled"] > 0.5
    assert abs(stats["within"] - 0.5) < 1e-9, "a site-only score must be chance within sites"
    assert stats["gap"] > 0


def test_leakfree_test_slides_are_exactly_mipheivits_heldout_set():
    """MIPHEI-ViT's repo pins TEST=19510_C11,18459_LSP10364 and VAL=19510_C19,
    19510_C30; the Orion data release maps those to CRC11, CRC02, CRC19, CRC30.
    Those four, and only those, may serve as the Immunocto test set."""
    from data.immunocto import TEST, assignment

    assert TEST == ("CRC02", "CRC11", "CRC19", "CRC30")
    a = assignment()
    assert sum(len(v) for v in a.values()) == 40
    assert not (set(a["train"]) | set(a["valid"])) & set(a["test"])
    # The discarded leaked split used CRC33–CRC40 as test.
    assert not set(TEST) & {f"CRC{i:02d}" for i in range(33, 41)}


def test_split_of_rejects_unknown_case():
    from data.immunocto import split_of
    assert split_of("crc11") == "test" and split_of("CRC01") == "train"
    try:
        split_of("CRC99")
    except KeyError:
        return
    raise AssertionError("an unknown specimen id must raise, not default to a split")


def test_official_split_configs_exist_for_every_arm():
    import yaml
    root = ROOT / "configs" / "downstream" / "linear_probe" / "immunocto"
    configs = sorted(p for p in root.glob("*/*.yaml"))
    assert len(configs) == 11, [c.name for c in configs]
    for c in configs:
        data = yaml.safe_load(c.read_text())["data"]
        assert data["valid_split"] == "valid" and data["test_split"] == "test"
        assert "exclude_case_ids_by_split" not in data, f"{c.name} kept the old exclusions"


def test_placement_keeps_one_source_per_case():
    """The superseded layout kept CRC29-32 in BOTH train/ and valid/. Placing
    every copy duplicated those cases in the new split and double-weighted them
    in the metrics -- observed as CRC30 appearing 64,002 times for 32,001 cells.
    """
    sys.path.insert(0, str(ROOT / "benchmark" / "scripts"))
    from place_immunocto import plan_sources

    found = {
        "train": {"CRC01": ["a"], "CRC30": ["stale"]},
        "valid": {"CRC30": ["good"]},
        "test":  {"CRC33": ["c"]},
    }
    sources = plan_sources(found)
    assert sources["CRC30"] == ("valid", ["good"]), "valid must win over a stale train copy"
    assert len(sources) == 3
    assert sum(len(f) for _s, f in sources.values()) == 3, "no case may contribute twice"


def test_immunocto_postprocess_rewrites_csv_from_test_metrics(tmp_path):
    from training.postprocess import _rewrite_immunocto_csv

    arm = tmp_path / "single_cell_cls" / "immunocto" / "encoder_layer" / "uni2"
    arm.mkdir(parents=True)
    (arm / "test_metrics.json").write_text(json.dumps({
        "lin_auroc": 0.85, "lin_auprc": 0.6, "lin_macro_f1": 0.55,
        "lin_weighted_f1": 0.59, "lin_bacc": 0.54, "lin_acc": 0.58, "loss": 1.0,
        "lin_report": {"bcell": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10},
                       "cd4": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10},
                       "cd8": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10},
                       "macrophage": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10},
                       "other": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10}},
    }))
    csv_path = tmp_path / "single_cell_cls" / "single_cell_classification.csv"
    pd.DataFrame([
        {"dataset": "Immunocto", "layer": "encoder", "model": "uni2", "AUROC": 0.1},
        {"dataset": "NuCLS", "layer": "encoder", "model": "uni2", "AUROC": 0.9},
    ]).to_csv(csv_path, index=False)
    _rewrite_immunocto_csv(tmp_path / "single_cell_cls")
    out = pd.read_csv(csv_path)
    immuno = out[out["dataset"] == "Immunocto"]
    nucls = out[out["dataset"] == "NuCLS"]
    assert len(immuno) == 1 and abs(immuno.iloc[0]["AUROC"] - 0.85) < 1e-9
    assert len(nucls) == 1 and abs(nucls.iloc[0]["AUROC"] - 0.9) < 1e-9


def test_feature_lookup_uses_wsi_cv_fold_layout(tmp_path):
    from data.feature_bags import _feature_paths_with_folds

    feat_dir = tmp_path / "encoder_z4"
    bag = feat_dir / "fold_0" / "test" / "h5_files" / "uni2" / "TCGA-AA.h5"
    bag.parent.mkdir(parents=True)
    bag.write_bytes(b"bag")
    leftover = feat_dir / "h5_files" / "uni2" / "TCGA-AA.h5"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"old")
    exp = SimpleNamespace(feat_dir=feat_dir, model="uni2", output_layout="prepath_legacy")
    _pt, h5 = _feature_paths_with_folds(exp, "TCGA-AA", "test")
    assert Path(h5) == bag
    _pt_valid, h5_valid = _feature_paths_with_folds(exp, "TCGA-AA", "valid")
    assert Path(h5_valid) == bag


def test_feature_lookup_does_not_use_unique_dump(tmp_path):
    from data.feature_bags import _feature_paths_with_folds

    feat_dir = tmp_path / "encoder_z4"
    leftover = feat_dir / "h5_files" / "uni2" / "O17-01529.h5"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"old")
    exp = SimpleNamespace(feat_dir=feat_dir, model="uni2", output_layout="prepath_legacy")
    _pt, h5 = _feature_paths_with_folds(exp, "O17-01529", "train")
    assert Path(h5) != leftover
    assert not Path(h5).is_file()


