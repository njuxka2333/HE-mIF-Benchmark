"""Tests for TCGA-COAD prognosis cross-validation split generation and wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PREPATH_ROOT = Path(__file__).resolve().parent.parent


def test_build_tcga_coad_cv_splits_counts(tmp_path):
    from scripts.build_tcga_coad_cv_splits import build_cv_splits

    base = Path("/data3/houjunlin/fxx/HE-mIF-Benchmark/splits/prognosis_os/tcga_coad_split.csv")
    if not base.is_file():
        return

    written = build_cv_splits(
        base,
        n_folds=5,
        seed=42,
        output_dir=tmp_path,
        output_pattern="cv_outer_fold{fold}.csv",
    )
    assert len(written) == 5
    all_test_patients: set[str] = set()
    for path in written:
        frame = pd.read_csv(path)
        assert set(frame["split"].unique()) <= {"train", "valid", "test"}
        patient_counts = frame.drop_duplicates("case_id").groupby("split")["case_id"].nunique()
        assert patient_counts.get("train", 0) >= 280
        assert patient_counts.get("valid", 0) >= 70
        assert 85 <= patient_counts.get("test", 0) <= 91
        assert patient_counts.sum() == frame["case_id"].nunique()
        all_test_patients.update(
            frame.loc[frame["split"] == "test", "case_id"].astype(str).unique().tolist()
        )
    assert len(all_test_patients) == frame["case_id"].nunique()


def test_resolve_prognosis_cv_split_csv():
    from training.config import load_eval_config
    from training.train import _cross_validation_folds, _resolve_cv_split_csv

    cfg_path = (
        PREPATH_ROOT
        / "configs/downstream/prognosis/tcga_coad/encoder/gigatime.yaml"
    )
    if not cfg_path.is_file():
        return

    cfg = load_eval_config(cfg_path)
    folds = _cross_validation_folds(cfg)
    assert folds == [0, 1, 2, 3, 4]
    resolved = _resolve_cv_split_csv(cfg, 0)
    assert resolved.endswith("tcga_coad_split_shazam_fold0.csv")
    assert Path(resolved).is_file()
    assert cfg.data.get("valid_split") == "valid"
    assert cfg.data.get("test_split") == "test"


def test_run_survival_uses_shared_cv_orchestrator(monkeypatch, tmp_path):
    from training.config import EvalConfig
    from training import offline_train

    seen_folds: list[int] = []

    def fake_once(run_cfg: EvalConfig) -> dict[str, float]:
        fold_name = run_cfg.output_dir.name
        seen_folds.append(int(fold_name.split("_")[-1]))
        return {"seed": 42.0, "valid_c_index": 0.5}

    monkeypatch.setattr(offline_train, "_run_survival_once", fake_once)
    monkeypatch.setattr(offline_train._mod, "_run_survival_once", fake_once)
    cfg = EvalConfig(
        id="cv-test",
        task="prognosis_survival",
        dataset="survival_features",
        head="survival_attention_mil",
        output_dir=tmp_path,
        data={
            "split_csv": "/data3/houjunlin/fxx/HE-mIF-Benchmark/splits/prognosis_os/tcga_coad_split.csv",
            "cross_validation": {
                "enabled": True,
                "folds": [0, 1],
                "split_csv_pattern": "tcga_coad_split_shazam_fold{fold}.csv",
                "split_csv_dir": "/data3/houjunlin/fxx/HE-mIF-Benchmark/splits/prognosis_os",
            },
        },
        head_config={},
        training={"seed": 42},
        metrics={},
        survival={},
    )
    offline_train.run_survival(cfg)
    assert seen_folds == [0, 1]
    assert (tmp_path / "cv_summary.json").is_file()
