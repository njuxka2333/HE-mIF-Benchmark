"""Standalone example-tile rendering helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

def test_collect_cd3_ranking_joins_models(tmp_path, monkeypatch):
    mod = pytest.importorskip("examples.virtual_staining")
    example_patch_id = mod.EXAMPLE_PATCH_ID

    def fake_csv(model: str):
        path = tmp_path / model / "per_patch_spearman.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "patch_id": [example_patch_id, "other"],
                "CD3_spearman": [0.3 if model == "rosie" else 0.1, 0.0],
                "CD3_auprc": [0.2, 0.0],
            }
        ).to_csv(path, index=False)
        return path

    monkeypatch.setattr(mod, "_per_patch_csv", fake_csv)
    ranking = mod.collect_cd3_ranking(("rosie", "hemit"))
    row = ranking.loc[ranking["patch_id"] == example_patch_id].iloc[0]
    assert abs(float(row["rosie_spearman"]) - 0.3) < 1e-9
    assert abs(float(row["hemit_spearman"]) - 0.1) < 1e-9
    assert abs(float(row["mean_spearman"]) - 0.2) < 1e-9


def test_write_cd3_figure_makes_panel(tmp_path):
    mod = pytest.importorskip("examples.virtual_staining")
    example_patch_id = mod.EXAMPLE_PATCH_ID
    rng = np.random.default_rng(0)
    he = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    gt = rng.random((32, 32)).astype(np.float32)
    maps = {name: rng.random((32, 32)).astype(np.float32) for name in mod.MODELS}
    ranking = pd.DataFrame(
        {
            "patch_id": [mod.EXAMPLE_PATCH_ID],
            **{f"{m}_spearman": [0.1] for m in mod.MODELS},
            **{f"{m}_auprc": [0.2] for m in mod.MODELS},
            "mean_spearman": [0.1],
            "min_spearman": [0.1],
            "mean_auprc": [0.2],
        }
    )
    rec = {"entity": "LUAD", "fov_id": "x", "tissue_frac": 1.0, "he_relpath": "he.png", "mif_relpath": "mif.tif"}
    mod.write_cd3_figure(he, gt, maps, ranking.iloc[0], rec, ranking, out_dir=tmp_path)
    assert (tmp_path / "panel_cd3.png").is_file()
    assert (tmp_path / "panel_gt_cd3.png").is_file()
    assert (tmp_path / "pick.json").is_file()


def test_benchmark_pipeline_does_not_render_example_tiles():
    import inspect

    from training import pipeline

    src = inspect.getsource(pipeline.run_task)
    assert "render_task_example_tiles" not in src
    assert "examples." not in src


def test_per_patch_csv_prefers_repo_evaluation(tmp_path, monkeypatch):
    mod = pytest.importorskip("examples.virtual_staining")
    example_patch_id = mod.EXAMPLE_PATCH_ID

    repo_eval = tmp_path / "repo_eval"
    sibling = tmp_path / "sibling"
    frozen = tmp_path / "frozen"
    fresh = repo_eval / "virtual_staining/cell_niches/output/rosie/per_patch_spearman.csv"
    stale = sibling / "virtual_staining/cell_niches/cache_4um/rosie/per_patch_spearman.csv"
    old = frozen / "rosie/per_patch_spearman.csv"
    for path, value in ((fresh, 0.9), (stale, 0.1), (old, 0.0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"patch_id": [example_patch_id], "CD3_spearman": [value]}).to_csv(path, index=False)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "evaluation_root", lambda: sibling)
    monkeypatch.setattr(mod, "_eval_roots", lambda: (repo_eval, sibling))
    csv = mod._per_patch_csv("rosie")
    assert csv == fresh


def test_example_tile_renderer_is_outside_benchmark():
    examples = ROOT / "examples"
    if not examples.is_dir():
        pytest.skip("Local examples are not included in the Git repository")
    assert (examples / "render_tiles.py").is_file()
    assert (examples / "pathocell_cellseg.py").is_file()
    assert (examples / "hooknet_tls.py").is_file()
    assert not (ROOT / "benchmark" / "examples").exists()
    assert not (ROOT / "benchmark" / "training" / "example_tiles.py").exists()


def test_hist_match_still_rank_preserving():
    mod = pytest.importorskip("examples.marker_visualization")
    rng = np.random.default_rng(2)
    src = rng.normal(size=(16, 16))
    ref = rng.random((16, 16))
    out = mod.match_histogram(src, ref)
    from metrics.vstain import spearman_r

    assert abs(spearman_r(src, out) - 1.0) < 1e-12
