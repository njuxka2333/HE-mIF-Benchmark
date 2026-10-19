"""Example-tile helpers used by the virtual-staining pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmark", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mif.example_tile import (  # noqa: E402
    EXAMPLE_PATCH_ID,
    MODELS,
    collect_cd3_ranking,
    maybe_save_scored_cd3,
    write_cd3_figure,
)
from mif.preview import match_histogram  # noqa: E402


def test_maybe_save_scored_cd3_only_official_tile(tmp_path, monkeypatch):
    from mif import example_tile as mod

    monkeypatch.setattr(mod, "maps_dir", lambda: tmp_path)
    pred = np.zeros((3, 8, 8), dtype=np.float32)
    pred[1] = 1.0

    class M:
        canonical = "CD3"
        model_index = 1

    maybe_save_scored_cd3("rosie", "other_patch", pred, [M()])
    assert not list(tmp_path.glob("*.npy"))
    maybe_save_scored_cd3("rosie", EXAMPLE_PATCH_ID, pred, [M()])
    saved = np.load(tmp_path / "rosie_cd3.npy")
    assert saved.shape == (8, 8)
    assert float(saved.max()) == 1.0


def test_collect_cd3_ranking_joins_models(tmp_path, monkeypatch):
    from mif import example_tile as mod

    def fake_csv(model: str):
        path = tmp_path / model / "per_patch_spearman.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "patch_id": [EXAMPLE_PATCH_ID, "other"],
                "CD3_spearman": [0.3 if model == "rosie" else 0.1, 0.0],
                "CD3_auprc": [0.2, 0.0],
            }
        ).to_csv(path, index=False)
        return path

    monkeypatch.setattr(mod, "_per_patch_csv", fake_csv)
    ranking = collect_cd3_ranking(("rosie", "hemit"))
    row = ranking.loc[ranking["patch_id"] == EXAMPLE_PATCH_ID].iloc[0]
    assert abs(float(row["rosie_spearman"]) - 0.3) < 1e-9
    assert abs(float(row["hemit_spearman"]) - 0.1) < 1e-9
    assert abs(float(row["mean_spearman"]) - 0.2) < 1e-9


def test_write_cd3_figure_makes_panel(tmp_path):
    rng = np.random.default_rng(0)
    he = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    gt = rng.random((32, 32)).astype(np.float32)
    maps = {name: rng.random((32, 32)).astype(np.float32) for name in MODELS}
    ranking = pd.DataFrame(
        {
            "patch_id": [EXAMPLE_PATCH_ID],
            **{f"{m}_spearman": [0.1] for m in MODELS},
            **{f"{m}_auprc": [0.2] for m in MODELS},
            "mean_spearman": [0.1],
            "min_spearman": [0.1],
            "mean_auprc": [0.2],
        }
    )
    rec = {"entity": "LUAD", "fov_id": "x", "tissue_frac": 1.0, "he_relpath": "he.png", "mif_relpath": "mif.tif"}
    write_cd3_figure(he, gt, maps, ranking.iloc[0], rec, ranking, out_dir=tmp_path)
    assert (tmp_path / "panel_cd3.png").is_file()
    assert (tmp_path / "panel_gt_cd3.png").is_file()
    assert (tmp_path / "pick.json").is_file()


def test_pipeline_hooks_example_tiles_for_vstain_cellseg_and_tls():
    import inspect

    from training import pipeline

    src = inspect.getsource(pipeline.run_task)
    assert "render_task_example_tiles" in src
    assert "virtual_staining" in src
    assert "cell_semantic_segmentation" in src
    assert "tls_region_segmentation" in src


def test_cellseg_example_tile_forced_from_pipeline():
    import inspect

    from training import example_tiles

    src = inspect.getsource(example_tiles.render_task_example_tiles)
    assert "force=True" in src
    assert "render_tls_example_tile" in src


def test_per_patch_csv_prefers_repo_evaluation(tmp_path, monkeypatch):
    from mif import example_tile as mod

    repo_eval = tmp_path / "repo_eval"
    sibling = tmp_path / "sibling"
    frozen = tmp_path / "frozen"
    fresh = repo_eval / "virtual_staining/cell_niches/output/rosie/per_patch_spearman.csv"
    stale = sibling / "virtual_staining/cell_niches/cache_4um/rosie/per_patch_spearman.csv"
    old = frozen / "rosie/per_patch_spearman.csv"
    for path, value in ((fresh, 0.9), (stale, 0.1), (old, 0.0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"patch_id": [EXAMPLE_PATCH_ID], "CD3_spearman": [value]}).to_csv(path, index=False)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "evaluation_root", lambda: sibling)
    monkeypatch.setattr(mod, "_eval_roots", lambda: (repo_eval, sibling))
    csv = mod._per_patch_csv("rosie")
    assert csv == fresh


def test_standalone_example_tile_scripts_removed():
    scripts = ROOT / "benchmark" / "scripts"
    assert not (scripts / "render_vstain_example_tile.py").exists()
    assert not (scripts / "render_pathocell_example_tile.py").exists()
    assert not (scripts / "render_tls_example_tile.py").exists()


def test_hist_match_still_rank_preserving():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(16, 16))
    ref = rng.random((16, 16))
    out = match_histogram(src, ref)
    from metrics.vstain import spearman_r

    assert abs(spearman_r(src, out) - 1.0) < 1e-12
