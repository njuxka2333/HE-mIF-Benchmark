"""Pipeline tests: core extract_features helpers, CLI entry, and WSI path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import (
    EXPERIMENTS_DIR,
    OUTPUT_ROOT,
    WSI_SMOKE_CONFIG,
    MockWsiHandle,
    apply_test_dataset_config,
    assert_output_contents,
    assert_output_layout,
    ensure_model_weights,
)
from data.loader import load_split_samples
from utils.experiment_config import load_experiment_config


class _IdentityExtractor(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


def test_plan_split_runs_defaults_to_three_subfolders():
    import feature_extraction.extract_features_fp as extract_features

    assert extract_features.plan_split_runs(None, None) == [
        ("train", "train"),
        ("valid", "valid"),
        ("test", "test"),
    ]
    assert extract_features.plan_split_runs("test", None) == [("test", None)]
    assert extract_features.plan_split_runs(None, "train") == [("train", None)]
    assert extract_features.plan_split_runs("all", None) == [(None, None)]
    assert extract_features.plan_split_runs("val", None) == [("val", None)]


def test_run_loader_concatenates_batches():
    import feature_extraction.extract_features_fp as extract_features

    class _FakeLoader:
        def __iter__(self):
            yield (
                torch.randn(2, 3, 8, 8),
                np.array([[0, 0], [1, 1]], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
            )
            yield (
                torch.randn(1, 3, 8, 8),
                np.array([[2, 2]], dtype=np.int32),
                np.array([2], dtype=np.int32),
            )

    model = _IdentityExtractor()
    features, out_coords, out_masks = extract_features._run_loader(
        model, _FakeLoader(), torch.device("cpu")
    )

    assert features.shape[0] == 3
    assert out_coords.shape == (3, 2)
    assert out_masks is not None and out_masks.shape[0] == 3


def test_save_outputs_writes_coords_and_masks(tmp_path, sync_save):
    import feature_extraction.extract_features_fp as extract_features

    exp = load_experiment_config(EXPERIMENTS_DIR / "abmil_tcga_coad_mipheivit_20x_s256.yaml")
    exp.feat_dir = tmp_path
    exp.save_masks = True
    features = torch.randn(2, 16)
    coords = np.array([[0, 0], [8, 8]], dtype=np.int32)
    masks = np.array([[[1]], [[2]]], dtype=np.int32)

    extract_features._save_outputs(exp, "tile_a", features, coords, masks, exp.model)

    assert_output_layout(
        exp.feat_dir,
        exp.model,
        "tile_a",
        save_masks=True,
        feature_storage=exp.feature_storage,
    )
    assert_output_contents(
        exp.feat_dir,
        exp.model,
        "tile_a",
        save_masks=True,
        feature_storage=exp.feature_storage,
    )


def test_save_outputs_patch_h5_mode(tmp_path, sync_save):
    import feature_extraction.extract_features_fp as extract_features
    import h5py

    from utils.file_utils import is_patch_bag_h5

    exp = load_experiment_config(EXPERIMENTS_DIR / "abmil_tcga_coad_rosie_40x_s128.yaml")
    exp.feat_dir = tmp_path
    exp.save_masks = True
    exp.feature_storage = "patch_h5"
    features = {"z1": torch.randn(2, 4, 2, 2)}
    coords = np.array([[0, 0], [8, 8]], dtype=np.int32)
    masks = np.array([[[1, 0], [0, 1]], [[2, 1], [1, 2]]], dtype=np.int32)

    extract_features._save_outputs(exp, "tile_a", features, coords, masks, exp.model)

    h5_path = tmp_path / "h5_files" / exp.model / "tile_a.h5"
    pt_path = tmp_path / "pt_files" / exp.model / "tile_a.pt"
    assert h5_path.is_file()
    assert not pt_path.exists()
    assert is_patch_bag_h5(h5_path)
    with h5py.File(h5_path, "r") as handle:
        assert "features" in handle
        assert handle["features"]["z1"].shape[0] == 2
        assert "masks" in handle


@pytest.mark.integration
@pytest.mark.requires_weights
def test_wsi_extraction_smoke(device, sync_save, monkeypatch):
    import feature_extraction.extract_features_fp as extract_features

    exp = load_experiment_config(EXPERIMENTS_DIR / WSI_SMOKE_CONFIG)
    ensure_model_weights(exp.model)
    apply_test_dataset_config(exp)
    exp.feat_dir = OUTPUT_ROOT / exp.id

    cfg = exp.dataset_config
    samples = load_split_samples(cfg, max_samples=1)
    sample = samples[0]
    h5_path = cfg.coords_dir() / f"{sample.sample_id}.h5"
    assert h5_path.is_file()

    monkeypatch.setattr(extract_features, "get_wsi_handle", lambda _path: MockWsiHandle())
    monkeypatch.setattr(
        "feature_extraction.extract_features_fp.DataLoader",
        lambda *args, **kwargs: DataLoader(*args, **{**kwargs, "num_workers": 0}),
    )

    args = SimpleNamespace(target_patch_size=-1, batch_size=1)
    try:
        model, transform = extract_features._build_model(exp, device)
        extract_features.extract_wsi_slide(
            exp, sample.sample_id, MockWsiHandle(), h5_path, model, transform, device, args
        )
    except ImportError as exc:
        pytest.skip(f"Missing dependency for {exp.model}: {exc}")
    except Exception as exc:
        name = type(exc).__name__
        if name in {"GatedRepoError", "HTTPStatusError", "ProxyError", "FileNotFoundError"}:
            pytest.skip(f"Cannot run WSI extraction for {exp.model}: {exc}")
        raise

    # Fixture coord bags have no mask arrays; WSI extraction skips mask_files.
    assert_output_layout(
        exp.feat_dir,
        exp.model,
        sample.sample_id,
        save_masks=False,
        feature_storage=exp.feature_storage,
    )
    assert_output_contents(
        exp.feat_dir,
        exp.model,
        sample.sample_id,
        save_masks=False,
        expect_multi_layer=True,
        feature_storage=exp.feature_storage,
    )
