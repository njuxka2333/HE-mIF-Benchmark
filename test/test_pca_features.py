"""PCA fit is train-only and projects bags to a fixed width."""

from __future__ import annotations

import numpy as np
import torch

from training.pca_features import attach_pca, fit_train_pca, apply_pca_item


class _BagDS:
    def __init__(self, bags):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return {"features": self.bags[idx], "label": 0}


def test_pca_fit_on_train_projects_to_requested_dim():
    rng = np.random.RandomState(0)
    bags = [torch.tensor(rng.randn(20, 64), dtype=torch.float32) for _ in range(8)]
    ds = _BagDS(bags)
    pca = fit_train_pca(ds, 16, max_patches=200, per_bag=20, seed=0, getitem=ds.__getitem__)
    assert pca is not None
    assert pca["components"].shape == (16, 64)
    attach_pca(ds, pca)
    out = apply_pca_item(ds, ds[0])
    assert out["features"].shape == (20, 16)


def test_pca_skips_when_already_narrow():
    bags = [torch.randn(10, 8) for _ in range(3)]
    ds = _BagDS(bags)
    assert fit_train_pca(ds, 16, getitem=ds.__getitem__) is None


def test_pca_fit_accepts_unbound_getitem():
    bags = [torch.randn(12, 32) for _ in range(4)]
    ds = _BagDS(bags)
    pca = fit_train_pca(ds, 8, getitem=_BagDS.__getitem__)
    assert pca is not None
    assert pca["components"].shape == (8, 32)


def test_prepare_hook_fits_after_dataset_construction():
    """zscore stats are attached during prepare; PCA must not load bags earlier."""
    from types import SimpleNamespace
    from training.pca_features import install_pca_hook

    bags = [torch.randn(10, 32) for _ in range(5)]
    train_ds = _BagDS(bags)
    valid_ds = _BagDS(bags[:2])

    class _Mod:
        def _make_dataset(self, cfg, split, *args, **kwargs):
            return train_ds if split == "train" else valid_ds

        def _prepare_classification_datasets(self, cfg):
            return train_ds, valid_ds, {}

    mod = _Mod()
    install_pca_hook(mod)
    cfg = SimpleNamespace(training={"pca_dim": 16, "seed": 0}, data={"train_split": "train"})
    out_train, out_valid, _ = mod._prepare_classification_datasets(cfg)
    assert cfg._pca_state is not None
    assert cfg._pca_state["components"].shape[1] == 32
    assert out_train._pca_components.shape == (16, 32)
    assert out_valid._pca_components.shape == (16, 32)
    projected = apply_pca_item(out_train, out_train[0])
    assert projected["features"].shape[-1] == 16


def test_survival_prepare_hook_fits_after_dataset_construction(tmp_path):
    from types import SimpleNamespace
    from training.pca_features import install_pca_hook

    bags = [torch.randn(10, 32) for _ in range(5)]
    train_ds = _BagDS(bags)
    valid_ds = _BagDS(bags[:2])

    class _Mod:
        def _make_dataset(self, cfg, split, *args, **kwargs):
            return train_ds if split == "train" else valid_ds

        def _prepare_survival_datasets(self, cfg):
            return train_ds, valid_ds, {}

    mod = _Mod()
    install_pca_hook(mod)
    cfg = SimpleNamespace(
        training={"pca_dim": 8, "seed": 0},
        data={"train_split": "train"},
        output_dir=tmp_path,
    )
    out_train, out_valid, _ = mod._prepare_survival_datasets(cfg)
    assert cfg._pca_state is not None
    assert cfg._pca_state["components"].shape == (8, 32)
    assert out_train._pca_components.shape == (8, 32)
    assert out_valid._pca_components.shape == (8, 32)
    assert (tmp_path / "pca.json").is_file()


def test_cell_pca_is_train_only_and_unwhitened():
    from training.linear_probe_adamw import _apply_train_pca

    rng = np.random.RandomState(42)
    train = torch.tensor(rng.randn(800, 32), dtype=torch.float32)
    valid = torch.tensor(rng.randn(100, 32), dtype=torch.float32)
    test = torch.tensor(rng.randn(100, 32), dtype=torch.float32)
    proj_train, extras, payload = _apply_train_pca(
        train, [valid, test], n_components=8, seed=42, max_cells=200
    )
    assert payload is not None
    assert payload["n_fit_cells"] == 200
    assert payload["whiten"] is False
    assert proj_train.shape == (800, 8)
    assert extras[0].shape == (100, 8)
    again, _, payload2 = _apply_train_pca(
        train, [valid, test], n_components=8, seed=42, max_cells=200
    )
    assert torch.allclose(proj_train, again)
    assert payload2["n_fit_cells"] == 200


def test_cell_pca_skips_when_already_narrow():
    from training.linear_probe_adamw import _apply_train_pca

    train = torch.randn(50, 8)
    out, extras, payload = _apply_train_pca(
        train, [None], n_components=16, seed=0, max_cells=50
    )
    assert payload is None
    assert out.shape == train.shape

