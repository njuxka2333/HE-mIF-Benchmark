"""Train-only PCA projection for classification and survival MIL bags.

Fit on the current config's training split, then apply to every split.
Used by the encoder-width control (output-layer channel count).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

PCA_MAX_PATCHES = 80_000
PCA_PER_BAG = 256


def _features_from_item(item: Any) -> torch.Tensor | None:
    if torch.is_tensor(item):
        return item
    if isinstance(item, dict) and "features" in item:
        feats = item["features"]
        return feats if torch.is_tensor(feats) else None
    if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
        return item[0]
    return None


def _set_features(item: Any, feats: torch.Tensor) -> Any:
    if torch.is_tensor(item):
        return feats
    if isinstance(item, dict):
        out = dict(item)
        out["features"] = feats
        return out
    if isinstance(item, tuple):
        return (feats,) + item[1:]
    if isinstance(item, list):
        out = list(item)
        out[0] = feats
        return out
    return item


def _project_tensor(feats: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    x = feats.float()
    flat = x.reshape(-1, x.shape[-1])
    z = (flat - mean.to(dtype=flat.dtype, device=flat.device)) @ components.to(
        dtype=flat.dtype, device=flat.device
    ).transpose(0, 1)
    return z.reshape(*x.shape[:-1], components.shape[0]).to(dtype=feats.dtype)


def _call_getitem(dataset, idx: int, getitem=None):
    """Call a bound ``__getitem__`` or an unbound ``Type.__getitem__(self, idx)``."""
    if getitem is None:
        return dataset[idx]
    try:
        import inspect

        n_params = len(inspect.signature(getitem).parameters)
    except (TypeError, ValueError):
        n_params = 2
    if n_params == 1:
        return getitem(idx)
    return getitem(dataset, idx)


def fit_train_pca(
    dataset,
    n_components: int,
    *,
    max_patches: int = PCA_MAX_PATCHES,
    per_bag: int = PCA_PER_BAG,
    seed: int = 42,
    getitem=None,
) -> dict[str, torch.Tensor] | None:
    """Fit PCA on training bag patches. Returns None if width is already <= n_components."""
    rng = np.random.RandomState(int(seed))
    chunks: list[np.ndarray] = []
    n_kept = 0
    dim = None
    for idx in range(len(dataset)):
        feats = _features_from_item(_call_getitem(dataset, idx, getitem))
        if feats is None or feats.ndim < 2:
            continue
        arr = feats.detach().float().cpu().numpy()
        dim = int(arr.shape[-1])
        if dim <= int(n_components):
            return None
        rows = arr.reshape(-1, dim)
        if len(rows) > per_bag:
            pick = rng.choice(len(rows), size=per_bag, replace=False)
            rows = rows[pick]
        take = min(len(rows), max_patches - n_kept)
        if take <= 0:
            break
        chunks.append(np.ascontiguousarray(rows[:take]))
        n_kept += take
        if n_kept >= max_patches:
            break
    if not chunks or dim is None:
        raise ValueError("PCA fit found no training patches")
    stacked = np.concatenate(chunks, axis=0)
    k = min(int(n_components), stacked.shape[0], stacked.shape[1])
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="randomized", random_state=int(seed))
    pca.fit(stacked)
    return {
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),
        "components": torch.tensor(pca.components_, dtype=torch.float32),
        "n_components": int(k),
        "input_dim": int(dim),
        "n_fit_patches": int(stacked.shape[0]),
    }


def attach_pca(dataset, pca: dict[str, torch.Tensor] | None) -> None:
    dataset._pca_mean = None if pca is None else pca["mean"]
    dataset._pca_components = None if pca is None else pca["components"]


def apply_pca_item(dataset, item: Any) -> Any:
    mean = getattr(dataset, "_pca_mean", None)
    components = getattr(dataset, "_pca_components", None)
    if mean is None or components is None:
        return item
    feats = _features_from_item(item)
    if feats is None:
        return item
    return _set_features(item, _project_tensor(feats, mean, components))


def _pca_dim(cfg) -> int:
    return int((cfg.training or {}).get("pca_dim") or (cfg.data or {}).get("pca_dim") or 0)


def _log_pca(pca: dict[str, torch.Tensor] | None, pca_dim: int) -> None:
    if pca is not None:
        print(
            f"PCA({pca['n_components']}) fit on {pca['n_fit_patches']} patches "
            f"({pca['input_dim']} -> {pca['n_components']})",
            flush=True,
        )
    else:
        print(f"PCA skipped: feature width already <= {pca_dim}", flush=True)


def _iter_train_modules(module: Any | None):
    import sys

    seen: set[int] = set()
    for candidate in (
        module,
        sys.modules.get("training.offline_train"),
        sys.modules.get("training.train"),
        sys.modules.get("_bytecode_offline_train"),
        sys.modules.get("_bytecode_train_helpers"),
    ):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate


def _pca_payload(pca: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    return {
        "n_components": int(pca["n_components"]),
        "input_dim": int(pca["input_dim"]),
        "n_fit_patches": int(pca["n_fit_patches"]),
        "whiten": False,
        "seed": int(seed),
        "fit_partition": "train",
    }


def _dump_pca_json(cfg, pca: dict[str, torch.Tensor] | None, seed: int) -> None:
    if pca is None:
        return
    out_dir = getattr(cfg, "output_dir", None)
    if out_dir is None:
        return
    from pathlib import Path
    import json

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "pca.json").write_text(json.dumps(_pca_payload(pca, seed)), encoding="utf-8")


def install_pca_hook(module: Any | None = None) -> None:
    """Apply ``training.pca_dim`` after zscore stats exist, then to later splits."""
    for target in _iter_train_modules(module):
        _wrap_make_dataset(target)
        _wrap_prepare(target)
        _wrap_survival_prepare(target)


def _wrap_make_dataset(target) -> None:
    orig = getattr(target, "_make_dataset", None)
    if orig is None or getattr(orig, "_pca_wrapped", False):
        return

    def _make_dataset_with_pca(cfg, split, *args, **kwargs):
        dataset = orig(cfg, split, *args, **kwargs)
        if _pca_dim(cfg) <= 0:
            return dataset
        attach_pca(dataset, getattr(cfg, "_pca_state", None))
        return dataset

    _make_dataset_with_pca._pca_wrapped = True  # type: ignore[attr-defined]
    target._make_dataset = _make_dataset_with_pca


def _wrap_prepare(target) -> None:
    orig = getattr(target, "_prepare_classification_datasets", None)
    if orig is None or getattr(orig, "_pca_wrapped", False):
        return

    def _prepare_with_pca(cfg, *args, **kwargs):
        pca_dim = _pca_dim(cfg)
        if pca_dim > 0:
            cfg._pca_state = None
        train_ds, valid_ds, audit = orig(cfg, *args, **kwargs)
        if pca_dim <= 0:
            return train_ds, valid_ds, audit
        seed = int((cfg.training or {}).get("seed") or 42)
        pca = fit_train_pca(train_ds, pca_dim, seed=seed)
        cfg._pca_state = pca
        _log_pca(pca, pca_dim)
        _dump_pca_json(cfg, pca, seed)
        attach_pca(train_ds, pca)
        if valid_ds is not None:
            attach_pca(valid_ds, pca)
        return train_ds, valid_ds, audit

    _prepare_with_pca._pca_wrapped = True  # type: ignore[attr-defined]
    target._prepare_classification_datasets = _prepare_with_pca


def _wrap_survival_prepare(target) -> None:
    orig = getattr(target, "_prepare_survival_datasets", None)
    if orig is None or getattr(orig, "_pca_wrapped", False):
        return

    def _prepare_survival_with_pca(cfg, *args, **kwargs):
        pca_dim = _pca_dim(cfg)
        if pca_dim > 0:
            cfg._pca_state = None
        train_ds, valid_ds, audit = orig(cfg, *args, **kwargs)
        if pca_dim <= 0:
            return train_ds, valid_ds, audit
        seed = int((cfg.training or {}).get("seed") or 42)
        pca = fit_train_pca(train_ds, pca_dim, seed=seed)
        cfg._pca_state = pca
        _log_pca(pca, pca_dim)
        _dump_pca_json(cfg, pca, seed)
        attach_pca(train_ds, pca)
        if valid_ds is not None:
            attach_pca(valid_ds, pca)
        return train_ds, valid_ds, audit

    _prepare_survival_with_pca._pca_wrapped = True  # type: ignore[attr-defined]
    target._prepare_survival_datasets = _prepare_survival_with_pca


from data.feature_bags import ClassificationFeatureDataset, SurvivalFeatureDataset  # noqa: E402

_ORIG_GETITEM = ClassificationFeatureDataset.__getitem__
if not getattr(_ORIG_GETITEM, "_pca_wrapped", False):

    def _getitem_with_pca(self, idx: int):
        return apply_pca_item(self, _ORIG_GETITEM(self, idx))

    _getitem_with_pca._pca_wrapped = True  # type: ignore[attr-defined]
    ClassificationFeatureDataset.__getitem__ = _getitem_with_pca

_ORIG_SURVIVAL_GETITEM = SurvivalFeatureDataset.__getitem__
if not getattr(_ORIG_SURVIVAL_GETITEM, "_pca_wrapped", False):

    def _survival_getitem_with_pca(self, idx: int):
        return apply_pca_item(self, _ORIG_SURVIVAL_GETITEM(self, idx))

    _survival_getitem_with_pca._pca_wrapped = True  # type: ignore[attr-defined]
    SurvivalFeatureDataset.__getitem__ = _survival_getitem_with_pca
