"""Extracted feature-bag datasets (linear probe, survival).

Lives in ``data/`` with the other dataset IO. Archived bytecode still imports
``training.datasets``; that name is aliased to this module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True

from training._bytecode_loader import install_bytecode_shim

install_bytecode_shim(
    "datasets",
    Path(__file__).resolve().parents[1] / "training",
    globals(),
    min_bytecode_size=10_000,
)

# Official bag layouts (configs/extraction/protocol.yaml):
#   linear_probe:
#     {feat_dir}/{train,valid,test}/h5_files/{model}/{slide}.h5
#   WSI CV (TCGA):
#     {feat_dir}/fold_{k}/{train,valid,test}/h5_files/{model}/{slide}.h5
_bc = sys.modules.get("_bytecode_datasets")
if _bc is not None and hasattr(_bc, "_feature_paths"):

    def _pt_h5(split_dir: Path, model: str, sample_id: str) -> tuple[Path, Path]:
        return (
            split_dir / "pt_files" / model / f"{sample_id}.pt",
            split_dir / "h5_files" / model / f"{sample_id}.h5",
        )

    def _sample_id_aliases(sample_id) -> list[str]:
        raw = str(sample_id)
        aliases = [raw]
        if raw.endswith("_HE"):
            stem = raw[:-3]
            aliases.append(stem)
            aliases.append(stem.lstrip("0") or "0")
        elif raw.isdigit():
            aliases.append(raw.zfill(3))
            aliases.append(f"{raw.zfill(3)}_HE")
        seen = set()
        out = []
        for name in aliases:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _feature_paths_protocol(exp, sample_id, split):
        model = str(getattr(exp, "model"))
        split_name = str(split).lower()
        split_name = {"training": "train", "val": "valid"}.get(split_name, split_name)
        feat_dir = Path(getattr(exp, "feat_dir"))

        split_dirs: list[Path] = []
        if split_name:
            split_dirs.append(feat_dir / split_name)
        if split_name not in {"train", "test"}:
            split_dirs.append(feat_dir / "test")

        seen_dirs: set[Path] = set()
        ordered_dirs: list[Path] = []
        for path in split_dirs:
            resolved = path.resolve() if path.exists() else path
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            ordered_dirs.append(path)

        fold_dirs = sorted(feat_dir.glob("fold_*"))

        for sid in _sample_id_aliases(sample_id):
            for split_dir in ordered_dirs:
                pt_path, h5_path = _pt_h5(split_dir, model, sid)
                if h5_path.is_file() or pt_path.is_file():
                    return (pt_path, h5_path)
            for fold_dir in fold_dirs:
                pt_path, h5_path = _pt_h5(fold_dir / split_name, model, sid)
                if h5_path.is_file() or pt_path.is_file():
                    return (pt_path, h5_path)
            if split_name == "valid":
                for fold_dir in fold_dirs:
                    pt_path, h5_path = _pt_h5(fold_dir / "test", model, sid)
                    if h5_path.is_file() or pt_path.is_file():
                        return (pt_path, h5_path)
        return _pt_h5(feat_dir / (split_name or "train"), model, str(sample_id))

    _bc._feature_paths = _feature_paths_protocol
    globals()["_feature_paths"] = _feature_paths_protocol
    globals()["_feature_paths_with_folds"] = _feature_paths_protocol

# Survival training re-reads the same patch-h5 bag every epoch. Cache float16
# on CPU after the first load. Opt out with HEMIF_SURVIVAL_BAG_CACHE=0.
_SURVIVAL_BAG_CACHE_ENABLED = os.environ.get("HEMIF_SURVIVAL_BAG_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_ORIG_SURVIVAL_GETITEM = SurvivalFeatureDataset.__getitem__


def _clone_cached_survival_item(item: dict):
    cloned = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            tensor = value.float() if key == "features" else value
            cloned[key] = tensor.clone()
        else:
            cloned[key] = value
    return cloned


def _survival_getitem_cached(self, idx: int):
    if not _SURVIVAL_BAG_CACHE_ENABLED:
        return _ORIG_SURVIVAL_GETITEM(self, idx)
    cache = self.__dict__.get("_bag_cache")
    if cache is None:
        cache = {}
        self._bag_cache = cache
    cached = cache.get(idx)
    if cached is not None:
        return _clone_cached_survival_item(cached)
    item = _ORIG_SURVIVAL_GETITEM(self, idx)
    stored = {}
    for key, value in item.items():
        if key == "features" and torch.is_tensor(value):
            stored[key] = value.detach().to(dtype=torch.float16, device="cpu").contiguous()
        elif torch.is_tensor(value):
            stored[key] = value.detach().to("cpu")
        else:
            stored[key] = value
    cache[idx] = stored
    return item


SurvivalFeatureDataset.__getitem__ = _survival_getitem_cached
