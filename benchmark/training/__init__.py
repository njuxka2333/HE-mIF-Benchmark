"""YAML-driven training entry. Heads, metrics, losses, and data live next door.

Layout::

    train.py / pipeline.py / postprocess.py / test.py   official entry
    config.py / augmentations.py / distributed.py
    offline_train.py / online_segmentation_train.py     archived loops
    linear_probe_adamw.py / pca_features.py

Sibling packages (not nested here)::

    heads/     MIL / UNETR / cellseg / mIF decoders
    metrics/   classification, survival, segmentation, vstain, clinical
    losses/    train losses
    mif/       generated-mIF trainers + Cell-niches scorer
    data/      WSI/FOV loaders + feature bags

Archived bytecode still imports ``training.heads``, ``training.datasets``, etc.
Those names are remapped below.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_ALIASES = {
    "training.heads": "heads",
    "training.heads.unetr": "heads.unetr",
    "training.heads.cellseg": "heads.cellseg",
    "training.heads.upsampler": "heads.upsampler",
    "training.heads.vit_pyramid": "heads.vit_pyramid",
    "training.heads.bilinear": "heads.bilinear",
    "training.heads.interfaces": "heads.interfaces",
    "training.heads.mif_unet": "heads.mif_unet",
    "training.heads.mif_feature": "heads.mif_feature",
    "training.metrics": "metrics",
    "training.metrics.vstain": "metrics.vstain",
    "training.metrics.segmentation": "metrics.segmentation",
    "training.metrics.classification": "metrics.classification",
    "training.metrics.clinical": "metrics.clinical",
    "training.losses": "losses",
    "training.datasets": "data.feature_bags",
    "training.vstain": "mif.score",
    "training.vstain.score": "mif.score",
    "training.vstain.cell_niches": "mif.score",
    "training.vstain.mipheivit": "mif.score",
    "training.vstain.protocol": "metrics.vstain",
    "training.vstain.pixel": "metrics.vstain",
    "training.mif": "mif",
    "training.mif.resample": "mif.resample",
    "training.mif.train_unet": "mif.train_unet",
    "training.mif.train_feature": "mif.train_feature",
    "training.mif.score": "mif.score",
    "training.mif.common": "mif.common",
    "training.unetr": "heads.unetr",
    "training.cellseg_unetr": "heads.cellseg",
    "training.cellseg_encoder_upsampler": "heads.upsampler",
    "training.vit_feature_pyramid": "heads.vit_pyramid",
    "training.vit_bilinear_pyramid": "heads.bilinear",
    "training.unetr_interfaces": "heads.interfaces",
    "training.mif_unet": "heads.mif_unet",
    "training.mif_feature_unetr": "heads.mif_feature",
    "training.pfmb_unetr": "heads.unetr",
    "training.mif_resample": "mif.resample",
    "training.mif_unet_train": "mif.train_unet",
    "training.mif_feature_unetr_train": "mif.train_feature",
    "training.mif_seg_common": "mif.common",
    "training.cell_niches_vstain": "mif.score",
    "training.cell_niches_mipheivit_vstain": "mif.score",
    "training.shared_vstain_protocol": "metrics.vstain",
    "training.miphei_pixel_protocol": "metrics.vstain",
    "training.vstain_metrics": "metrics.vstain",
    "training.vstain_significance": "metrics.vstain",
    "training.classification_significance": "metrics.classification",
    "training.clinical_stats": "metrics.clinical",
    "training.run": "training.train",
    "training.linear_probe_train": "training.linear_probe_adamw",
    "models.config.checkpoints": "utils.checkpoints",
    "models.config.registry": "models.registry",
    "models.config.schema": "models.schema",
    "models.config.runtime": "models.registry",
    "models.config.resolution": "models.registry",
}


class _AliasFinder(MetaPathFinder):
    """Resolve pre-split module names to their new package homes."""

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        real = _ALIASES.get(fullname)
        if real is None:
            return None
        return ModuleSpec(fullname, _AliasLoader(fullname, real))


class _AliasLoader(Loader):
    def __init__(self, alias: str, real: str) -> None:
        self.alias = alias
        self.real = real

    def create_module(self, spec):  # noqa: ARG002
        if self.alias in sys.modules:
            return sys.modules[self.alias]
        if self.real in sys.modules:
            sys.modules[self.alias] = sys.modules[self.real]
            return sys.modules[self.real]
        actual = importlib.import_module(self.real)
        sys.modules[self.alias] = actual
        return actual

    def exec_module(self, module):  # noqa: ARG002
        return None


if not any(type(finder).__name__ == "_AliasFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
