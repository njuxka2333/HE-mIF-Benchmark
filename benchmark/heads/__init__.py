"""Head factory (MIL / UNETR / Shazam) plus decoder implementations.

Sibling of ``training/``. Archived ``heads.pyc`` is loaded into this namespace
without replacing the package in ``sys.modules``. Bytecode still imports
``training.heads``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.dont_write_bytecode = True

_TRAINING_ROOT = Path(__file__).resolve().parents[1] / "training"


def _load_heads_bytecode():
    from training._bytecode_pick import pick_pyc

    pyc = pick_pyc("heads", _TRAINING_ROOT, min_size=20000)
    spec = importlib.util.spec_from_file_location("_bytecode_heads", pyc)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {pyc}")
    mod = importlib.util.module_from_spec(spec)
    # Bytecode checks Path(__file__).stem == "heads" (not "heads.cpython-311").
    mod.__file__ = str(_TRAINING_ROOT / "heads.py")
    mod.__package__ = "training"
    sys.modules["_bytecode_heads"] = mod
    spec.loader.exec_module(mod)
    skip = {
        "__name__",
        "__file__",
        "__cached__",
        "__spec__",
        "__loader__",
        "__package__",
        "__builtins__",
        "__doc__",
    }
    return mod, {k: v for k, v in vars(mod).items() if k not in skip}


_mod, _exported = _load_heads_bytecode()
globals().update(_exported)

_ORIGINAL_RESOLVE_DECODER_MODE = resolve_decoder_mode
_ORIGINAL_BUILD_SEGMENTATION_HEAD = build_segmentation_head
_ORIGINAL_FORWARD_SEGMENTATION_HEAD = forward_segmentation_head
DECODER_MODES = frozenset(set(DECODER_MODES) | {"cellseg", "mif_unet", "mif_feature"})


def resolve_decoder_mode(head_name: str, config: dict | None = None) -> str:
    config = dict(config or {})
    mode = config.get("decoder_mode")
    if mode is not None:
        key = str(mode).lower()
        if key in {"cellseg", "mif_unet", "mif_feature"}:
            return key
    return _ORIGINAL_RESOLVE_DECODER_MODE(head_name, config)


def build_segmentation_head(
    head_name: str,
    *,
    feature_layers,
    in_channels,
    num_classes: int,
    head_config=None,
    **kwargs,
):
    head_config = dict(head_config or {})
    mode = resolve_decoder_mode(head_name, head_config)
    if mode == "cellseg":
        from heads.cellseg import CellsegUNETR

        layers = feature_layers or head_config.get("feature_layers")
        if layers is None:
            raise ValueError("cellseg decoder requires feature_layers")
        decoder_channels = head_config.get("decoder_channels", [128, 256, 512, 512])
        drop_rate = float(head_config.get("dropout", head_config.get("drop_rate", 0.0)))
        return CellsegUNETR(
            feature_layers=list(layers),
            in_channels=[int(ch) for ch in in_channels],
            num_classes=int(num_classes),
            decoder_channels=[int(ch) for ch in decoder_channels],
            drop_rate=drop_rate,
        )
    if mode == "mif_unet":
        from heads.mif_unet import build_mif_unet

        cfg = dict(head_config)
        cfg["num_classes"] = int(num_classes)
        if "in_channels" not in cfg:
            cfg["in_channels"] = _resolve_scalar_in_channels(in_channels, head_config)
        if feature_layers is not None and "feature_layers" not in cfg:
            cfg["feature_layers"] = list(feature_layers)
        return build_mif_unet(cfg)
    if mode == "mif_feature":
        from heads.mif_feature import build_mif_feature_unetr

        cfg = dict(head_config)
        cfg["num_classes"] = int(num_classes)
        if "in_channels" not in cfg:
            cfg["in_channels"] = _resolve_scalar_in_channels(in_channels, head_config)
        if feature_layers is not None and "feature_layers" not in cfg:
            cfg["feature_layers"] = list(feature_layers)
        return build_mif_feature_unetr(cfg)
    return _ORIGINAL_BUILD_SEGMENTATION_HEAD(
        head_name,
        feature_layers=feature_layers,
        in_channels=in_channels,
        num_classes=num_classes,
        head_config=head_config,
        **kwargs,
    )


def _resolve_scalar_in_channels(in_channels, head_config: dict) -> int:
    if isinstance(in_channels, int):
        return int(in_channels)
    if isinstance(in_channels, (list, tuple)) and len(in_channels) == 1:
        return int(in_channels[0])
    if "in_channels" in head_config:
        return int(head_config["in_channels"])
    raise ValueError(
        "mif_unet/mif_feature require scalar in_channels in head_config "
        f"(got in_channels={in_channels!r})"
    )


def forward_segmentation_head(model, features, masks, images=None):
    from heads.cellseg import CellsegUNETR
    from heads.mif_feature import MifFeatureUNETR
    from heads.mif_unet import MifUNet

    routed = getattr(model, "module", model)
    output_size = tuple(masks.shape[-2:])
    if isinstance(routed, CellsegUNETR):
        return model(features, image=images, output_size=output_size)
    if isinstance(routed, MifUNet):
        if images is not None and images.shape[1] == 3 and routed.generator is not None:
            return model(rgb=images, output_size=output_size)
        return model(image=images, output_size=output_size)
    if isinstance(routed, MifFeatureUNETR):
        return model(rgb=images, output_size=output_size)
    return _ORIGINAL_FORWARD_SEGMENTATION_HEAD(model, features, masks, images=images)


class ShazamSurvivalABMIL(nn.Module):
    """Shazam single-teacher ABMIL survival head (bias-free attention)."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        embed_dim: int | None = None,
        dropout: float = 0.25,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        embed = int(hidden_dim if embed_dim is None else embed_dim)
        hidden = int(hidden_dim)
        self.fc1 = nn.Linear(int(input_dim), embed)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(embed, hidden)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(float(dropout))
        self.attention = nn.Sequential(nn.Linear(hidden, 1, bias=False))
        self.classifier = nn.Linear(embed, int(n_classes))
        self.last_attention = None

    def forward(self, x_path):
        x = self.dropout1(self.relu(self.fc1(x_path)))
        a_h = self.dropout2(self.tanh(self.fc2(x)))
        a = self.attention(a_h)
        w = torch.softmax(a, dim=0)
        self.last_attention = w.detach()
        z = (w * x).sum(dim=0, keepdim=True)
        logits = self.classifier(z)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        prediction = torch.topk(logits, 1, dim=1)[1]
        return hazards, survival, prediction


HEAD_REGISTRY = dict(HEAD_REGISTRY)
HEAD_REGISTRY["shazam_survival_abmil"] = ShazamSurvivalABMIL

for _name in (
    "DECODER_MODES",
    "resolve_decoder_mode",
    "build_segmentation_head",
    "forward_segmentation_head",
    "ShazamSurvivalABMIL",
    "HEAD_REGISTRY",
):
    setattr(_mod, _name, globals()[_name])
