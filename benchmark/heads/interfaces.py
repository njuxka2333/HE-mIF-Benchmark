"""Canonical encoder-skip interfaces for dense UNETR heads.

The YAML at ``configs/model_unetr_interfaces.yaml`` is the source
of truth. Production TLS / cellseg configs must match ``get_default_interface``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

import yaml

from utils.paths import CONFIG_DIR

_INTERFACE_PATH = CONFIG_DIR / "model_unetr_interfaces.yaml"
_MODEL_ALIASES = {"hemmit": "hemit", "miphei": "mipheivit"}


@lru_cache(maxsize=1)
def load_interface_registry() -> dict[str, Any]:
    with _INTERFACE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _canonical_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def get_default_interface(model: str) -> dict[str, Any]:
    key = _canonical_model(model)
    block = load_interface_registry().get(key)
    if not block or "default" not in block:
        raise KeyError(f"No default UNETR interface registered for model '{model}'")
    return dict(block["default"])


def get_legacy_interface(model: str, legacy_name: str) -> dict[str, Any]:
    key = _canonical_model(model)
    legacy = (load_interface_registry().get(key) or {}).get("legacy") or {}
    if legacy_name not in legacy:
        raise KeyError(f"No legacy interface '{legacy_name}' for model '{model}'")
    return dict(legacy[legacy_name])


def validate_dense_config(
    model: str,
    target_features: Sequence[str],
    head_config: Mapping[str, Any],
    training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a production dense config against its canonical encoder boundary."""
    interface = get_default_interface(model)
    expected = list(interface["target_features"])
    observed = list(target_features)
    if observed != expected:
        raise ValueError(
            f"{model} dense interface requires ordered features {expected}, got {observed}"
        )

    expected_processor = interface.get("feature_processor")
    processor_cfg = head_config.get("feature_processor")
    observed_processor = (
        processor_cfg.get("name") if isinstance(processor_cfg, Mapping) else None
    )
    if observed_processor != expected_processor:
        raise ValueError(
            f"{model} dense interface {expected} requires feature_processor="
            f"{expected_processor!r}, got {observed_processor!r}"
        )

    training_config = training_config or {}
    if model.lower() in {"hemit", "hemmit"} and bool(
        training_config.get("phase1c_pool_z4", False)
    ):
        raise ValueError(
            "HEMIT dense interface ['fused_z1', 'fused_z2', 'fused_z3', 'z4'] "
            "forbids phase1c_pool_z4; z4 must remain unchanged"
        )
    return interface


def validate_dense_runtime_features(
    model: str,
    features: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Assert the canonical ordered feature names, channels, and spatial geometry."""
    interface = get_default_interface(model)
    expected_names = list(interface["target_features"])
    observed_names = list(features)
    shapes = {name: tuple(tensor.shape) for name, tensor in features.items()}
    if observed_names != expected_names:
        raise ValueError(
            f"{model} dense runtime requires ordered features {expected_names}; "
            f"observed names={observed_names}, shapes={shapes}"
        )
    if any(len(shape) != 4 for shape in shapes.values()):
        raise ValueError(
            f"{model} dense runtime features {expected_names} must all be BCHW; "
            f"observed shapes={shapes}"
        )

    expected_channels = interface.get("raw_channels")
    observed_channels = [shapes[name][1] for name in expected_names]
    if expected_channels and observed_channels != list(expected_channels):
        raise ValueError(
            f"{model} dense runtime features {expected_names} require channels "
            f"{list(expected_channels)}; observed shapes={shapes}"
        )

    spatial = [(shapes[name][2], shapes[name][3]) for name in expected_names]
    geometry = interface.get("geometry")
    if geometry == "strict_dyadic":
        valid = all(
            h0 == 2 * h1 and w0 == 2 * w1
            for (h0, w0), (h1, w1) in zip(spatial, spatial[1:])
        )
    elif geometry == "deepest_plateau":
        valid = (
            len(spatial) == 4
            and spatial[0][0] == 2 * spatial[1][0]
            and spatial[0][1] == 2 * spatial[1][1]
            and spatial[1][0] == 2 * spatial[2][0]
            and spatial[1][1] == 2 * spatial[2][1]
            and spatial[2] == spatial[3]
        )
    elif geometry == "equal_grid":
        valid = len(spatial) == 4 and len(set(spatial)) == 1
    else:
        raise ValueError(f"{model} has unsupported dense geometry policy {geometry!r}")
    if not valid:
        raise ValueError(
            f"{model} dense runtime features {expected_names} violate geometry "
            f"{geometry!r}; observed shapes={shapes}"
        )
    return shapes
