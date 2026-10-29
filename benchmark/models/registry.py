"""
Load and resolve benchmark model configs from YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from models.schema import (
    ExtractionRunConfig,
    FeatureOutputType,
    HeadContract,
    HeadKind,
    HubCheckpointConfig,
    ModelConfig,
    ModelOutputConfig,
    MppConfig,
    NormConfig,
    NormType,
    PostProcessConfig,
    SpatialConfig,
)

from utils.paths import CONFIG_DIR

_DEFAULT_YAML = CONFIG_DIR / "benchmark_models.yaml"


def _tuple3(values: List[float]) -> tuple:
    return (float(values[0]), float(values[1]), float(values[2]))


def _parse_norm(data: Dict[str, Any]) -> NormConfig:
    norm_type = NormType(data.get("norm_type", NormType.IMAGENET_01.value))
    mean = _tuple3(data.get("mean", [0.485, 0.456, 0.406]))
    std = _tuple3(data.get("std", [0.229, 0.224, 0.225]))
    mean_255 = _tuple3(data.get("mean_255", [123.675, 116.28, 103.53]))
    std_255 = _tuple3(data.get("std_255", [58.395, 57.12, 57.375]))
    return NormConfig(
        norm_type=norm_type,
        mean=mean,
        std=std,
        mean_255=mean_255,
        std_255=std_255,
    )


def _parse_spatial(data: Dict[str, Any]) -> SpatialConfig:
    return SpatialConfig(
        native_input_size=data.get("native_input_size"),
        window_size=data.get("window_size"),
        context_patch_size=data.get("context_patch_size"),
        resize_to=data.get("resize_to"),
        min_size=int(data.get("min_size", 1)),
        allow_variable_hw=bool(data.get("allow_variable_hw", True)),
    )


def _parse_mpp(data: Dict[str, Any]) -> MppConfig:
    return MppConfig(
        target_mpp=data.get("target_mpp"),
        fallback_mpp=float(data.get("fallback_mpp", 0.25)),
        resolution_key=data.get("resolution_key"),
    )


def _parse_output(data: Dict[str, Any]) -> ModelOutputConfig:
    return ModelOutputConfig(
        native_type=FeatureOutputType(data.get("native_type", FeatureOutputType.EMBEDDING.value)),
        channels=data.get("channels"),
        embedding_dim=data.get("embedding_dim"),
        activation=data.get("activation"),
    )


def _parse_postprocess(data: Dict[str, Any]) -> PostProcessConfig:
    return PostProcessConfig(
        name=data.get("name", "identity"),
        params=dict(data.get("params") or {}),
    )


def _parse_heads(data: Dict[str, Any]) -> Dict[str, HeadContract]:
    out: Dict[str, HeadContract] = {}
    for key, val in (data or {}).items():
        if not isinstance(val, dict):
            continue
        out[key] = HeadContract(
            enabled=bool(val.get("enabled", True)),
            layer_hint=str(val.get("layer_hint", "")),
            notes=str(val.get("notes", "")),
        )
    return out


def _parse_hub(data: Any) -> Optional[HubCheckpointConfig]:
    if not data or not isinstance(data, dict):
        return None
    repo_id = str(data.get("repo_id") or "").strip()
    url = data.get("url")
    if not repo_id and not url:
        return None
    return HubCheckpointConfig(
        repo_id=repo_id,
        filename=str(data.get("filename") or ""),
        revision=str(data.get("revision") or "main"),
        cache_dir=data.get("cache_dir"),
        url=url,
    )


def _parse_model(name: str, data: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=name,
        aliases=list(data.get("aliases") or []),
        repo=str(data.get("repo", "")),
        checkpoint_key=str(data.get("checkpoint_key") or name),
        checkpoint_path=data.get("checkpoint_path"),
        hub=_parse_hub(data.get("hub")),
        loader=str(data.get("loader", "legacy")),
        wrapper_module=data.get("wrapper_module"),
        batch_size_default=int(data.get("batch_size_default", 32)),
        norm=_parse_norm(data.get("norm") or {}),
        spatial=_parse_spatial(data.get("spatial") or {}),
        mpp=_parse_mpp(data.get("mpp") or {}),
        output=_parse_output(data.get("output") or {}),
        postprocess=_parse_postprocess(data.get("postprocess") or {}),
        heads=_parse_heads(data.get("heads") or {}),
        extra=dict(data.get("extra") or {}),
    )


@lru_cache(maxsize=1)
def load_benchmark_registry(config_path: Optional[str] = None) -> Dict[str, ModelConfig]:
    path = Path(config_path) if config_path else _DEFAULT_YAML
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    models_raw = raw.get("models") or {}
    registry: Dict[str, ModelConfig] = {}
    for name, data in models_raw.items():
        cfg = _parse_model(name, data)
        registry[name.lower()] = cfg
        for alias in cfg.aliases:
            registry[alias.lower()] = cfg
    return registry


def list_benchmark_models(config_path: Optional[str] = None) -> List[str]:
    registry = load_benchmark_registry(config_path)
    return sorted({cfg.name for cfg in registry.values()})


def is_benchmark_model(model_name: str, config_path: Optional[str] = None) -> bool:
    return model_name.lower() in load_benchmark_registry(config_path)


def get_model_config(model_name: str, config_path: Optional[str] = None) -> ModelConfig:
    key = model_name.lower()
    registry = load_benchmark_registry(config_path)
    if key not in registry:
        raise KeyError(
            f"Model '{model_name}' not in benchmark registry. "
            f"Available: {', '.join(list_benchmark_models(config_path))}"
        )
    return registry[key]


def resolve_extraction_run(
    model_name: str,
    feat_dir: str,
    *,
    target_mpp: Optional[float] = None,
    fallback_mpp: Optional[float] = None,
    target_patch_size: int = -1,
    custom_downsample: int = 1,
    batch_size: Optional[int] = None,
    output_layout: str = "prepath_legacy",
    save_storage: str = "yes",
    is_patch_dataset: bool = False,
    config_path: Optional[str] = None,
) -> ExtractionRunConfig:
    cfg = get_model_config(model_name, config_path)
    resolved_mpp = target_mpp if target_mpp is not None else cfg.mpp.target_mpp
    resolved_fallback = fallback_mpp if fallback_mpp is not None else cfg.mpp.fallback_mpp
    resolved_batch = batch_size if batch_size is not None else cfg.batch_size_default
    return ExtractionRunConfig(
        model=cfg.name,
        feat_dir=feat_dir,
        target_mpp=resolved_mpp,
        fallback_mpp=resolved_fallback,
        target_patch_size=target_patch_size,
        custom_downsample=custom_downsample,
        batch_size=resolved_batch,
        output_layout=output_layout,
        save_storage=save_storage,
        is_patch_dataset=is_patch_dataset,
    )


def get_output_layout(layout_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_YAML
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    layouts = raw.get("output_layouts") or {}
    if layout_name not in layouts:
        raise KeyError(f"Unknown output layout '{layout_name}'. Options: {list(layouts)}")
    return layouts[layout_name]


def head_is_enabled(model_name: str, head: HeadKind | str, config_path: Optional[str] = None) -> bool:
    cfg = get_model_config(model_name, config_path)
    key = head.value if isinstance(head, HeadKind) else str(head)
    contract = cfg.heads.get(key)
    return contract.enabled if contract else False


def apply_model_env(cfg: ModelConfig) -> None:
    """Apply per-model runtime settings (env vars used by inference wrappers)."""
    if cfg.name == "mipheivit":
        img_size = cfg.extra.get("img_size") or cfg.spatial.native_input_size
        if img_size:
            os.environ["MIPHEI_IMG_SIZE"] = str(int(img_size))
        if cfg.extra.get("encoder_name"):
            os.environ["MIPHEI_ENCODER_NAME"] = str(cfg.extra["encoder_name"])
        if cfg.extra.get("num_classes") is not None:
            os.environ["MIPHEI_NUM_CLASSES"] = str(cfg.extra["num_classes"])
        from feature_extraction.frozen_hoptimus import frozen_hoptimus_requested

        if frozen_hoptimus_requested():
            os.environ["MIPHEI_USE_LORA"] = "0"
        elif cfg.extra.get("use_lora") is not None:
            os.environ["MIPHEI_USE_LORA"] = "1" if cfg.extra["use_lora"] else "0"
        return

    if cfg.name == "hemit":
        force = os.environ.get("HEMIT_FORCE_SIZE")
        if force is not None and str(force).strip() != "":
            os.environ["HEMIT_TARGET_SIZE"] = str(int(force))
            return
        sz = cfg.spatial.native_input_size
        if sz:
            os.environ["HEMIT_TARGET_SIZE"] = str(int(sz))
        return

    if cfg.name == "histoplexer":
        sz = cfg.spatial.native_input_size
        if sz:
            os.environ["HISTOPLEXER_TARGET_SIZE"] = str(int(sz))
        return


STEPS = {"20x": 256, "40x": 512, "80x": 1024}
SIZES = {"20x": 256, "40x": 512, "80x": 1024}


def patch_size_for_key(resolution_key: str) -> Optional[int]:
    return SIZES.get(resolution_key)


def recommended_patch_size(model_name: str, config_path: Optional[str] = None) -> Optional[int]:
    """Resolve default target_patch_size from benchmark_models.yaml resolution_key."""
    try:
        cfg = get_model_config(model_name, config_path)
        if cfg.mpp.resolution_key:
            return patch_size_for_key(cfg.mpp.resolution_key)
    except KeyError:
        pass
    return None
