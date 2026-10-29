"""Model registry for HE-mIF-Benchmark feature extraction."""

from __future__ import annotations

import os

__all__ = ["list_models", "get_model", "get_custom_transformer", "BENCHMARK_MODELS"]

BENCHMARK_MODELS = ("rosie", "hemit", "gigatime", "mipheivit", "uni2", "histoplexer")


def _normalize_name(model_name: str) -> str:
    key = model_name.lower()
    aliases = {
        "hemmit": "hemit",
        "miphei": "mipheivit",
    }
    return aliases.get(key, key)


def _resolve_ckpt(name: str, config_path=None, must_exist: bool = True) -> str:
    from models.registry import get_model_config
    from utils.checkpoints import resolve_model_checkpoint

    cfg = get_model_config(name, config_path)
    return resolve_model_checkpoint(cfg, must_exist=must_exist, config_path=config_path)


def list_models(config_path=None):
    from models.registry import list_benchmark_models

    print("Implemented HE-mIF-Benchmark encoders (checkpoints from benchmark_models.yaml):")
    for name in list_benchmark_models(config_path):
        try:
            ckpt = _resolve_ckpt(name, config_path, must_exist=False)
        except FileNotFoundError:
            ckpt = "(not configured)"
        print(f"  {name}: {ckpt}")
    return {name: _resolve_ckpt(name, config_path, must_exist=False) for name in BENCHMARK_MODELS}


def get_model(
    model_name,
    device,
    gpu_num,
    jit=False,
    use_benchmark_config=False,
    config_path=None,
):
    del jit
    name = _normalize_name(model_name)
    if name not in BENCHMARK_MODELS:
        raise NotImplementedError(
            f"{model_name} is not supported. Use one of: {', '.join(BENCHMARK_MODELS)}"
        )

    if use_benchmark_config:
        from models.loader import load_model_from_config
        from models.registry import apply_model_env, get_model_config, is_benchmark_model

        if is_benchmark_model(model_name, config_path):

            cfg = get_model_config(model_name, config_path)
            apply_model_env(cfg)
            from feature_extraction.frozen_hoptimus import (
                frozen_hoptimus_requested,
                configure_frozen_hoptimus,
            )

            if frozen_hoptimus_requested():
                configure_frozen_hoptimus()
            return load_model_from_config(model_name, device, gpu_num, config_path)

    ckpt = None if name == "uni2" else _resolve_ckpt(name, config_path)

    if name == "hemit":
        from models.hemmit_inference import get_model as get_hemit_model

        model = get_hemit_model(device, ckpt, gpu_num)
    elif name == "gigatime":
        from models.gigatime_inference import get_model as get_gigatime_model

        model = get_gigatime_model(device, ckpt, gpu_num)
    elif name == "mipheivit":
        from models.mipheivit_inference import get_model as get_mipheivit_model

        img_size = int(os.environ.get("MIPHEI_IMG_SIZE", "256"))
        model = get_mipheivit_model(device, ckpt, gpu_num, img_size=img_size)
    elif name == "rosie":
        from models.rosie_inference import get_model as get_rosie_model

        model = get_rosie_model(device, ckpt, gpu_num)
    elif name == "uni2":
        from models.uni2_inference import get_model as get_uni2_model

        model = get_uni2_model(device, gpu_num)
    elif name == "histoplexer":
        from models.histoplexer_inference import get_model as get_histoplexer_model

        model = get_histoplexer_model(device, ckpt, gpu_num)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")

    return model.eval()


def get_custom_transformer(model_name, use_benchmark_config=False, config_path=None):
    name = _normalize_name(model_name)
    if name not in BENCHMARK_MODELS:
        raise NotImplementedError(
            f"Transform for {model_name} is not supported. "
            f"Use one of: {', '.join(BENCHMARK_MODELS)}"
        )

    if use_benchmark_config:
        from models.loader import load_transform_from_config
        from models.registry import apply_model_env, get_model_config, is_benchmark_model

        if is_benchmark_model(model_name, config_path):
            cfg = get_model_config(model_name, config_path)
            apply_model_env(cfg)
            return load_transform_from_config(model_name, config_path)

    if name == "hemit":
        from models.hemmit_inference import get_trans as get_hemit_trans

        hemit_size = os.environ.get("HEMIT_TARGET_SIZE", "1024")
        sz = int(hemit_size)
        return get_hemit_trans(sz if sz > 0 else None)
    if name == "gigatime":
        from models.gigatime_inference import get_trans as get_gigatime_trans

        return get_gigatime_trans()
    if name == "mipheivit":
        from models.mipheivit_inference import get_trans as get_mipheivit_trans

        img_size = int(os.environ.get("MIPHEI_IMG_SIZE", "256"))
        return get_mipheivit_trans(img_size if img_size > 0 else None)
    if name == "rosie":
        from models.rosie_inference import get_trans as get_rosie_trans

        return get_rosie_trans()
    if name == "uni2":
        from models.uni2_inference import get_trans as get_uni2_trans

        return get_uni2_trans()
    if name == "histoplexer":
        from models.histoplexer_inference import get_trans as get_histoplexer_trans

        target_size = int(os.environ.get("HISTOPLEXER_TARGET_SIZE", "1024"))
        return get_histoplexer_trans(target_size if target_size > 0 else None)

    raise NotImplementedError(f"Transform for {model_name} is not implemented")


def get_model_config(model_name, config_path=None):
    """Re-export for callers that imported from models."""
    from models.registry import get_model_config as _get

    return _get(model_name, config_path)
