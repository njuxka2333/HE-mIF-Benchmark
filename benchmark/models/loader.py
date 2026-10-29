"""Load a benchmark model / transform from ``benchmark_models.yaml``.

The archived ``_orig_pyc/loader.pyc`` is gone. Dispatch matches
``models.get_model`` / ``get_custom_transformer`` when
``use_benchmark_config=True``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def load_model_from_config(
    model_name,
    device,
    gpu_num: int = 1,
    config_path: Optional[str] = None,
) -> nn.Module:
    from models.registry import get_model_config
    from utils.checkpoints import resolve_model_checkpoint

    name = str(model_name).lower()
    aliases = {"hemmit": "hemit", "miphei": "mipheivit"}
    name = aliases.get(name, name)
    cfg = get_model_config(model_name, config_path)
    from feature_extraction.frozen_hoptimus import frozen_hoptimus_requested

    skip_generator = name == "mipheivit" and frozen_hoptimus_requested()
    ckpt = None if name == "uni2" or skip_generator else resolve_model_checkpoint(
        cfg, config_path=config_path
    )

    if name == "hemit":
        from models.hemmit_inference import get_model as fn

        model = fn(device, ckpt, gpu_num)
    elif name == "gigatime":
        from models.gigatime_inference import get_model as fn

        model = fn(device, ckpt, gpu_num)
    elif name == "mipheivit":
        from models.mipheivit_inference import get_model as fn

        img_size = int(os.environ.get("MIPHEI_IMG_SIZE") or cfg.extra.get("img_size") or 256)
        model = fn(device, ckpt, gpu_num, img_size=img_size)
    elif name == "rosie":
        from models.rosie_inference import get_model as fn

        model = fn(device, ckpt, gpu_num)
    elif name == "uni2":
        from models.uni2_inference import get_model as fn

        model = fn(device, gpu_num)
    elif name == "histoplexer":
        from models.histoplexer_inference import get_model as fn

        model = fn(device, ckpt, gpu_num)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    return model.eval()


def load_transform_from_config(model_name, config_path: Optional[str] = None):
    from models.registry import get_model_config

    name = str(model_name).lower()
    aliases = {"hemmit": "hemit", "miphei": "mipheivit"}
    name = aliases.get(name, name)
    cfg = get_model_config(model_name, config_path)

    if name == "hemit":
        from models.hemmit_inference import get_trans as fn

        force = os.environ.get("HEMIT_FORCE_SIZE")
        if force is not None and str(force).strip() != "":
            size = int(force)
        else:
            size = int(os.environ.get("HEMIT_TARGET_SIZE") or cfg.spatial.native_input_size or 1024)
        return fn(size if size > 0 else None)
    if name == "gigatime":
        from models.gigatime_inference import get_trans as fn

        return fn()
    if name == "mipheivit":
        from models.mipheivit_inference import get_trans as fn

        img_size = int(os.environ.get("MIPHEI_IMG_SIZE") or cfg.extra.get("img_size") or 256)
        return fn(img_size if img_size > 0 else None)
    if name == "rosie":
        from models.rosie_inference import get_trans as fn

        return fn()
    if name == "uni2":
        from models.uni2_inference import get_trans as fn

        return fn()
    if name == "histoplexer":
        from models.histoplexer_inference import get_trans as fn

        size = int(os.environ.get("HISTOPLEXER_TARGET_SIZE") or cfg.spatial.native_input_size or 1024)
        return fn(size if size > 0 else None)
    raise NotImplementedError(f"Transform for {model_name} is not implemented")
