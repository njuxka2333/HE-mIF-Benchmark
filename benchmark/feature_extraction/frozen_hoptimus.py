"""Select the frozen H-optimus-0 encoder through the MIPHEI wrapper.

H-optimus-0 is the pretrained ViT backbone inside MIPHEI-ViT. This route
extracts its encoder features with LoRA and the MIPHEI generator checkpoint off.
"""

from __future__ import annotations

import os

_SKIP_VALUES = {"1", "true", "yes", "on"}


def frozen_hoptimus_requested() -> bool:
    """Whether the current config selects the frozen foundation-encoder route."""
    return str(os.environ.get("MIPHEI_SKIP_GENERATOR_CKPT") or "").strip().lower() in _SKIP_VALUES


def apply_extract_env_from_config(config_path) -> None:
    """Apply ``extract.env`` and configure the frozen route when requested."""
    from pathlib import Path

    import yaml

    from utils.paths import resolve_project_path

    # Configs are run in one process by some entry points; clear route state so
    # a previous H-optimus run cannot affect a normal MIPHEI run.
    os.environ.pop("MIPHEI_SKIP_GENERATOR_CKPT", None)
    path = resolve_project_path(config_path)
    if not path.is_file():
        path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("extract") or {}
    env = block.get("env") if isinstance(block, dict) else None
    if isinstance(env, dict):
        for key, value in env.items():
            os.environ[str(key)] = "" if value is None else str(value)
    if frozen_hoptimus_requested():
        configure_frozen_hoptimus()


def configure_frozen_hoptimus() -> None:
    """Disable LoRA and the generator checkpoint before loading MIPHEI-ViT."""
    os.environ["MIPHEI_USE_LORA"] = "0"
    os.environ.setdefault("MIPHEI_ENCODER_NAME", "hoptimus0")
    os.environ.setdefault("MIPHEI_IMG_SIZE", "256")
    os.environ["MIPHEI_SKIP_GENERATOR_CKPT"] = "1"

    from utils.hf_auth import login_huggingface

    login_huggingface()
