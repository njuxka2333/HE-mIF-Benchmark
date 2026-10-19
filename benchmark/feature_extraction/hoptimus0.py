"""Frozen H-optimus-0 = MIPHEI-ViT encoder route with LoRA and the generator checkpoint off.

The MIPHEI wrapper still builds the H-optimus-0 ViT (256 px, final-block GAP).
Skipping ``model.safetensors`` is required: that file holds LoRA + decoder
weights, and a non-strict load would still touch encoder keys.
"""

from __future__ import annotations

import os
import sys

_SKIP_VALUES = {"1", "true", "yes", "on"}


def frozen_hoptimus_requested() -> bool:
    return str(os.environ.get("MIPHEI_SKIP_GENERATOR_CKPT") or "").strip().lower() in _SKIP_VALUES


def apply_extract_env_from_config(config_path) -> None:
    """Load YAML ``extract.env`` and install the frozen encoder if requested.

    Feature extract already does this. Online TLS / cellseg training must too,
    otherwise ``benchmark_models.yaml`` turns LoRA back on and loads MIPHEI's
    generator checkpoint.
    """
    from pathlib import Path

    import yaml

    from utils.paths import resolve_project_path

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
        install_frozen_encoder()


def install_frozen_encoder() -> None:
    """Disable LoRA and skip the MIPHEI generator checkpoint before extract loads the net."""
    os.environ["MIPHEI_USE_LORA"] = "0"
    os.environ.setdefault("MIPHEI_ENCODER_NAME", "hoptimus0")
    os.environ.setdefault("MIPHEI_IMG_SIZE", "256")
    os.environ.setdefault("MIPHEI_SKIP_GENERATOR_CKPT", "1")

    from utils.hf_auth import login_huggingface

    login_huggingface()

    import models.mipheivit_inference as miphei

    def _skip_generator_checkpoint(net, ckpt_path) -> None:
        del net
        print(
            "H-optimus-0: foundation encoder only; skipping MIPHEI generator/LoRA "
            f"checkpoint {ckpt_path}",
            flush=True,
        )

    bytecode = sys.modules.get("_bytecode_mipheivit_inference")
    if bytecode is not None:
        bytecode._load_checkpoint = _skip_generator_checkpoint
    miphei._load_checkpoint = _skip_generator_checkpoint
