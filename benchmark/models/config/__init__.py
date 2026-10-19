"""Compatibility shim for ``models.config`` imports.

Model schema/registry live in ``models/``; checkpoint loading lives in
``utils/checkpoints.py``.
"""

from models.registry import (
    apply_model_env,
    get_model_config,
    get_output_layout,
    head_is_enabled,
    is_benchmark_model,
    list_benchmark_models,
    load_benchmark_registry,
    patch_size_for_key,
    recommended_patch_size,
    resolve_extraction_run,
)
from models.schema import ExtractionRunConfig, HeadKind, ModelConfig, NormType
from utils import checkpoints
from utils.checkpoints import resolve_model_checkpoint

__all__ = [
    "ExtractionRunConfig",
    "HeadKind",
    "ModelConfig",
    "NormType",
    "apply_model_env",
    "get_model_config",
    "get_output_layout",
    "head_is_enabled",
    "is_benchmark_model",
    "list_benchmark_models",
    "load_benchmark_registry",
    "patch_size_for_key",
    "recommended_patch_size",
    "resolve_extraction_run",
    "resolve_model_checkpoint",
]
