"""Shim: registry lives in ``models.registry``."""

from models.registry import *  # noqa: F403
from models.registry import (  # noqa: F401
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
