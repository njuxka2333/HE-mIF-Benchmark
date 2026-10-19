"""Shim: resolution helpers live in ``models.registry``."""

from models.registry import (  # noqa: F401
    SIZES,
    STEPS,
    patch_size_for_key,
    recommended_patch_size,
)
