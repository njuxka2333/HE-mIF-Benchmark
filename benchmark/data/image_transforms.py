"""Shared conversion from RGB NumPy patches to model image tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def patch_to_tensor(patch: np.ndarray, transform) -> Any:
    image = Image.fromarray(np.asarray(patch, dtype=np.uint8)).convert("RGB")
    return transform(image)
