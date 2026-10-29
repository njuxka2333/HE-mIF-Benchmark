"""Shared batch operations for generated-mIF segmentation trainers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from virtual_staining.common import estimate_channel_stats


def prepare_segmentation_batch(
    batch: dict[str, Any],
    augmenter: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move RGB images and semantic masks to device, then apply joint augmentation."""
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["semantic_mask"].to(device, non_blocking=True)
    if augmenter is None:
        return images, masks
    augmented = augmenter(images, masks)
    if isinstance(augmented, (tuple, list)):
        return augmented[0], augmented[1]
    return images, masks


@torch.no_grad()
def estimate_generator_channel_stats(
    model: nn.Module,
    sampler: Any,
    *,
    device: torch.device,
    num_batches: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate generated-mIF channel mean/std from sampled RGB batches."""
    generator = getattr(model, "generator", None)
    if generator is None:
        raise ValueError("model.generator is required to estimate online channel stats")
    model.eval()
    generator.eval()
    generated_batches: list[torch.Tensor] = []
    for _ in range(max(1, int(num_batches))):
        batch = sampler.next_batch()
        rgb = batch["image"].to(device, non_blocking=True)
        generated = generator(rgb)
        generated_batches.append(generated.detach().float().cpu())
    return estimate_channel_stats(generated_batches)
