"""Config-driven segmentation augmentations aligned with PFMB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    import kornia.augmentation as Kaug
    from kornia.constants import Resample as KorniaResample

    _HAS_KORNIA = True
except ImportError:  # pragma: no cover - optional dependency
    Kaug = None
    KorniaResample = None
    _HAS_KORNIA = False

import numpy as np


@dataclass(frozen=True)
class InputNormalization:
    """Channel normalization applied by a model's input transform."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def _tensors(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, len(self.mean), 1, 1)
        mean = images.new_tensor(self.mean).view(shape)
        std = images.new_tensor(self.std).view(shape)
        return mean, std

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        """Undo model normalization so augmentation receives RGB in [0, 1]."""
        mean, std = self._tensors(images)
        return (images * std + mean).clamp_(0.0, 1.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Reapply the model's official normalization after augmentation."""
        mean, std = self._tensors(images)
        return (images - mean) / std


def find_input_normalization(transform: Any) -> InputNormalization | None:
    """Find the torchvision Normalize operation inside a composed transform."""
    children = getattr(transform, "transforms", None)
    if children is not None:
        for child in children:
            found = find_input_normalization(child)
            if found is not None:
                return found
    mean = getattr(transform, "mean", None)
    std = getattr(transform, "std", None)
    if mean is None or std is None:
        return None
    mean_values = tuple(float(value) for value in mean)
    std_values = tuple(float(value) for value in std)
    if not mean_values or len(mean_values) != len(std_values):
        return None
    return InputNormalization(mean=mean_values, std=std_values)


RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def _torch_rgb2hed(img: torch.Tensor, hed_t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    img = img.movedim(-3, -1)
    img = torch.clamp(img, min=eps)
    img = torch.log(img) / torch.log(eps)
    img = torch.matmul(img, hed_t)
    return img.movedim(-1, -3)


def _torch_hed2rgb(img: torch.Tensor, rgb_t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    neg_log_eps = -torch.log(eps)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * neg_log_eps), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


class HEDNormalize(nn.Module):
    """PFMB-style HED color augmentation for RGB inputs."""

    def __init__(self, sigma: float = 0.025, bias: float = 0.05) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.bias = float(bias)
        self.register_buffer("eps", torch.tensor(1e-6))
        self.register_buffer("rgb_t", torch.from_numpy(RGB_FROM_HED))
        self.register_buffer("hed_t", torch.from_numpy(HED_FROM_RGB))

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        device = images.device
        sigmas = torch.empty(batch_size, 3, device=device).uniform_(-self.sigma, self.sigma)
        biases = torch.empty(batch_size, 3, device=device).uniform_(-self.bias, self.bias)
        hed = _torch_rgb2hed(images, self.hed_t, self.eps)
        hed = hed * (1 + sigmas.view(batch_size, 3, 1, 1)) + biases.view(batch_size, 3, 1, 1)
        return _torch_hed2rgb(hed, self.rgb_t, self.eps), masks


def _resolve_resample(value: Any):
    if isinstance(value, str) and value.upper() == "BILINEAR":
        if _HAS_KORNIA and KorniaResample is not None:
            return KorniaResample.BILINEAR
        return "bilinear"
    return value


def _build_transform(name: str, params: dict[str, Any] | None) -> nn.Module:
    params = dict(params or {})
    if name == "HEDNormalize":
        return HEDNormalize(**params)

    if not _HAS_KORNIA:
        raise ImportError(
            f"Augmentation '{name}' requires kornia. Install kornia or remove it from the config."
        )

    if name == "RandomAffine" and "resample" in params:
        params["resample"] = _resolve_resample(params["resample"])
    if name == "RandomElasticTransform":
        for key in ("kernel_size", "sigma", "alpha"):
            if key in params and isinstance(params[key], list):
                # Kornia expects 2-tuples for these ranges. Converting YAML
                # lists to tensors breaks its scalar/range validation.
                params[key] = tuple(params[key])

    if hasattr(Kaug, name):
        return getattr(Kaug, name)(**params)
    raise ValueError(f"Unknown augmentation '{name}'")


class SegmentationAugmenter(nn.Module):
    """Apply PFMB-style geometric and color augmentations to image/mask batches."""

    def __init__(self, config: dict[str, Any] | None) -> None:
        super().__init__()
        self.config = dict(config or {})
        geom_transforms: list[nn.Module] = []
        self._color = nn.ModuleList()
        for name, params in self.config.items():
            if name == "HEDNormalize":
                self._color.append(_build_transform(name, params))
            else:
                geom_transforms.append(_build_transform(name, params))

        self._geom: Kaug.AugmentationSequential | None = None
        if geom_transforms:
            if not _HAS_KORNIA:
                raise ImportError("Geometric augmentations require kornia.")
            self._geom = Kaug.AugmentationSequential(
                *geom_transforms,
                data_keys=("input", "mask", "mask"),
            )

    @property
    def enabled(self) -> bool:
        return self._geom is not None or len(self._color) > 0

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        instance_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            if instance_masks is None:
                return images, masks
            return images, masks, instance_masks

        mask_tensor = masks.float()
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        inst_tensor = None
        if instance_masks is not None:
            inst_tensor = instance_masks.float()
            if inst_tensor.ndim == 3:
                inst_tensor = inst_tensor.unsqueeze(1)

        if self._geom is not None:
            if inst_tensor is not None:
                images, mask_tensor, inst_tensor = self._geom(images, mask_tensor, inst_tensor)
            else:
                images, mask_tensor, _ = self._geom(images, mask_tensor, mask_tensor.clone())

        for transform in self._color:
            images, mask_tensor = transform(images, mask_tensor)

        if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.squeeze(1)
        masks_out = mask_tensor.round().to(dtype=masks.dtype)
        if inst_tensor is None:
            return images, masks_out

        if inst_tensor.ndim == 4 and inst_tensor.shape[1] == 1:
            inst_tensor = inst_tensor.squeeze(1)
        return images, masks_out, inst_tensor.round().to(dtype=instance_masks.dtype)


def build_augmenter(training_cfg: dict[str, Any] | None) -> SegmentationAugmenter | None:
    training_cfg = dict(training_cfg or {})
    aug_cfg = training_cfg.get("augmentations")
    if not aug_cfg:
        return None
    augmenter = SegmentationAugmenter(aug_cfg)
    return augmenter if augmenter.enabled else None
