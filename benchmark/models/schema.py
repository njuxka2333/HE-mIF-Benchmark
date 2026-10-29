"""
Configuration schema for HE-mIF-Benchmark feature extraction.

Defines typed configs for model loading, preprocessing, outputs, and downstream heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NormType(str, Enum):
    """Input normalization strategy (matches upstream repo conventions)."""

    IMAGENET_01 = "imagenet_01"  # ToTensor [0,1] then ImageNet mean/std
    IMAGENET_255 = "imagenet_255"  # uint8 HWC, stats in 0–255 space (MIPHEI native)
    PIX2PIX_01 = "pix2pix_01"  # ToTensor then (x-0.5)/0.5 → [-1, 1] (HEMIT)
    NONE = "none"  # ToTensor only


class FeatureOutputType(str, Enum):
    """Native backbone output before head-specific post-processing."""

    EMBEDDING = "embedding"  # [B, D] or [N, D]
    DENSE_SPATIAL = "dense_spatial"  # [B, C, H, W]


class HeadKind(str, Enum):
    LINEAR_PROBE = "linear_probe"
    ABMIL = "abmil"
    UNETR = "unetr"


@dataclass
class NormConfig:
    norm_type: NormType = NormType.IMAGENET_01
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    # MIPHEI-style 0–255 stats (used when norm_type=imagenet_255)
    mean_255: Tuple[float, ...] = (123.675, 116.28, 103.53)
    std_255: Tuple[float, ...] = (58.395, 57.12, 57.375)


@dataclass
class SpatialConfig:
    """Patch / tile geometry expected at model input."""

    native_input_size: Optional[int] = None  # e.g. 224 ROSIE, 1024 HEMIT, 256 MIPHEI
    window_size: Optional[int] = None  # tiled inference window (GigaTIME=256)
    context_patch_size: Optional[int] = None  # ROSIE crop before resize
    resize_to: Optional[int] = None  # final spatial size fed to trunk
    min_size: int = 1  # pad upsample threshold (patch_compat)
    allow_variable_hw: bool = True


@dataclass
class MppConfig:
    """MPP handling for WSI coordinate extraction (PrePATH patch pipeline)."""

    target_mpp: Optional[float] = None
    fallback_mpp: float = 0.25
    # Legacy PrePATH resolution.py keys (20x/40x/80x → patch size)
    resolution_key: Optional[str] = None  # e.g. "20x", "40x", "80x"


@dataclass
class ModelOutputConfig:
    native_type: FeatureOutputType = FeatureOutputType.EMBEDDING
    channels: Optional[int] = None  # dense spatial channel count
    embedding_dim: Optional[int] = None
    activation: Optional[str] = None  # e.g. tanh for mIF generators


@dataclass
class HeadContract:
    """Expected encoder extraction for a downstream task (linear_probe / abmil / unetr)."""

    enabled: bool = True
    layer_hint: str = ""
    notes: str = ""


@dataclass
class PostProcessConfig:
    """Model-specific output post-processing."""

    name: str = "identity"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HubCheckpointConfig:
    """
    Remote checkpoint source.

    Use ``repo_id`` + ``filename`` for Hugging Face Hub (``hf_hub_download``),
    or ``url`` for a direct artifact URL (e.g. GitHub release).
    """

    repo_id: str = ""
    filename: str = ""
    revision: str = "main"
    cache_dir: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    aliases: List[str] = field(default_factory=list)
    repo: str = ""
    checkpoint_key: str = ""
    checkpoint_path: Optional[str] = None
    hub: Optional[HubCheckpointConfig] = None
    loader: str = "legacy"  # legacy | benchmark_wrapper
    wrapper_module: Optional[str] = None
    batch_size_default: int = 32
    norm: NormConfig = field(default_factory=NormConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    mpp: MppConfig = field(default_factory=MppConfig)
    output: ModelOutputConfig = field(default_factory=ModelOutputConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    heads: Dict[str, HeadContract] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def uses_hub(self) -> bool:
        if not self.hub:
            return False
        return bool(self.hub.repo_id or self.hub.url)


@dataclass
class ExtractionRunConfig:
    """Top-level run settings consumed by the feature extraction engine."""

    model: str
    feat_dir: str
    target_mpp: Optional[float] = None
    fallback_mpp: float = 0.25
    target_patch_size: int = -1
    custom_downsample: int = 1
    batch_size: Optional[int] = None
    output_layout: str = "prepath_legacy"  # prepath_legacy | benchmark_v1
    save_storage: str = "yes"
    is_patch_dataset: bool = False
