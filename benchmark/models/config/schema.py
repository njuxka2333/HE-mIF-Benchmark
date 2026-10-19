"""Shim: schema lives in ``models.schema``."""

from models.schema import *  # noqa: F403
from models.schema import (  # noqa: F401
    ExtractionRunConfig,
    FeatureOutputType,
    HeadContract,
    HeadKind,
    HubCheckpointConfig,
    ModelConfig,
    ModelOutputConfig,
    MppConfig,
    NormConfig,
    NormType,
    PostProcessConfig,
    SpatialConfig,
)
