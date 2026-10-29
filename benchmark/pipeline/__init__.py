"""Configuration-driven single-run pipeline contracts and orchestration."""

from pipeline.plan import PipelinePlan, load_pipeline_plan
from pipeline.runner import PipelineResult, run_pipeline

__all__ = ["PipelinePlan", "PipelineResult", "load_pipeline_plan", "run_pipeline"]
