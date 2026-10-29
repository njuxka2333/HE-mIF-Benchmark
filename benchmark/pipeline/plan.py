"""Resolve one experiment YAML into explicit pipeline stage choices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from data.loader import get_dataset_config
from models.registry import get_model_config
from utils.paths import PREPATH_ROOT, relocate_lab_path, resolve_project_path

ONLINE_TASKS = {"cell_semantic_segmentation", "tls_region_segmentation"}


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve_trainer(task: str, head_name: str, decoder_mode: str) -> str:
    if task == "virtual_staining" or head_name == "virtual_stain":
        return "virtual_staining_score"
    if decoder_mode == "mif_unet":
        return "virtual_staining_unet"
    if decoder_mode == "mif_feature":
        return "virtual_staining_feature"
    if task in ONLINE_TASKS:
        return "online"
    return "offline"


def _resolve_loader(
    *,
    task: str,
    dataset_name: str,
    data: dict[str, Any],
    run: dict[str, Any],
    dataset_mode: str | None,
) -> str:
    configured_loader = data.get("loader")
    if configured_loader:
        return str(configured_loader).strip().lower()
    if task == "virtual_staining":
        return "paired_fov"
    if data.get("feature_experiment_config") or dataset_name == "linear_probe_features":
        return "feature_bags"
    if task == "tls_region_segmentation" and run.get("fixed_grid_sampling"):
        return "fixed_grid_png_tiles"
    if dataset_mode == "patch_hdf":
        return "hdf_volume_tiles"
    if dataset_mode == "wsi":
        return "wsi_manifest"
    return "image_manifest"


@dataclass(frozen=True)
class PipelinePlan:
    """Normalized input, processing, model, and output choices for one run."""

    config_path: Path
    run_id: str
    task: str
    dataset: str
    source_dataset: str
    model: str | None
    trainer: str
    execution_mode: str
    data_loader: str
    feature_layers: tuple[str, ...]
    feature_experiment_config: Path | None
    prepare_dataset: bool
    prepare_features: bool
    preprocessing: dict[str, Any]
    head: dict[str, Any]
    postprocessing: dict[str, Any]
    metrics: dict[str, Any]
    output_dir: Path | None
    feature_dir: Path | None

    def to_manifest(self) -> dict[str, Any]:
        """Return the stable stage description written beside run artifacts."""
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "task": self.task,
            "dataset": self.dataset,
            "source_dataset": self.source_dataset,
            "model": self.model,
            "execution_mode": self.execution_mode,
            "trainer": self.trainer,
            "feature_experiment_config": (
                str(self.feature_experiment_config)
                if self.feature_experiment_config
                else None
            ),
            "prepare_dataset": self.prepare_dataset,
            "prepare_features": self.prepare_features,
            "stage_order": [
                "preprocessing",
                "data_loader",
                "feature_layers",
                "head",
                "postprocessing",
                "metrics",
            ],
            "stages": {
                "preprocessing": self.preprocessing,
                "data_loader": self.data_loader,
                "feature_layers": list(self.feature_layers),
                "head": self.head,
                "postprocessing": self.postprocessing,
                "metrics": self.metrics,
            },
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "feature_dir": str(self.feature_dir) if self.feature_dir else None,
        }


def load_pipeline_plan(config_path: str | Path) -> PipelinePlan:
    """Resolve the run YAML and referenced feature config into stage choices."""
    config_path = resolve_project_path(config_path)
    raw = _load_yaml(config_path)
    evaluation = _mapping(raw, "evaluation")
    experiment = _mapping(raw, "experiment")
    data = _mapping(raw, "data")
    head_config = _mapping(raw, "head")
    training = _mapping(raw, "training")
    run = _mapping(raw, "run")
    metrics = _mapping(raw, "metrics")

    task = str(evaluation.get("task") or raw.get("task") or "")
    dataset_name = str(evaluation.get("dataset") or experiment.get("dataset") or "")
    head_name = str(evaluation.get("head") or experiment.get("head") or "")
    decoder_mode = str(head_config.get("decoder_mode") or "").strip().lower()
    trainer = _resolve_trainer(task, head_name, decoder_mode)
    execution_mode = (
        "offline"
        if trainer == "offline"
        else "scoring"
        if trainer == "virtual_staining_score"
        else "online"
    )

    feature_config_path = data.get("feature_experiment_config")
    resolved_feature_path = (
        resolve_project_path(feature_config_path) if feature_config_path else None
    )
    prepare_features = bool(run.get("prepare_features", False))
    prepare_dataset = bool(run.get("prepare_dataset", False))
    if prepare_features and resolved_feature_path is None:
        raise ValueError(
            f"{config_path}: run.prepare_features requires "
            "data.feature_experiment_config"
        )
    feature_experiment: dict[str, Any] = {}
    if resolved_feature_path:
        feature_experiment = _mapping(_load_yaml(resolved_feature_path), "experiment")
    source_dataset = str(
        feature_experiment.get("dataset")
        or experiment.get("dataset")
        or dataset_name
    )
    model_name = str(experiment.get("model") or feature_experiment.get("model") or "") or None
    try:
        dataset_mode = get_dataset_config(source_dataset).mode if source_dataset else None
    except KeyError:
        dataset_mode = None

    data_loader = _resolve_loader(
        task=task,
        dataset_name=dataset_name,
        data=data,
        run=run,
        dataset_mode=dataset_mode,
    )
    configured_layers = (
        experiment.get("target_features")
        or feature_experiment.get("target_features")
        or data.get("feature_layer")
        or head_config.get("feature_layers")
        or ()
    )
    if isinstance(configured_layers, str):
        feature_layers = tuple(
            layer.strip() for layer in configured_layers.split(",") if layer.strip()
        )
    else:
        feature_layers = tuple(str(layer) for layer in configured_layers if layer)

    model_config = None
    if model_name:
        model_config_path = run.get("benchmark_config_path")
        if model_config_path:
            model_config_path = str(resolve_project_path(model_config_path))
        try:
            model_config = get_model_config(model_name, model_config_path)
        except KeyError:
            model_config = None

    model_preprocessing: dict[str, Any] = {}
    model_postprocessing: dict[str, Any] = {}
    if model_config is not None:
        model_preprocessing = {
            "normalization": model_config.norm.norm_type.value,
            "spatial": asdict(model_config.spatial),
            "mpp": asdict(model_config.mpp),
        }
        model_postprocessing = {
            "name": model_config.postprocess.name,
            "params": dict(model_config.postprocess.params),
        }

    target_mpp = experiment.get("target_mpp") or feature_experiment.get("target_mpp")
    target_patch_size = (
        experiment.get("target_patch_size")
        or feature_experiment.get("target_patch_size")
    )
    preprocessing = {
        "model_profile": model_preprocessing,
        "target_mpp": target_mpp,
        "target_patch_size": target_patch_size,
        "input_format": run.get("input_format", "manifest"),
        "augmentations": dict(training.get("augmentations") or {}),
        "prepare_dataset": prepare_dataset,
        "prepare_features": prepare_features,
    }
    postprocessing = {
        "task": task,
        "model_output": model_postprocessing,
        "config": dict(raw.get("postprocessing") or {}),
    }
    output_dir_raw = run.get("output_dir")
    output_dir = (
        relocate_lab_path(resolve_project_path(output_dir_raw, PREPATH_ROOT))
        if output_dir_raw
        else None
    )
    feature_dir_raw = run.get("feat_dir")
    feature_dir = (
        relocate_lab_path(resolve_project_path(feature_dir_raw, PREPATH_ROOT))
        if feature_dir_raw
        else None
    )

    return PipelinePlan(
        config_path=config_path,
        run_id=str(evaluation.get("id") or experiment.get("id") or config_path.stem),
        task=task,
        dataset=dataset_name,
        source_dataset=source_dataset,
        model=model_name,
        trainer=trainer,
        execution_mode=execution_mode,
        data_loader=data_loader,
        feature_layers=feature_layers,
        feature_experiment_config=resolved_feature_path,
        prepare_dataset=prepare_dataset,
        prepare_features=prepare_features,
        preprocessing=preprocessing,
        head={"name": head_name, "decoder_mode": decoder_mode, "config": head_config},
        postprocessing=postprocessing,
        metrics={
            "profile": task,
            "config": metrics,
            "result_filename": str(data.get("test_result_name") or "").strip() or None,
        },
        output_dir=output_dir,
        feature_dir=feature_dir,
    )
