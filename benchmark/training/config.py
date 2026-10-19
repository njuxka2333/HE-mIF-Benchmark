"""Config loading for downstream evaluation pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.paths import PREPATH_ROOT, relocate_lab_path, resolve_project_path


@dataclass
class EvalConfig:
    id: str
    task: str
    dataset: str
    head: str
    output_dir: Path
    data: dict[str, Any] = field(default_factory=dict)
    head_config: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    survival: dict[str, Any] = field(default_factory=dict)


def resolve_path(value: str | Path, base: Path = PREPATH_ROOT) -> Path:
    return relocate_lab_path(resolve_project_path(value, base))


def load_eval_config(path: str | Path) -> EvalConfig:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    evaluation = raw.get("evaluation") or {}
    run = raw.get("run") or {}
    eval_id = str(evaluation.get("id") or config_path.stem)
    output_dir = resolve_path(run.get("output_dir", f"../Evaluation/{eval_id}"))

    data = dict(raw.get("data") or {})
    if "feature_experiment_config" in data:
        data["feature_experiment_config"] = str(
            resolve_path(data["feature_experiment_config"])
        )
    for path_key in ("split_csv", "selection_csv"):
        if path_key in data:
            data[path_key] = str(resolve_path(data[path_key]))
    cv = data.get("cross_validation")
    if isinstance(cv, dict) and cv.get("split_csv_dir"):
        cv = dict(cv)
        cv["split_csv_dir"] = str(resolve_path(cv["split_csv_dir"]))
        data["cross_validation"] = cv

    training = dict(raw.get("training") or {})
    optimizer = dict(training.get("optimizer") or {})
    optimizer_params = dict(optimizer.get("params") or {})
    for key in ("lr", "weight_decay", "eps"):
        value = optimizer_params.get(key)
        if isinstance(value, str):
            try:
                optimizer_params[key] = float(value)
            except ValueError as error:
                raise ValueError(
                    f"training.optimizer.params.{key} must be numeric, got {value!r}"
                ) from error
    if optimizer:
        optimizer["params"] = optimizer_params
        training["optimizer"] = optimizer

    return EvalConfig(
        eval_id,
        str(evaluation["task"]),
        str(evaluation["dataset"]),
        str(evaluation["head"]),
        output_dir,
        data,
        dict(raw.get("head") or {}),
        training,
        dict(raw.get("metrics") or {}),
        dict(raw.get("survival") or {}),
    )
