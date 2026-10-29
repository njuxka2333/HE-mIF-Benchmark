"""Fixed execution boundary and result manifest for a single benchmark run."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pipeline.plan import PipelinePlan


@dataclass(frozen=True)
class PipelineResult:
    """Stable outer result, independent of task-specific metric contents."""

    run_id: str
    task: str
    dataset: str
    model: str | None
    execution_mode: str
    state: str
    started_at: str
    finished_at: str
    output_dir: str | None
    artifacts: dict[str, str]
    metrics: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "task": self.task,
            "dataset": self.dataset,
            "model": self.model,
            "execution_mode": self.execution_mode,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "error": self.error,
        }


def _artifact_paths(plan: PipelinePlan) -> dict[str, str]:
    output_dir = plan.output_dir
    if output_dir is None or not output_dir.is_dir():
        return {}
    configured_metrics_name = plan.metrics.get("result_filename")
    candidates = {
        "metrics": (
            output_dir / "test_metrics.json",
            output_dir / "test_result" / "test_metrics.json",
            output_dir / "metrics.json",
            output_dir / "vstain_shared_metrics.json",
            output_dir / "mipheivit_vstain_metrics.json",
            output_dir / "test_result" / "vstain_shared_metrics.json",
            output_dir / "test_result" / "mipheivit_vstain_metrics.json",
            *(
                (
                    output_dir / configured_metrics_name,
                    output_dir / "test_result" / configured_metrics_name,
                )
                if configured_metrics_name
                else ()
            ),
        ),
        "predictions": (
            output_dir / "test_result" / "predictions.csv",
            output_dir / "predictions.csv",
        ),
        "checkpoint": (
            output_dir / "best_model.pt",
            output_dir / "best_checkpoint.pt",
            output_dir / "checkpoint.pt",
        ),
    }
    artifacts = {
        name: str(path)
        for name, paths in candidates.items()
        if (path := next((candidate for candidate in paths if candidate.is_file()), None))
    }
    if "metrics" not in artifacts:
        metric_files = sorted(
            path
            for path in output_dir.rglob("*metrics*.json")
            if path.name != "pipeline_manifest.json"
        )
        if metric_files:
            artifacts["metrics"] = str(metric_files[0])
    return artifacts


def _read_metrics(artifacts: dict[str, str]) -> dict[str, Any]:
    metrics_path = artifacts.get("metrics")
    if not metrics_path:
        return {}
    path = Path(metrics_path)
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            return {"rows": rows} if rows else {}
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, csv.Error):
        return {}
    return raw if isinstance(raw, dict) else {"value": raw}


def _write_manifest(plan: PipelinePlan, result: PipelineResult) -> Path | None:
    if plan.output_dir is None:
        return None
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    path = plan.output_dir / "pipeline_manifest.json"
    manifest = plan.to_manifest()
    manifest["result"] = result.to_dict()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_pipeline(
    plan: PipelinePlan,
    args: Any,
    execute_trainer: Callable[[Any], Any],
) -> PipelineResult:
    """Execute one configured trainer and emit the shared result contract."""
    started_at = datetime.now(timezone.utc).isoformat()
    state = "completed"
    error_message = None
    try:
        if plan.prepare_dataset:
            from preprocessing.build import main as build_dataset

            dataset_name = {
                "pathocell_online": "pathocell",
                "lizard_online": "lizard",
            }.get(plan.source_dataset, plan.source_dataset)
            print(f"Pipeline preprocessing dataset: {dataset_name}", flush=True)
            build_dataset(["--dataset", dataset_name])
        if plan.prepare_features and plan.feature_experiment_config is not None:
            from feature_extraction.extract import main as extract_features

            print(f"Pipeline preprocessing: {plan.feature_experiment_config}", flush=True)
            extract_features(
                ["--config", str(plan.feature_experiment_config), "--layers", "yaml"]
            )
        execute_trainer(args)
    except BaseException as error:
        state = "failed"
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        artifacts = _artifact_paths(plan)
        result = PipelineResult(
            run_id=plan.run_id,
            task=plan.task,
            dataset=plan.dataset,
            model=plan.model,
            execution_mode=plan.execution_mode,
            state=state,
            started_at=started_at,
            finished_at=finished_at,
            output_dir=str(plan.output_dir) if plan.output_dir else None,
            artifacts=artifacts,
            metrics=_read_metrics(artifacts),
            error=error_message,
        )
        manifest_path = _write_manifest(plan, result)
        if manifest_path is not None:
            print(f"Pipeline manifest: {manifest_path}")
    return result
