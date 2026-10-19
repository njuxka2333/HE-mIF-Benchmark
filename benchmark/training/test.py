"""Run test-only inference for a trained downstream head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.config import EvalConfig, load_eval_config
from training.train import (
    _aggregate_metrics,
    _loss_fn,
    _make_dataset,
    _make_head,
    _validate_task_config,
    evaluate,
    set_seed,
)


def _resolve_device(cfg: EvalConfig, override: str | None) -> torch.device:
    value = override or str(
        cfg.training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return torch.device(value)


def run_test(
    cfg: EvalConfig,
    *,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
    device_override: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, float]:
    """Evaluate one trained checkpoint on a configured test split."""
    _validate_task_config(cfg)
    set_seed(int(cfg.training.get("seed", 42)))
    if batch_size is not None:
        cfg.training["batch_size"] = int(batch_size)
    if num_workers is not None:
        cfg.training["num_workers"] = int(num_workers)

    device = _resolve_device(cfg, device_override)
    test_split = split or str(cfg.data.get("test_split", "test"))
    checkpoint = Path(checkpoint_path) if checkpoint_path else cfg.output_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    ckpt_layers = state.get("feature_layers")
    if ckpt_layers is not None:
        cfg.head_config["feature_layers"] = list(ckpt_layers)

    test_dataset = _make_dataset(cfg, split=test_split, max_samples=None, feature_layers=ckpt_layers)
    model = _make_head(cfg, test_dataset).to(device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    criterion = _loss_fn(cfg, test_dataset.num_classes)

    return evaluate(
        cfg,
        model,
        criterion,
        device,
        split=test_split,
        num_classes=test_dataset.num_classes,
        label_dict=test_dataset.label_dict,
        feature_layers=list(test_dataset.feature_layers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test-only inference for a trained head")
    parser.add_argument("--config", required=True, type=Path, help="Run config.yaml or evaluation YAML")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to output_dir/best_model.pt")
    parser.add_argument("--split", default=None, help="Defaults to data.test_split or test")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    metrics = run_test(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        device_override=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"updated {cfg.output_dir / 'test_metrics.json'}")
    if cfg.task == "prognosis_survival":
        print(f"updated {cfg.output_dir / 'test_result' / 'patient_risk.csv'}")
    else:
        print(f"updated {cfg.output_dir / 'test_result' / cfg.data.get('test_result_name', 'pathocell_test_metrics.csv')}")
    print(f"summary {_aggregate_metrics(metrics)}")


if __name__ == "__main__":
    main()
