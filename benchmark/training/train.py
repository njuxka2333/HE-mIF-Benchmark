"""Unified training entry point.

Every reported downstream YAML is started the same way::

    python -m training.train --config configs/downstream/...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from pipeline.plan import ONLINE_TASKS, load_pipeline_plan  # noqa: E402
from pipeline.runner import run_pipeline  # noqa: E402

CPU_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


# The router stays independent from trainer internals. Offline helpers are
# imported explicitly for the test-only and PCA paths that still share them.
from training.offline_train import (  # noqa: E402
    HEAD_REGISTRY,
    _aggregate_metrics,
    _classwise_test_rows,
    _configure_torch_threads,
    _cross_validation_folds,
    _eval_linear_probe,
    _loss_fn,
    _make_dataset,
    _make_head,
    _make_optimizer,
    _make_scheduler,
    _log,
    _prepare_output_dirs,
    _resolve_cv_split_csv,
    _validate_task_config,
    _write_linear_probe_test_outputs,
    _write_metrics,
    _write_test_csv,
    evaluate,
    set_seed,
)

from training.pca_features import install_pca_hook  # noqa: E402

install_pca_hook()


def _load_config_dict(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _config_task(config_path: Path) -> str:
    raw = _load_config_dict(config_path)
    evaluation = raw.get("evaluation") or {}
    return str(evaluation.get("task") or raw.get("task") or "")


def _config_route(config_path: Path) -> str:
    """Pick the trainer from task + decoder_mode, not from a scripts/ wrapper."""
    return load_pipeline_plan(config_path).trainer


def _normalise_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "-test-only",
        "--test-only",
        "--eval-only",
        dest="test_only",
        action="store_true",
        help="Evaluate an existing checkpoint/best_model instead of training.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Offline-only split to evaluate when --test-only is set.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Offline-only seed override.")
    parser.add_argument("--fold", type=int, default=None, help="Offline-only CV fold override.")
    parser.add_argument("--device", default=None, help="Online-only device override.")
    parser.add_argument("--train-split", default=None, help="Online-only train split override.")
    parser.add_argument("--valid-split", default=None, help="Online-only validation split override.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Online-only checkpoint for --test-only.",
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help=(
            "Virtual-staining (cell_niches) only: CD3 shuffled-pairing null for "
            "Spearman and AUPRC, plus FOV-bootstrap AUPRC. Spearman has no CI. "
            "Can also be turned on per-config via training.significance: true."
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help="Virtual-staining significance only: bootstrap resamples (overrides config/default).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=None,
        help="Virtual-staining significance only: shuffled-null permutations (overrides config/default).",
    )
    parser.add_argument(
        "--sig-seed",
        type=int,
        default=None,
        help="Virtual-staining significance only: RNG seed (overrides config/default).",
    )
    return parser.parse_args(argv)


def _limit_cpu_threads() -> None:
    n = max(1, int(os.environ.get("VSTAIN_CPU_THREADS", "4")))
    for key in CPU_THREAD_KEYS:
        os.environ[key] = str(n)
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(min(2, n))
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass


def _run_offline(args: argparse.Namespace) -> None:
    from training.offline_train import main as _offline_main

    forwarded = ["--config", str(args.config)]
    if args.test_only:
        forwarded.append("--eval-only")
    if args.eval_split:
        forwarded.extend(["--eval-split", args.eval_split])
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.fold is not None:
        forwarded.extend(["--fold", str(args.fold)])
    sys.argv = ["offline_train.py", *forwarded]
    _offline_main()


def _run_online(args: argparse.Namespace) -> None:
    from training.online_segmentation_train import main as online_main

    forwarded = ["--config", str(args.config)]
    if args.test_only:
        forwarded.append("--test-only")
    if args.device:
        forwarded.extend(["--device", args.device])
    if args.train_split:
        forwarded.extend(["--train-split", args.train_split])
    if args.valid_split:
        forwarded.extend(["--valid-split", args.valid_split])
    if args.checkpoint:
        forwarded.extend(["--checkpoint", str(args.checkpoint)])
    sys.argv = ["online_segmentation_train.py", *forwarded]
    online_main()


def _run_virtual_staining_unet(args: argparse.Namespace) -> None:
    _limit_cpu_threads()
    from virtual_staining.train_unet import main as train_main

    train_main(["--config", str(args.config)])


def _run_virtual_staining_feature(args: argparse.Namespace) -> None:
    _limit_cpu_threads()
    from virtual_staining.train_feature import main as train_main

    train_main(["--config", str(args.config)])


def _significance_forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if getattr(args, "significance", False):
        forwarded.append("--significance")
    if getattr(args, "n_boot", None) is not None:
        forwarded.extend(["--n-boot", str(args.n_boot)])
    if getattr(args, "n_perm", None) is not None:
        forwarded.extend(["--n-perm", str(args.n_perm)])
    if getattr(args, "sig_seed", None) is not None:
        forwarded.extend(["--sig-seed", str(args.sig_seed)])
    return forwarded


def _run_virtual_staining_score(args: argparse.Namespace) -> None:
    from virtual_staining.score import main as score_main

    score_main(["--config", str(args.config), *_significance_forwarded_args(args)])


_ROUTERS = {
    "offline": _run_offline,
    "online": _run_online,
    "virtual_staining_unet": _run_virtual_staining_unet,
    "virtual_staining_feature": _run_virtual_staining_feature,
    "virtual_staining_score": _run_virtual_staining_score,
}


def main(argv: list[str] | None = None) -> None:
    args = _normalise_args(sys.argv[1:] if argv is None else argv)
    from feature_extraction.frozen_hoptimus import apply_extract_env_from_config

    apply_extract_env_from_config(args.config)
    plan = load_pipeline_plan(args.config)
    try:
        trainer = _ROUTERS[plan.trainer]
    except KeyError as exc:
        raise SystemExit(
            f"unknown trainer {plan.trainer!r} for pipeline {plan.run_id!r}"
        ) from exc
    print(
        f"Pipeline: {plan.run_id} | {plan.execution_mode} | "
        f"loader={plan.data_loader} | head={plan.head['name']}"
    )
    run_pipeline(plan, args, trainer)


if __name__ == "__main__":
    main()
