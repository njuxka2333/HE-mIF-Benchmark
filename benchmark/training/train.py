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

ONLINE_TASKS = {"cell_semantic_segmentation", "tls_region_segmentation"}
CPU_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def _hydrate_training_symbols_from_pyc() -> None:
    """Online bytecode imports helpers from ``training.train``; keep router, fill gaps from .pyc."""
    if globals().get("_TRAIN_HYDRATE_ACTIVE") or globals().get("_aggregate_metrics") is not None:
        return
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        return
    import importlib.util

    from training._bytecode_pick import pick_pyc, pyc_magic_matches

    globals()["_TRAIN_HYDRATE_ACTIVE"] = True
    try:
        here = Path(__file__).resolve().parent
        if here.name == "__pycache__":
            here = here.parent
        pyc = here / "_orig_pyc" / "train.pyc"
        if not pyc.is_file() or not pyc_magic_matches(pyc):
            pyc = pick_pyc("train", here, min_size=2000)
        modname = "_bytecode_train_helpers"
        spec = importlib.util.spec_from_file_location(modname, pyc)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        keep = {
            "main",
            "_run_offline",
            "_run_online",
            "_run_mif_unet",
            "_run_mif_feature",
            "_run_mif_score",
            "_run_vstain",
            "_run_cell_niches",
            "_run_cell_niches_mipheivit",
            "_normalise_args",
            "_config_task",
            "_config_route",
            "_load_config_dict",
            "_limit_cpu_threads",
            "_ROUTERS",
            "ONLINE_TASKS",
            "CPU_THREAD_KEYS",
            "BENCHMARK_ROOT",
            "_hydrate_training_symbols_from_pyc",
            "_TRAIN_HYDRATE_ACTIVE",
            "argparse",
            "os",
            "sys",
            "Path",
            "yaml",
            "__name__",
            "__doc__",
            "__file__",
            "__package__",
            "__cached__",
            "__spec__",
        }
        for key, value in vars(mod).items():
            if key in keep or key.startswith("__"):
                continue
            globals()[key] = value
    finally:
        globals().pop("_TRAIN_HYDRATE_ACTIVE", None)


_hydrate_training_symbols_from_pyc()

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
    raw = _load_config_dict(config_path)
    evaluation = raw.get("evaluation") or {}
    head = raw.get("head") or {}
    task = str(evaluation.get("task") or raw.get("task") or "")
    eval_head = str(evaluation.get("head") or "")
    mode = str(head.get("decoder_mode") or "").strip().lower()
    if task == "virtual_staining" or eval_head == "virtual_stain":
        return "mif_score"
    if mode == "mif_unet":
        return "mif_unet"
    if mode == "mif_feature":
        return "mif_feature"
    if task in ONLINE_TASKS:
        return "online"
    return "offline"


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


def _run_mif_unet(args: argparse.Namespace) -> None:
    _limit_cpu_threads()
    from mif.train_unet import main as mif_main

    mif_main(["--config", str(args.config)])


def _run_mif_feature(args: argparse.Namespace) -> None:
    _limit_cpu_threads()
    from mif.train_feature import main as mif_main

    mif_main(["--config", str(args.config)])


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


def _run_mif_score(args: argparse.Namespace) -> None:
    from mif.score import main as score_main

    score_main(["--config", str(args.config), *_significance_forwarded_args(args)])


_run_vstain = _run_mif_score
_run_cell_niches = _run_mif_score
_run_cell_niches_mipheivit = _run_mif_score


_ROUTERS = {
    "offline": _run_offline,
    "online": _run_online,
    "mif_unet": _run_mif_unet,
    "mif_feature": _run_mif_feature,
    "mif_score": _run_mif_score,
}


def main(argv: list[str] | None = None) -> None:
    args = _normalise_args(sys.argv[1:] if argv is None else argv)
    from feature_extraction.hoptimus0 import apply_extract_env_from_config

    apply_extract_env_from_config(args.config)
    route = _config_route(args.config)
    try:
        runner = _ROUTERS[route]
    except KeyError as exc:
        raise SystemExit(f"unknown train route {route!r} for {args.config}") from exc
    runner(args)


if __name__ == "__main__":
    main()
