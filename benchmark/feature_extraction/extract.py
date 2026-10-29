"""Unified feature extraction entry point.

Experiment YAML chooses the dataset, model layers, geometry, and output layout.
TCGA cross-validation bags are placed into ``fold_*`` after extraction.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from feature_extraction.fold_placement import (
    cleanup_unique_wsi_files,
    place_layer,
    wsi_bag_complete,
    write_unique_selection_csv,
)
from feature_extraction.protocol import (
    dataset_protocol,
    load_protocol,
    model_protocol,
    parse_layer_tokens,
    protocol_layer_feat_dirs,
    run_id,
    run_root_from_feat_dir,
    set_protocol_yaml,
)
from utils.experiment_config import EXPERIMENTS_DIR, ExperimentConfig, load_experiment_config


def _feature_engine():
    from feature_extraction import engine

    return engine


def _find_experiment_yaml(dataset: str, model: str) -> Path:
    proto = load_protocol()
    ds = dataset_protocol(dataset, proto)
    spec = model_protocol(model, proto)
    rid = run_id(dataset, model, proto)
    candidates = [
        EXPERIMENTS_DIR / f"{rid}.yaml",
        EXPERIMENTS_DIR / f"abmil_{dataset}_{model}_{spec.wsi_tag}.yaml",
        EXPERIMENTS_DIR / f"abmil_{dataset}_{model}.yaml",
        EXPERIMENTS_DIR / f"linear_probe_{dataset}_{model}_{spec.lp_tag}.yaml",
        EXPERIMENTS_DIR / f"linear_probe_{dataset}_{model}.yaml",
    ]
    if ds.kind == "linear_probe":
        candidates.insert(0, EXPERIMENTS_DIR / f"linear_probe_{dataset}_{model}_{spec.lp_tag}.yaml")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No experiment YAML for dataset={dataset} model={model}; tried "
        + ", ".join(str(p.name) for p in candidates)
    )


def apply_protocol_layers(
    exp: ExperimentConfig,
    layers: Sequence[str],
    *,
    feat_root: Path | None = None,
) -> list[Path]:
    """Point ``exp`` at protocol encoder_*/output_layers roots for ``layers``."""
    proto = load_protocol()
    run_root = run_root_from_feat_dir(Path(exp.feat_dir))
    if feat_root is not None:
        run_root = Path(feat_root) / run_id(exp.dataset, exp.model, proto)
    mapping = protocol_layer_feat_dirs(run_root, exp.model, layers, proto)
    exp.target_features = list(layers)
    exp.feat_dir = mapping[layers[0]]
    # Split layouts are relative to the first configured layer directory
    # (which becomes feat_dir), not the parent run directory.
    exp.feat_dir_base = mapping[layers[0]]
    if len(layers) > 1:
        exp.split_layer_feat_dirs = True
        exp.layer_feat_dirs = mapping
    else:
        exp.split_layer_feat_dirs = False
        exp.layer_feat_dirs = None
    return [mapping[layer] for layer in layers]


def _yaml_extract_block(config_path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    block = raw.get("extract") or {}
    return block if isinstance(block, dict) else {}


def _apply_extract_runtime(yaml_extract: dict[str, Any]) -> None:
    """YAML ``extract.env`` plus the frozen H-optimus-0 skip of MIPHEI LoRA weights."""
    os.environ.pop("MIPHEI_SKIP_GENERATOR_CKPT", None)
    env = yaml_extract.get("env")
    if isinstance(env, dict):
        for key, value in env.items():
            os.environ[str(key)] = "" if value is None else str(value)
    from feature_extraction.frozen_hoptimus import frozen_hoptimus_requested, configure_frozen_hoptimus

    if frozen_hoptimus_requested():
        configure_frozen_hoptimus()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract features into the Pre_Features protocol layout"
    )
    parser.add_argument("--config", type=Path, help="Experiment YAML (encoder or mif).")
    parser.add_argument("--dataset", type=str, help="Dataset name (alternative to --config).")
    parser.add_argument("--model", type=str, help="Model name (with --dataset).")
    parser.add_argument(
        "--layers",
        default=None,
        help="encoder, output, all, yaml, or comma-separated layer names. "
        "Default: layers listed in the experiment YAML, otherwise all.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--target_patch_size", type=int, default=-1)
    parser.add_argument("--feat-dir", type=Path, default=None, help="Override run root or layer dir.")
    parser.add_argument("--feat-root", type=Path, default=None, help="Override Pre_Features root.")
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Pilot asset CSV containing sample_id and split; only listed samples are extracted.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split filter: train|valid|val|test|all. Linear-probe default: all three subfolders. "
        "WSI CV default: all unique slides then place into fold_0..fold_4.",
    )
    parser.add_argument("--feature-dtype", type=str, default=None, choices=("float16", "float32"))
    parser.add_argument("--mask-dtype", type=str, default=None, choices=("uint8", "int32"))
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract all samples (default: resume — skip complete outputs)",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    gap = parser.add_mutually_exclusive_group()
    gap.add_argument("--use-gap", action="store_true", default=None)
    gap.add_argument("--no-gap", action="store_true")
    parser.add_argument(
        "--no-place-folds",
        action="store_true",
        help="Skip WSI fold placement (leave unique bags under h5_files/).",
    )
    parser.add_argument("--protocol", type=Path, default=None, help="Override protocol.yaml")
    return parser


def _apply_runtime_overrides(exp: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.feat_dir is not None:
        exp.feat_dir = Path(args.feat_dir).resolve()
        exp.feat_dir_base = exp.feat_dir
    if args.batch_size is not None:
        exp.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        exp.iterator_cpus = int(args.num_workers)
    if args.feature_dtype:
        exp.feature_dtype = str(args.feature_dtype)
    if args.mask_dtype:
        exp.mask_dtype = str(args.mask_dtype)
    if args.use_gap:
        exp.use_gap = True
    if args.no_gap:
        exp.use_gap = False
    if args.target_patch_size is not None and int(args.target_patch_size) > 0:
        exp.target_patch_size = int(args.target_patch_size)


def _install_fold_complete_check(engine, layer_roots: Sequence[Path], model: str) -> None:
    orig = getattr(engine, "_is_complete_output", None)
    if orig is None:
        return

    def _is_complete_output(exp, sample_id, model_name=None):
        name = model_name or getattr(exp, "model", model)
        if all(
            wsi_bag_complete(Path(root), str(name), str(sample_id))
            for root in layer_roots
        ):
            return True
        return bool(orig(exp, sample_id, model_name) if model_name is not None else orig(exp, sample_id))

    engine._is_complete_output = _is_complete_output


def _run_engine(
    exp: ExperimentConfig,
    args: argparse.Namespace,
    *,
    wsi_cv: bool,
    ds=None,
) -> int:
    engine = _feature_engine()
    device_mod = __import__("torch")
    device = device_mod.device("cuda" if device_mod.cuda.is_available() else "cpu")
    cli_split = args.split
    if wsi_cv and cli_split is None:
        cli_split = "all"
    if ds is not None and ds.kind == "wsi_split" and cli_split is None:
        runs = [(split, split) for split in ds.splits]
    else:
        runs = engine.plan_split_runs(cli_split, exp.split_filter)
    dcfg = exp.dataset_config
    base_feat_dir = Path(exp.feat_dir)
    print(
        f"Experiment: {exp.id}\n"
        f"  head={exp.head} model={exp.model} dataset={exp.dataset} use_gap={exp.use_gap}\n"
        f"  storage={exp.feature_storage} feature_dtype={exp.feature_dtype} mask_dtype={exp.mask_dtype}\n"
        f"  layers={list(exp.target_features)}\n"
        f"  dataset_dir={dcfg.dataset_dir} split_level={dcfg.split_level.value}\n"
        f"  output={base_feat_dir}\n"
        f"  split plan: {', '.join(repr(r) for r in runs)}"
    )
    if int(getattr(args, "num_shards", 1) or 1) > 1:
        print(f"  parallel shard: {args.shard_id}/{args.num_shards}")

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = getattr(exp, "iterator_cpus", None)
        if num_workers is None:
            num_workers = getattr(exp, "num_workers", None)
    if num_workers is None:
        num_workers = 0
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = getattr(exp, "batch_size", None)
    engine_args = SimpleNamespace(
        target_patch_size=int(args.target_patch_size),
        batch_size=batch_size,
        skip_existing=not bool(args.no_skip_existing),
        selection_csv=getattr(args, "selection_csv", None),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        num_workers=int(num_workers),
    )
    model, transform = engine._build_model(exp, device)
    total = 0
    for split_filter, subdir in runs:
        if subdir:
            exp.feat_dir = base_feat_dir / subdir
        else:
            exp.feat_dir = base_feat_dir
        total += int(engine.run_extraction(exp, model, transform, device, engine_args, split_filter))
    exp.feat_dir = base_feat_dir
    return total


def _place_and_cleanup(exp: ExperimentConfig, layer_roots: Sequence[Path], *, place: bool) -> None:
    ds = dataset_protocol(exp.dataset)
    if ds.kind != "wsi_cv" or not place:
        return
    for root in layer_roots:
        stats = place_layer(Path(root), exp.model, ds)
        removed = cleanup_unique_wsi_files(Path(root), exp.model)
        print(
            f"  placed {root.name}: linked={stats['linked']} already={stats['already']} "
            f"copied={stats['copied']} missing={len(stats['missing_slides'])} cleaned_unique={removed}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.protocol:
        set_protocol_yaml(args.protocol)

    if args.config is None:
        if not args.dataset or not args.model:
            parser.error("provide --config, or both --dataset and --model")
        args.config = _find_experiment_yaml(args.dataset, args.model)

    exp = load_experiment_config(args.config)
    yaml_extract = _yaml_extract_block(Path(args.config))
    _apply_extract_runtime(yaml_extract)
    layer_token: Any = args.layers
    if layer_token is None:
        layer_token = yaml_extract.get("layers", "all")
    layers = parse_layer_tokens(layer_token, exp.model, yaml_layers=exp.target_features)
    if not layers:
        raise SystemExit("No layers resolved for extraction")

    _apply_runtime_overrides(exp, args)
    feat_root = Path(args.feat_root).resolve() if args.feat_root else None
    layer_roots = apply_protocol_layers(exp, layers, feat_root=feat_root)

    ds = dataset_protocol(exp.dataset)
    if ds.online:
        raise SystemExit(
            f"{exp.dataset} is an online task (TLS/PathoCell/Lizard/Cell-niches); "
            "it has no Pre_Features bags."
        )

    wsi_cv = ds.kind == "wsi_cv"
    selection_tmp: Path | None = None
    if wsi_cv and args.selection_csv is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{exp.id}_slides_",
            suffix=".csv",
            delete=False,
            mode="w",
            newline="",
            encoding="utf-8",
        )
        tmp.close()
        selection_tmp = write_unique_selection_csv(ds, Path(tmp.name))
        args.selection_csv = selection_tmp

    engine = _feature_engine()
    if wsi_cv:
        _install_fold_complete_check(engine, layer_roots, exp.model)

    try:
        _run_engine(exp, args, wsi_cv=wsi_cv, ds=ds)
        place = (not args.no_place_folds) and bool(yaml_extract.get("place_folds", True))
        _place_and_cleanup(exp, layer_roots, place=place)
    except FileNotFoundError as exc:
        dcfg = exp.dataset_config
        raise SystemExit(
            f"No samples for dataset '{exp.dataset}' under {dcfg.dataset_dir}\n"
            "Check configs/datasets.yaml: dataset_dir, split, split_level"
        ) from exc
    finally:
        if selection_tmp is not None:
            try:
                selection_tmp.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
