"""One entrance for a whole task: train every arm, then post-process them together.

``training.train`` runs one config. ``training.postprocess`` turns a finished
set of runs into the reported statistics. Virtual staining, TLS, and cell
semantic segmentation then write the paper example tiles. Neither is the unit a
person actually works in - the unit is a *task*: all arms of a downstream task,
then the comparison across them. This module is that unit, so a task is
reproduced with one command instead of a hand-written loop plus a remembered
follow-up::

    python -m training.pipeline --task virtual_staining --significance
    python -m training.pipeline --task prognosis --models uni2
    python -m training.pipeline --all

Why postprocessing cannot live inside ``training.train``: every statistic it
produces is cross-run. DeLong's test and the paired bootstrap compare a model
against UNI2 *on the same cells*; the fold t-interval aggregates five folds; the
comparability audit checks that all six arms share one evaluation grid. A single
run has none of that in scope. So the pipeline is sequenced here, one level up,
where the whole set is in view.

``--models`` and ``--route`` narrow a run to one arm, which is the normal way to
work: launch a model, and when it lands the statistics are rebuilt from *every*
arm already on disk, not only the one just trained. So six sequential
single-model runs end in the same table as one six-arm run, and a partial task
gives a partial table that says which arms it covered.

Post-processing is not optional and has no entrance of its own: a task is not
finished until its arms have been compared, so :mod:`training.postprocess` is a
library this module calls, never a command a person runs against stale results.
``--test-only`` re-evaluates from existing checkpoints instead of retraining,
which is how a repaired arm is brought back without touching the others.

Configs are discovered from ``configs/downstream/<dir>`` - the same YAMLs
``training.train`` takes - so this adds no second source of truth about what a
task consists of.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from training import postprocess as _postprocess  # noqa: E402
from training import train as _train  # noqa: E402
from training.pca_features import install_pca_hook  # noqa: E402
from utils.paths import CONFIG_DIR, REPO_ROOT  # noqa: E402

install_pca_hook()

# task -> (config subdirectory, postprocessor key or None)
TASKS = {
    "single_cell_classification": ("linear_probe", "single_cell_classification"),
    "single_cell_classification_hoptimus": ("linear_probe_hoptimus", "single_cell_classification"),
    "single_cell_classification_pca": ("linear_probe_pca", "single_cell_classification"),
    "prognosis": ("prognosis", "prognosis"),
    "prognosis_pca": ("prognosis_pca", "prognosis"),
    "virtual_staining": ("virtual_staining", "virtual_staining"),
    "cell_semantic_segmentation": ("cellseg", "cell_semantic_segmentation"),
    "cell_semantic_segmentation_hoptimus": ("cellseg_hoptimus", "cell_semantic_segmentation"),
    "tls_region_segmentation": ("tls", "tls_region_segmentation"),
    "tls_region_segmentation_hoptimus": ("tls_hoptimus", "tls_region_segmentation"),
}

# Encoder-width and frozen-backbone controls; not part of the paper table.
SUPPLEMENTARY_TASKS = {
    "single_cell_classification_hoptimus",
    "single_cell_classification_pca",
    "prognosis_pca",
    "cell_semantic_segmentation_hoptimus",
    "tls_region_segmentation_hoptimus",
}

SUPPLEMENTARY_RESULTS_ROOT = {
    "single_cell_classification_hoptimus": "scc_hoptimus",
    "single_cell_classification_pca": "scc_pca",
    "prognosis_pca": "prognosis_pca",
    "cell_semantic_segmentation_hoptimus": "cellseg_hoptimus",
    "tls_region_segmentation_hoptimus": "tls_hoptimus",
}

# Order matters only for --all: cheap audits first, so a broken evaluation grid
# surfaces before the long bootstrap runs.
TASK_ORDER = [
    "virtual_staining",
    "cell_semantic_segmentation",
    "tls_region_segmentation",
    "single_cell_classification",
    "prognosis",
]


def _stem_matches_models(stem: str, models: list[str]) -> bool:
    """``uni2`` matches ``uni2.yaml``; an exact stem matches only itself."""
    wanted = {m.lower() for m in models}
    low = stem.lower()
    return low in wanted or any(low.startswith(f"{m}_") for m in wanted)


def discover_configs(task: str, models: list[str] | None = None,
                     routes: list[str] | None = None) -> list[Path]:
    """Every downstream YAML belonging to ``task``, sorted, optionally filtered.

    An arm is addressed as model x route: the config's stem is the model
    (``uni2.yaml``) and its parent directory is the route (``encoder`` or
    ``output``), which is how the configs are laid out for every task.
    """
    subdir, _ = TASKS[task]
    root = CONFIG_DIR / "downstream" / subdir
    configs = sorted(root.rglob("*.yaml"))
    if models:
        configs = [c for c in configs if _stem_matches_models(c.stem, models)]
    if routes:
        wanted_routes = {r.lower() for r in routes}
        configs = [c for c in configs if c.parent.name.lower() in wanted_routes]
    return configs


def _train_forwarded(args) -> list[str]:
    """Flags ``training.train`` accepts that are worth exposing at task level."""
    out: list[str] = []
    if args.test_only:
        out.append("--test-only")
    if args.significance:
        out.append("--significance")
    for flag, value in (("--n-boot", args.n_boot), ("--n-perm", args.n_perm),
                        ("--sig-seed", args.seed), ("--device", args.device)):
        if value is not None:
            out.extend([flag, str(value)])
    return out


def task_results_root(task: str, requested: Path) -> Path:
    """Paper tables and supplementary PCA / H-optimus arms live in ``results/``.

    An explicit ``--results-root`` wins.
    """
    if requested != _postprocess.DEFAULT_RESULTS_ROOT:
        return requested
    if task in SUPPLEMENTARY_RESULTS_ROOT:
        return _postprocess.DEFAULT_RESULTS_ROOT / SUPPLEMENTARY_RESULTS_ROOT[task]
    return requested


def run_task(task: str, args) -> dict:
    configs = discover_configs(task, args.models, args.routes)
    total = len(discover_configs(task))
    narrowed = bool(args.models or args.routes)
    scope = f"{len(configs)} of {total} arms" if narrowed else f"{len(configs)} configs"
    print(f"\n{'=' * 72}\n{task}  ({scope})\n{'=' * 72}")
    if narrowed and not configs:
        raise SystemExit(f"no {task} config matches "
                         f"--models {' '.join(args.models or ['*'])} "
                         f"--route {' '.join(args.routes or ['*'])}")
    trained, failed = [], []
    orig_results_root = args.results_root
    args.results_root = task_results_root(task, orig_results_root)
    try:
        forwarded = _train_forwarded(args)
        for i, config in enumerate(configs, 1):
            rel = config.relative_to(REPO_ROOT)
            print(f"\n[{i}/{len(configs)}] train {rel}", flush=True)
            if args.dry_run:
                print("    (dry-run)")
                continue
            started = time.time()
            try:
                _train.main(["--config", str(config), *forwarded])
                trained.append(str(rel))
                print(f"    done in {time.time() - started:.0f}s", flush=True)
            except SystemExit as exc:          # argparse/guard failures carry a message
                failed.append((str(rel), str(exc)))
                print(f"    FAILED: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
            except Exception as exc:           # noqa: BLE001 - one arm must not kill the task
                failed.append((str(rel), repr(exc)))
                print(f"    FAILED: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise

        if failed:
            print(f"\n  {len(failed)} run(s) failed:")
            for rel, why in failed:
                print(f"    {rel}: {why}")

        _, key = TASKS[task]
        if key and not args.dry_run:
            print(f"\n--- postprocess: {task} ---")
            if narrowed:
                print("    (statistics are rebuilt from every arm on disk, not only the ones just run)")
            _postprocess._POSTPROCESSORS[key](args)
        if not args.dry_run and task in {
            "virtual_staining",
            "cell_semantic_segmentation",
            "tls_region_segmentation",
        }:
            from training.example_tiles import render_task_example_tiles

            print(f"\n--- example tiles: {task} ---")
            try:
                render_task_example_tiles(task, args)
            except Exception as exc:  # noqa: BLE001
                failed.append(("example_tiles", repr(exc)))
                print(f"    FAILED: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise
        return {"task": task, "n_configs": len(configs), "trained": trained, "failed": failed}
    finally:
        args.results_root = orig_results_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), help="task to run end to end")
    parser.add_argument("--all", action="store_true", help="every task, in TASK_ORDER")
    parser.add_argument("--models", nargs="+", default=None,
                        help="restrict to these models, e.g. --models uni2 mipheivit")
    parser.add_argument("--route", "--routes", dest="routes", nargs="+", default=None,
                        choices=["encoder", "output"],
                        help="restrict to one route: encoder (frozen features) or "
                             "output (features off the generated mIF image)")
    parser.add_argument("--test-only", action="store_true",
                        help="evaluate from existing checkpoints instead of retraining")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going when one arm fails, and report failures at the end")
    parser.add_argument("--dry-run", action="store_true", help="list what would run")
    parser.add_argument("--device", default=None)
    # forwarded to the virtual-staining scorer
    parser.add_argument("--significance", action="store_true",
                        help="virtual staining: compute the FOV-level bootstrap during scoring")
    parser.add_argument("--n-perm", type=int, default=None)
    # shared with training.postprocess
    parser.add_argument("--results-root", type=Path, default=_postprocess.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--only", nargs="+", metavar="DATASET/ROUTE", default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--max-boot-rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=_postprocess.DEFAULT_SEED)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not args.task and not args.all:
        parser.error("pass --task or --all")
    if args.n_boot is None:
        args.n_boot = _postprocess.DEFAULT_N_BOOT

    tasks = TASK_ORDER if args.all else [args.task]
    summaries = [run_task(task, args) for task in tasks]

    print(f"\n{'=' * 72}\nsummary\n{'=' * 72}")
    for s in summaries:
        state = f"{len(s['failed'])} failed" if s["failed"] else "ok"
        print(f"  {s['task']:<30}{s['n_configs']:>3} configs   {state}")
    if any(s["failed"] for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
