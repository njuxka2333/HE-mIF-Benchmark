#!/usr/bin/env python3
"""Build or preview materialized HookNet-TLS fixed-grid patch datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from data.hooknet_tls import (  # noqa: E402
    DEFAULT_PATCH_SIZE_L0,
    apply_manifest_selection,
    build_slide_patch_plan,
    collect_split_candidate_records,
    format_split_manifest_summary,
    load_fixed_grid_build_config,
    load_selected_tiles_csv,
    load_slide_row_from_split,
    make_overlay,
    manifest_determinism_hash,
    materialize_selected_records,
    rasterize_patch_mask,
    read_wsi_patch_level0,
    render_slide_overview_png,
    resolve_slide_paths,
    select_zoom_example_records,
    slide_plan_stub_for_materialize,
    summarize_patch_plan,
    write_manifest_summary_json,
    write_selected_tiles_csv,
)
from data.loader import get_dataset_config  # noqa: E402
from PIL import Image  # noqa: E402


def _default_split_csv(hooknet_root: Path) -> Path:
    from utils.paths import SPLITS_ROOT

    official = SPLITS_ROOT / "tls_seg" / "hooknet-tls_split.csv"
    if official.is_file():
        return official
    return hooknet_root / "hooknet-tls_split.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HookNet-TLS fixed-grid patch preprocessing")
    parser.add_argument(
        "--mode",
        choices=("preview", "extract"),
        required=True,
        help="preview: QA artifacts only; extract: write PNG patches + split manifests",
    )
    parser.add_argument("--slide-id", default=None, help="Single slide (required for preview)")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="hooknet-tls_split.csv (default: splits/tls_seg/hooknet-tls_split.csv)",
    )
    parser.add_argument(
        "--hooknet-root",
        type=Path,
        default=Path("/data3/houjunlin/fxx/datasets/HookNet_TLS"),
    )
    parser.add_argument("--dataset", default="hooknet_tls")
    parser.add_argument("--data-config", type=Path, default=BENCHMARK_ROOT / "configs/datasets.yaml")
    parser.add_argument(
        "--build-config",
        type=Path,
        default=BENCHMARK_ROOT / "configs/hooknet_tls.yaml",
        help="YAML with rest_keep_fraction and random_seed",
    )
    parser.add_argument(
        "--rest-keep-fraction",
        type=float,
        default=None,
        help="Override YAML rest_keep_fraction (0–1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override YAML random_seed for Rest-only subsampling",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE_L0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="openslide")
    parser.add_argument("--max-zoom-examples", type=int, default=6)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip patches whose image+mask PNGs already exist (default: true)",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        help="PNG zlib level 0-9 (1=fast lossless; default was PIL 6 which is slow)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print materialize progress every N patches",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel slide workers for extract materialize (default: 4)",
    )
    return parser.parse_args()


def _resolve_build_params(args: argparse.Namespace) -> tuple[float, int]:
    cfg_payload = load_fixed_grid_build_config(args.build_config if args.build_config.is_file() else None)
    rest_keep_fraction = args.rest_keep_fraction
    if rest_keep_fraction is None:
        rest_keep_fraction = float(cfg_payload.get("rest_keep_fraction", 0.10))
    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(cfg_payload.get("random_seed", 42))
    return float(rest_keep_fraction), int(random_seed)


def _slides_by_split(split_csv: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with split_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row["split"])
            grouped[split].append(str(row["slide_id"]))
    for split in grouped:
        grouped[split] = sorted(grouped[split])
    return dict(grouped)


def _run_preview(args: argparse.Namespace) -> None:
    if not args.slide_id:
        raise SystemExit("preview mode requires --slide-id")

    cfg = get_dataset_config(args.dataset, str(args.data_config))
    split_csv = args.split_csv or _default_split_csv(args.hooknet_root)
    row = load_slide_row_from_split(split_csv, args.slide_id)
    wsi_path, xml_path, split = resolve_slide_paths(cfg, args.slide_id, row)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    plan = build_slide_patch_plan(
        slide_id=args.slide_id,
        split=split,
        wsi_path=wsi_path,
        xml_path=xml_path,
        patch_size=int(args.patch_size),
        backend=args.backend,
    )
    plan_repeat = build_slide_patch_plan(
        slide_id=args.slide_id,
        split=split,
        wsi_path=wsi_path,
        xml_path=xml_path,
        patch_size=int(args.patch_size),
        backend=args.backend,
    )

    tiles_csv = out / "selected_tiles.csv"
    write_selected_tiles_csv(tiles_csv, plan.records)

    overview_path = out / "slide_overview.png"
    render_slide_overview_png(plan, overview_path, backend=args.backend)

    zoom_dir = out / "zoom_examples"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    examples = select_zoom_example_records(plan.records, max_examples=int(args.max_zoom_examples))
    for rec in examples:
        rgb = read_wsi_patch_level0(
            plan.wsi_path,
            rec.x_level0,
            rec.y_level0,
            rec.patch_size,
            backend=args.backend,
        )
        from data.hooknet_tls_wsd import load_official_wsa

        wsa = load_official_wsa(plan.xml_path, backend=args.backend)
        mask = rasterize_patch_mask(wsa, rec.x_level0, rec.y_level0, rec.patch_size)
        overlay = make_overlay(rgb, mask)
        stem = f"tile_ix{rec.grid_ix}_iy{rec.grid_iy}"
        Image.fromarray(rgb).save(zoom_dir / f"{stem}_rgb.png")
        Image.fromarray(mask, mode="L").save(zoom_dir / f"{stem}_mask.png")
        Image.fromarray(overlay).save(zoom_dir / f"{stem}_overlay.png")

    summary = summarize_patch_plan(plan)
    summary["mode"] = "preview"
    summary["note"] = (
        "Preview lists all slide-level grid candidates; Rest-only subsampling runs at split scope in extract mode."
    )
    summary["outputs"] = {
        "slide_overview_png": str(overview_path),
        "selected_tiles_csv": str(tiles_csv),
        "zoom_examples_dir": str(zoom_dir),
        "zoom_example_count": len(examples),
    }
    summary["determinism_repeat_sha256"] = manifest_determinism_hash(plan.records)
    summary["determinism_repeat_match"] = (
        manifest_determinism_hash(plan.records) == manifest_determinism_hash(plan_repeat.records)
    )
    summary_path = out / "preview_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2))


def _materialize_records(
    records,
    *,
    split_csv: Path,
    cfg,
    patch_size: int,
    backend: str,
    images_dir: Path,
    masks_dir: Path,
    skip_existing: bool = True,
    png_compress_level: int = 1,
    progress_every: int = 100,
    workers: int = 4,
):
    """Build path-only plan stubs (no re-rasterize of full slide grids)."""
    plan_by_slide = {}
    seen = set()
    for rec in records:
        if rec.slide_id in seen:
            continue
        seen.add(rec.slide_id)
        row = load_slide_row_from_split(split_csv, rec.slide_id)
        wsi_path, xml_path, split = resolve_slide_paths(cfg, rec.slide_id, row)
        # Prefer path already stamped on the record when present.
        if getattr(rec, "wsi_path", None):
            wsi_path = Path(rec.wsi_path)
        plan_by_slide[rec.slide_id] = slide_plan_stub_for_materialize(
            slide_id=rec.slide_id,
            split=split,
            wsi_path=wsi_path,
            xml_path=xml_path,
            patch_size=patch_size,
        )
    return materialize_selected_records(
        records,
        plan_by_slide=plan_by_slide,
        images_dir=images_dir,
        masks_dir=masks_dir,
        backend=backend,
        skip_existing=skip_existing,
        png_compress_level=png_compress_level,
        progress_every=progress_every,
        workers=workers,
    )


def _run_extract(args: argparse.Namespace) -> None:
    cfg = get_dataset_config(args.dataset, str(args.data_config))
    split_csv = args.split_csv or _default_split_csv(args.hooknet_root)
    out = args.output_dir.expanduser().resolve()
    rest_keep_fraction, random_seed = _resolve_build_params(args)
    patch_size = int(args.patch_size)
    workers = max(1, int(args.workers))

    if args.slide_id:
        row = load_slide_row_from_split(split_csv, args.slide_id)
        _, _, split = resolve_slide_paths(cfg, args.slide_id, row)
        splits = {split: [args.slide_id]}
    else:
        splits = _slides_by_split(split_csv)

    all_split_stats: dict[str, dict] = {}
    all_catalog: list = []
    patches_csv = out / "patches.csv"
    existing_catalog = load_selected_tiles_csv(patches_csv) if patches_csv.is_file() else []
    for split in sorted(splits.keys()):
        slide_ids = splits[split]
        split_dir = out / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        split_dir.mkdir(parents=True, exist_ok=True)

        summary_path = split_dir / "manifest_summary.json"
        if args.skip_existing and images_dir.is_dir():
            n_img = sum(1 for _ in images_dir.glob("*.png"))
            expected = None
            if summary_path.is_file():
                try:
                    expected = int(json.loads(summary_path.read_text(encoding="utf-8")).get("final_patch_count"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    expected = None
            if expected is None:
                expected = sum(1 for rec in existing_catalog if rec.split == split)
            if expected is not None and n_img >= expected > 0:
                print(
                    f"Split: {split} — skip materialize "
                    f"(already complete: {n_img}/{expected} patches)",
                    flush=True,
                )
                stats = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
                    "final_patch_count": expected,
                    "skipped_complete": True,
                }
                all_split_stats[split] = stats
                all_catalog.extend(rec for rec in existing_catalog if rec.split == split)
                continue

        print(
            f"Split: {split} — planning candidates "
            f"({len(slide_ids)} slides, workers={workers})...",
            flush=True,
        )
        candidates = collect_split_candidate_records(
            split=split,
            slide_ids=slide_ids,
            split_csv=split_csv,
            cfg=cfg,
            patch_size=patch_size,
            backend=args.backend,
            workers=workers,
        )
        selected, stats = apply_manifest_selection(
            candidates,
            rest_keep_fraction=rest_keep_fraction,
            random_seed=random_seed,
        )

        all_split_stats[split] = stats
        print(format_split_manifest_summary(split, stats), flush=True)
        write_manifest_summary_json(split_dir / "manifest_summary.json", split, stats)

        print(
            f"  materializing with workers={workers}, "
            f"png_compress_level={args.png_compress_level}, "
            f"skip_existing={args.skip_existing}",
            flush=True,
        )
        materialized = _materialize_records(
            selected,
            split_csv=split_csv,
            cfg=cfg,
            patch_size=patch_size,
            backend=args.backend,
            images_dir=images_dir,
            masks_dir=masks_dir,
            skip_existing=bool(args.skip_existing),
            png_compress_level=int(args.png_compress_level),
            progress_every=int(args.progress_every),
            workers=workers,
        )
        all_catalog.extend(materialized)
        print(f"  wrote {len(materialized)} patches -> {split_dir}", flush=True)

    if all_catalog:
        write_selected_tiles_csv(patches_csv, all_catalog)

    root_summary = out / "manifest_summary.json"
    root_summary.write_text(
        json.dumps(
            {
                "rest_keep_fraction": rest_keep_fraction,
                "random_seed": random_seed,
                "patch_size": patch_size,
                "workers": workers,
                "splits": all_split_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if args.mode == "preview":
        _run_preview(args)
    else:
        _run_extract(args)


if __name__ == "__main__":
    main()
