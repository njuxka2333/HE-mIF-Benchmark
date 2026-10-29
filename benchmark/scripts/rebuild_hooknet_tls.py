#!/usr/bin/env python3
"""Destructively rebuild HookNet-TLS as 1024 px @ 0.25 µm/px physical tiles."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.hooknet_tls.planning import (  # noqa: E402
    apply_manifest_selection,
    load_fixed_grid_build_config,
    write_manifest_summary_json,
    write_selected_tiles_csv,
)
from data.hooknet_tls.physical_grid import (  # noqa: E402
    build_physical_slide_plan,
    materialize_physical_record,
)
from data.hooknet_tls.annotations import load_official_wsa  # noqa: E402
from data.loader import get_dataset_config  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--yes-delete-existing", action="store_true")
    p.add_argument("--split-csv", type=Path, default=ROOT.parent / "splits" / "tls_region_segmentation" / "hooknet_tls" / "slide_split.csv")
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/datasets.yaml")
    p.add_argument("--build-config", type=Path, default=ROOT / "configs/hooknet_tls.yaml")
    p.add_argument("--reference-mpp", type=float, default=0.25)
    p.add_argument("--output-size", type=int, default=1024)
    p.add_argument("--png-compress-level", type=int, default=1)
    p.add_argument("--backend", default="openslide")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent slide workers for planning/materialization (default: 4).",
    )
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def _plan_job(job: tuple[str, str, str, str, int, float, str]):
    sid, split, wsi_path, xml_path, output_size, reference_mpp, backend = job
    return build_physical_slide_plan(
        slide_id=sid,
        split=split,
        wsi_path=Path(wsi_path),
        xml_path=Path(xml_path),
        output_size=output_size,
        reference_mpp=reference_mpp,
        backend=backend,
    )


def _materialize_slide_job(job: tuple):
    (
        plan,
        source_mpp,
        records,
        reference_mpp,
        images_dir,
        masks_dir,
        backend,
        png_compress_level,
    ) = job
    from wholeslidedata.image.wholeslideimage import WholeSlideImage

    wsi = WholeSlideImage(str(plan.wsi_path), backend=backend)
    wsa = load_official_wsa(plan.xml_path, backend=backend)
    if wsa is None:
        wsi.close()
        raise RuntimeError(f"Missing WSA for {plan.slide_id}")
    try:
        return [
            materialize_physical_record(
                record,
                wsi=wsi,
                wsa=wsa,
                source_mpp=source_mpp,
                reference_mpp=reference_mpp,
                images_dir=Path(images_dir),
                masks_dir=Path(masks_dir),
                png_compress_level=png_compress_level,
            )
            for record in records
        ]
    finally:
        wsi.close()


def main() -> None:
    a = args()
    out = a.output_root.resolve()
    if out.exists():
        if not a.yes_delete_existing:
            raise SystemExit(f"{out} exists. Re-run only with --yes-delete-existing to remove it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    build = load_fixed_grid_build_config(a.build_config)
    keep = float(build.get("rest_keep_fraction", 0.01))
    seed = int(build.get("random_seed", 42))
    workers = max(1, int(a.workers))
    cfg = get_dataset_config("hooknet_tls", str(a.data_config))
    by_split: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    with a.split_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_split[row["split"]].append(row)

    summaries = {}
    all_catalog = []
    for split, rows in by_split.items():
        print(f"planning {split}: {len(rows)} slides, workers={workers}", flush=True)
        all_records = []
        plans = {}
        mpps = {}
        jobs = [
            (
                row["slide_id"],
                split,
                row["wsi_path"],
                str(cfg.dataset_dir / cfg.annotations_subdir / f"{row['slide_id']}.xml"),
                a.output_size,
                a.reference_mpp,
                a.backend,
            )
            for row in rows
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            for n, (plan, source_mpp) in enumerate(pool.map(_plan_job, jobs, chunksize=1), 1):
                all_records.extend(plan.records)
                plans[plan.slide_id], mpps[plan.slide_id] = plan, source_mpp
                if n % 25 == 0 or n == len(rows):
                    print(f"  planned {n}/{len(rows)} slides", flush=True)

        selected, stats = apply_manifest_selection(
            all_records, rest_keep_fraction=keep, random_seed=seed,
        )
        split_dir = out / split
        write_manifest_summary_json(split_dir / "manifest_summary.json", split, stats)
        summaries[split] = stats
        if a.plan_only:
            all_catalog.extend(selected)
            continue

        images, masks = split_dir / "images", split_dir / "masks"
        materialized_by_slide = {}
        by_slide: dict[str, list] = {}
        for r in selected:
            by_slide.setdefault(r.slide_id, []).append(r)
        materialize_jobs = [
            (
                plans[sid],
                mpps[sid],
                selected_records,
                a.reference_mpp,
                str(images),
                str(masks),
                a.backend,
                a.png_compress_level,
            )
            for sid, selected_records in sorted(by_slide.items())
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {
                pool.submit(_materialize_slide_job, job): job[0].slide_id for job in materialize_jobs
            }
            for n, future in enumerate(as_completed(futures), 1):
                materialized_by_slide[futures[future]] = future.result()
                completed_patches = sum(len(rows) for rows in materialized_by_slide.values())
                if n % 10 == 0 or n == len(materialize_jobs):
                    print(
                        f"  materialized {n}/{len(materialize_jobs)} slides "
                        f"({completed_patches}/{len(selected)} patches)",
                        flush=True,
                    )
        materialized = [
            record
            for sid in sorted(materialized_by_slide)
            for record in materialized_by_slide[sid]
        ]
        all_catalog.extend(materialized)

    if all_catalog:
        write_selected_tiles_csv(out / "patches.csv", all_catalog)

    root_summary = {
        "geometry_protocol": "canonical_physical_grid_v1",
        "reference_mpp": a.reference_mpp,
        "output_size": a.output_size,
        "physical_fov_um": a.reference_mpp * a.output_size,
        "workers": workers,
        "rest_keep_fraction": keep,
        "random_seed": seed,
        "splits": summaries,
    }
    (out / "manifest_summary.json").write_text(json.dumps(root_summary, indent=2) + "\n")
    print(json.dumps(root_summary, indent=2))


if __name__ == "__main__":
    main()
