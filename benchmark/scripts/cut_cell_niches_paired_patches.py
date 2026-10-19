#!/usr/bin/env python3
"""Cut shared HE–mIF physical patches for Cell-niches Task 1.

HE tissue uses CLAM ``segmentTissue`` + white/black. mIF uses Aligned overlap,
Ultivue stitch, DAPI/CK dynamic range, and HE–mIF disagreement. One frozen
``patches.csv`` is written for all five generators.

Example::

    python benchmark/scripts/cut_cell_niches_paired_patches.py \\
        --qc-only --fov 688305e4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from preprocessing.cell_niches.extract import FilterConfig, extract_cohort  # noqa: E402

DEFAULT_ROOT = Path("/jhcnas6/Cytology/Spatial_Proteomics_Temp/Cell-niches-data")
DEFAULT_OUT = Path("/data3/houjunlin/fxx/datasets/cell_niches_paired_patches")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Cell-niches-data root")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fov", action="append", default=None, help="FOV parent UUID or prefix; repeatable")
    p.add_argument("--patch-size-um", type=float, default=256.0)
    p.add_argument("--stride-um", type=float, default=256.0)
    p.add_argument("--write-patches", action="store_true", help="Write native HE PNG + 12-ch mIF TIFF")
    p.add_argument("--qc-only", action="store_true", help="Manifest + QC thumbs only (default if neither flag)")
    p.add_argument("--use-otsu", action="store_true", help="CLAM use_otsu=True (create_patches default is False)")
    p.add_argument("--sthresh", type=int, default=8)
    p.add_argument("--white-thresh", type=int, default=5)
    p.add_argument("--black-thresh", type=int, default=40)
    p.add_argument("--tissue-min-frac", type=float, default=0.5)
    p.add_argument("--aligned-min-frac", type=float, default=0.95)
    p.add_argument("--stitch-max-frac", type=float, default=0.03)
    p.add_argument("--stitch-z-thresh", type=float, default=10.0)
    p.add_argument("--dapi-fg-min", type=float, default=0.1)
    p.add_argument("--min-dynamic-range", type=float, default=200.0)
    p.add_argument("--contour-fn", default="four_pt", choices=["four_pt", "four_pt_hard", "center", "basic"])
    p.add_argument("--max-examples-per-reason", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    write_patches = bool(args.write_patches) and not args.qc_only
    cfg = FilterConfig(
        patch_size_um=args.patch_size_um,
        stride_um=args.stride_um,
        sthresh=args.sthresh,
        use_otsu=args.use_otsu,
        white_thresh=args.white_thresh,
        black_thresh=args.black_thresh,
        contour_fn=args.contour_fn,
        tissue_min_frac=args.tissue_min_frac,
        aligned_min_frac=args.aligned_min_frac,
        stitch_max_frac=args.stitch_max_frac,
        stitch_z_thresh=args.stitch_z_thresh,
        dapi_fg_min=args.dapi_fg_min,
        min_dynamic_range=args.min_dynamic_range,
    )
    candidates = extract_cohort(
        args.root,
        args.output_dir,
        cfg,
        write_patches=write_patches,
        fov_ids=args.fov,
        max_examples_per_reason=args.max_examples_per_reason,
    )
    n_keep = int(candidates["keep"].sum()) if len(candidates) else 0
    print(f"candidates={len(candidates)} keep={n_keep} out={args.output_dir}")
    if len(candidates):
        print(candidates["reject_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
