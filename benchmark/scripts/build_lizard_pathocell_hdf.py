#!/usr/bin/env python3
"""Convert Lizard PNG + .mat labels into PathoCell-compatible HDF volumes.

Outputs (under ``--dataset-root``):

- ``pathocell_hdf/<stem>.hdf`` with ``img``, ``gt_inst``, ``gt_ct_coarse``
- ``splits/cell_semantic_seg/lizard_split.csv`` (sample_name, train_test_val_split)
- ``lizard_label_dict.json``
- ``lizard_preprocessing_report.json``

Split mapping (``info.csv`` ``Split`` column): 1→train, 2→valid, 3→test.

Mat classes 1 and 5 are both Neutrophil in the official README; both map to
semantic id 1. Mat class 6 → semantic 5 (Connective tissue). Foreground ids 1–5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import get_dataset_config  # noqa: E402
from utils.paths import SPLITS_ROOT  # noqa: E402

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image

# Mat class id → coarse semantic id (uint8, 0 = background).
RAW_CLASS_TO_SEMANTIC: dict[int, int] = {
    1: 1,  # Neutrophil
    5: 1,  # Neutrophil (duplicate label in README)
    2: 2,  # Epithelial
    3: 3,  # Lymphocyte
    4: 4,  # Plasma
    6: 5,  # Connective tissue
}

SPLIT_FROM_INFO: dict[int, str] = {
    1: "train",
    2: "valid",
    3: "test",
}

LABEL_DICT: dict[str, str] = {
    "0": "Background",
    "1": "Neutrophil",
    "2": "Epithelial",
    "3": "Lymphocyte",
    "4": "Plasma",
    "5": "Connective tissue",
}


def _find_png(dataset_root: Path, stem: str) -> Path:
    for rel in (
        "lizard_images1/Lizard_Images1",
        "lizard_images2/Lizard_Images2",
    ):
        path = dataset_root / rel / f"{stem}.png"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No PNG for stem {stem!r} under {dataset_root}")


def _load_info_splits(info_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with info_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = str(row.get("Filename") or "").strip()
            if not stem:
                continue
            split_id = int(str(row.get("Split") or "0").strip())
            if split_id not in SPLIT_FROM_INFO:
                raise ValueError(f"Unknown Split={split_id} for {stem}")
            mapping[stem] = SPLIT_FROM_INFO[split_id]
    return mapping


def _png_to_img_chw_uint16(png_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint16)
    # Match PathoCell uint16 H&E storage (8-bit source scaled into 16-bit range).
    rgb = (rgb * 257).astype(np.uint16)
    return np.transpose(rgb, (2, 0, 1))


def _build_instance_to_semantic(mat: dict[str, Any]) -> dict[int, int]:
    inst_ids = np.squeeze(mat["id"]).astype(np.int64)
    raw_classes = np.squeeze(mat["class"]).astype(np.int64)
    out: dict[int, int] = {}
    for inst_id, raw in zip(inst_ids.tolist(), raw_classes.tolist(), strict=True):
        sem = RAW_CLASS_TO_SEMANTIC.get(int(raw))
        if sem is None:
            raise ValueError(f"Unknown mat class {raw} for instance {inst_id}")
        out[int(inst_id)] = int(sem)
    return out


def _rasterize_semantic(inst_map: np.ndarray, inst_to_sem: dict[int, int]) -> np.ndarray:
    sem = np.zeros(inst_map.shape, dtype=np.uint8)
    for inst_id in np.unique(inst_map):
        iid = int(inst_id)
        if iid == 0:
            continue
        if iid not in inst_to_sem:
            raise ValueError(f"Instance id {iid} missing from mat id/class tables")
        sem[inst_map == iid] = inst_to_sem[iid]
    return sem


def _write_hdf(
    out_path: Path,
    img_chw: np.ndarray,
    inst_map: np.ndarray,
    semantic: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()
    inst = inst_map.astype(np.uint16)
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("img", data=img_chw, dtype=np.uint16)
        handle.create_dataset("gt_inst", data=inst[np.newaxis, ...], dtype=np.uint16)
        handle.create_dataset("gt_ct_coarse", data=semantic[np.newaxis, ...], dtype=np.uint8)
        handle.attrs["source"] = "lizard"
        handle.attrs["native_mpp"] = 0.5
        handle.attrs["schema"] = "pathocell_he_seg_v1"


def _process_one(
    stem: str,
    dataset_root: Path,
    labels_dir: Path,
    out_hdf_dir: Path,
    *,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any] | None:
    out_path = out_hdf_dir / f"{stem}.hdf"
    if skip_existing and out_path.is_file() and not dry_run:
        return None
    mat_path = labels_dir / f"{stem}.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    png_path = _find_png(dataset_root, stem)
    mat = sio.loadmat(mat_path)
    inst_map = np.asarray(mat["inst_map"], dtype=np.int32)
    inst_to_sem = _build_instance_to_semantic(mat)
    with Image.open(png_path) as im:
        png_hw = (im.size[1], im.size[0])
    if inst_map.shape != png_hw:
        raise ValueError(f"{stem}: inst_map {inst_map.shape} != PNG HW {png_hw}")
    semantic = _rasterize_semantic(inst_map, inst_to_sem)
    report: dict[str, Any] = {
        "stem": stem,
        "png": str(png_path),
        "mat": str(mat_path),
        "hw": [int(inst_map.shape[0]), int(inst_map.shape[1])],
        "n_instances": int(len(inst_to_sem)),
        "semantic_counts": {str(k): int(v) for k, v in zip(*np.unique(semantic, return_counts=True))},
    }
    if dry_run:
        return report
    img_chw = _png_to_img_chw_uint16(png_path)
    _write_hdf(out_path, img_chw, inst_map, semantic)
    report["hdf"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Defaults to the lizard dataset_dir in configs/datasets.yaml.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/lizard_labels/Lizard_Labels/Labels",
    )
    parser.add_argument(
        "--info-csv",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/lizard_labels/Lizard_Labels/info.csv",
    )
    parser.add_argument(
        "--out-hdf-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/pathocell_hdf",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stems whose .hdf already exists (fast re-run to refresh CSV/report only).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N tiles (0 = all).")
    args = parser.parse_args()

    root = (args.dataset_root or get_dataset_config("lizard").dataset_dir).resolve()
    labels_dir = (args.labels_dir or root / "lizard_labels/Lizard_Labels/Labels").resolve()
    info_csv = (args.info_csv or root / "lizard_labels/Lizard_Labels/info.csv").resolve()
    out_hdf_dir = (args.out_hdf_dir or root / "pathocell_hdf").resolve()

    split_by_stem = _load_info_splits(info_csv)
    stems = sorted(labels_dir.glob("*.mat"))
    stems = [p.stem for p in stems]
    if args.limit > 0:
        stems = stems[: args.limit]

    print(
        f"Lizard → PathoCell HDF: {len(stems)} tiles → {out_hdf_dir} "
        f"(skip_existing={args.skip_existing}, dry_run={args.dry_run})",
        flush=True,
    )

    per_sample: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0
    for i, stem in enumerate(stems, start=1):
        try:
            rep = _process_one(
                stem,
                root,
                labels_dir,
                out_hdf_dir,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            if rep is None:
                skipped += 1
                if i == 1 or i % 25 == 0 or i == len(stems):
                    print(f"  [{i}/{len(stems)}] skipped (exists): {stem}", flush=True)
                continue
            per_sample.append(rep)
            print(f"  [{i}/{len(stems)}] wrote {stem}.hdf {rep['hw']}", flush=True)
        except (OSError, ValueError, FileNotFoundError) as exc:
            errors.append(f"{stem}: {exc}")
            print(f"  [{i}/{len(stems)}] ERROR {stem}: {exc}", flush=True)

    if args.skip_existing and skipped:
        print(f"Skipped {skipped} existing HDF(s); rebuilt metadata for all {len(stems)} stems.", flush=True)
        for stem in stems:
            if not any(r["stem"] == stem for r in per_sample):
                per_sample.append({"stem": stem})

    split_counts = {"train": 0, "valid": 0, "test": 0}
    split_rows: list[tuple[str, str]] = []
    for stem in [r["stem"] for r in per_sample]:
        split = split_by_stem.get(stem)
        if split is None:
            errors.append(f"{stem}: missing from info.csv")
            continue
        split_counts[split] += 1
        split_rows.append((f"{stem}.hdf", split))

    if not args.dry_run and split_rows:
        split_csv = SPLITS_ROOT / "cell_semantic_seg" / "lizard_split.csv"
        split_csv.parent.mkdir(parents=True, exist_ok=True)
        with split_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_name", "train_test_val_split"])
            for name, split in sorted(split_rows):
                writer.writerow([name, split])

        label_path = root / "lizard_label_dict.json"
        label_path.write_text(json.dumps(LABEL_DICT, indent=2) + "\n", encoding="utf-8")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "labels_dir": str(labels_dir),
        "out_hdf_dir": str(out_hdf_dir),
        "dry_run": args.dry_run,
        "processed": len(per_sample),
        "skipped_existing": skipped,
        "split_counts": split_counts,
        "errors": errors,
        "class_mapping_mat_to_semantic": {str(k): v for k, v in RAW_CLASS_TO_SEMANTIC.items()},
        "label_dict": LABEL_DICT,
    }
    report_path = root / "lizard_preprocessing_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(f"Completed with {len(errors)} error(s); see {report_path}")


if __name__ == "__main__":
    main()
