#!/usr/bin/env python3
"""Build ImmunOcto compact split manifests and WebDataset-style tar shards.

Each sample is stored as a ``.png`` plus a matching ``.json`` metadata file
with the same key. The tar files are intentionally not gzip-compressed so
DataLoader workers can stream large sequential files instead of opening
millions of tiny PNGs from NAS.

By default this script first compacts the large per-cell split CSV into a
patient-level split CSV plus an excluded-path CSV, then packs train/valid/test
tar shards. Excluded paths (surplus ``other`` cells aligned to each split's
``cd4`` count) are applied to **all** splits. Use ``--write-shards false`` to
only write the compact split files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import tarfile
import time
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SPLITS_ROOT  # noqa: E402
from data.loader import get_dataset_config, open_csv_table, resolve_table_path  # noqa: E402

IMMUNOCTO_SPLITS = SPLITS_ROOT / "single_cell_classification" / "immunocto"

TRUE_VALUES = {"true", "1", "yes", "y"}


def _norm_split(value: str) -> str:
    value = str(value or "").strip().lower()
    return {"val": "valid", "validation": "valid"}.get(value, value)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _label_from_image_id(image_id: str) -> str:
    parts = str(image_id).rsplit("_", 2)
    return parts[0] if len(parts) == 3 else str(image_id).split("_", 1)[0]


def _is_excluded(row: dict[str, str]) -> bool:
    return str(row.get("offline_sampler_excluded", "")).strip().lower() in TRUE_VALUES


def _relative_path(value: str, root: Path) -> str:
    path = Path(str(value).strip())
    if path.is_absolute():
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    return str(path)


def compact_manifest(
    root: Path,
    source_csv: Path,
    patient_split_csv: Path,
    excluded_csv: Path,
    report_json: Path,
    *,
    delete_source: bool = False,
) -> dict[str, object]:
    """Write patient split + excluded image manifests from the large per-cell CSV."""
    patients: dict[str, str] = {}
    split_counts = Counter()
    effective_counts = Counter()
    excluded_counts = Counter()

    patient_tmp = patient_split_csv.with_suffix(patient_split_csv.suffix + ".tmp")
    excluded_tmp = excluded_csv.with_suffix(excluded_csv.suffix + ".tmp")
    report_tmp = report_json.with_suffix(report_json.suffix + ".tmp")

    patient_split_csv.parent.mkdir(parents=True, exist_ok=True)
    excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    excluded_open = gzip.open if excluded_csv.name.endswith(".gz") else open
    with source_csv.open(newline="", encoding="utf-8-sig") as src, excluded_open(
        excluded_tmp, "wt", newline="", encoding="utf-8"
    ) as excl_handle:
        reader = csv.DictReader(src)
        required = {
            "patient_id",
            "split",
            "label",
            "offline_sampler_excluded",
            "image_path",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_csv} missing columns: {sorted(missing)}")

        excl_writer = csv.DictWriter(excl_handle, fieldnames=["image_path"])
        excl_writer.writeheader()

        for row in reader:
            patient_id = str(row["patient_id"]).strip()
            split = _norm_split(row["split"])
            label = str(row["label"]).strip()
            image_path = _relative_path(row["image_path"], root)
            if not patient_id:
                continue
            old_split = patients.setdefault(patient_id, split)
            if old_split != split:
                raise ValueError(f"patient {patient_id} has conflicting splits: {old_split}, {split}")

            split_counts[split] += 1
            key = f"{split}:{label}"
            if _is_excluded(row):
                excluded_counts[key] += 1
                excl_writer.writerow({"image_path": image_path})
            else:
                effective_counts[key] += 1

    with patient_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "split"])
        writer.writeheader()
        for patient_id, split in sorted(patients.items()):
            writer.writerow({"patient_id": patient_id, "split": split})

    report = {
        "source_csv": str(source_csv),
        "patient_split_csv": str(patient_split_csv),
        "excluded_csv": str(excluded_csv),
        "patient_count": len(patients),
        "split_counts_raw": dict(split_counts),
        "effective_counts_by_split_label": dict(effective_counts),
        "excluded_counts_by_split_label": dict(excluded_counts),
        "effective_total": int(sum(effective_counts.values())),
        "excluded_total": int(sum(excluded_counts.values())),
    }
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    os.replace(patient_tmp, patient_split_csv)
    os.replace(excluded_tmp, excluded_csv)
    os.replace(report_tmp, report_json)

    if delete_source:
        source_csv.unlink()

    return report


def _read_patient_split(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = str(row.get("patient_id") or row.get("case_id") or "").strip()
            split = _norm_split(row.get("split", ""))
            if patient_id and split:
                rows.append((patient_id, split))
    return rows


def _read_excluded_paths(path: Path | None) -> set[str]:
    path = resolve_table_path(path)
    if path is None or not path.is_file():
        return set()
    print(f"[manifest] loading excluded paths from {path}", flush=True)
    excluded: set[str] = set()
    with open_csv_table(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = str(row.get("image_path") or row.get("path") or row.get("relative_path") or "").strip()
            if rel:
                excluded.add(rel)
    print(f"[manifest] loaded {len(excluded)} excluded paths", flush=True)
    return excluded


def _iter_split_images(
    root: Path,
    patient_rows: Iterable[tuple[str, str]],
    split: str,
    excluded_paths: set[str],
):
    for patient_id, patient_split in patient_rows:
        if patient_split != split:
            continue
        he_dir = root / patient_id / "HE"
        if not he_dir.is_dir():
            continue
        for image_path in he_dir.glob("*.png"):
            rel = str(image_path.relative_to(root))
            if rel in excluded_paths:
                continue
            image_id = image_path.stem
            yield {
                "patient_id": patient_id,
                "image_id": image_id,
                "split": split,
                "label": _label_from_image_id(image_id),
                "image_path": rel,
                "abs_image_path": image_path,
            }


def _open_shard(output_dir: Path, split: str, shard_idx: int) -> tuple[tarfile.TarFile, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"immunocto-{split}-{shard_idx:06d}.tar"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tar = tarfile.open(tmp_path, mode="w")
    return tar, tmp_path


def _finalize_shard(tar: tarfile.TarFile, tmp_path: Path) -> Path:
    tar.close()
    final_path = tmp_path.with_suffix("")
    os.replace(tmp_path, final_path)
    return final_path


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(payload))


def _read_sample_payload(args: tuple[int, dict[str, object]]) -> tuple[str, bytes, bytes, str]:
    index, sample = args
    key = f"{index:09d}"
    image_path = Path(sample["abs_image_path"])
    png_bytes = image_path.read_bytes()
    metadata = {k: str(v) for k, v in sample.items() if k != "abs_image_path"}
    json_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return key, png_bytes, json_bytes, str(sample["label"])


def _chunked(iterable, chunk_size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _completed_shard_prefix(split_dir: Path, split: str, shard_size: int) -> tuple[int, list[dict[str, object]]]:
    shard_idx = 0
    rows: list[dict[str, object]] = []
    while True:
        path = split_dir / f"immunocto-{split}-{shard_idx:06d}.tar"
        if not path.is_file():
            break
        rows.append({"shard": path.name, "samples": int(shard_size)})
        shard_idx += 1
    return shard_idx, rows


def build_split_shards(
    root: Path,
    output_root: Path,
    patient_rows: list[tuple[str, str]],
    split: str,
    excluded_paths: set[str],
    *,
    shard_size: int,
    progress_every: int,
    read_workers: int,
    prefetch_samples: int,
    resume: bool,
    max_samples: int | None = None,
) -> dict[str, object]:
    split_dir = output_root / split
    manifest_path = split_dir / f"immunocto-{split}-shards.csv"
    report_path = split_dir / f"immunocto-{split}-report.json"
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] writing shards to {split_dir}", flush=True)

    shard_idx, shard_rows = _completed_shard_prefix(split_dir, split, shard_size) if resume else (0, [])
    resumed_samples = shard_idx * int(shard_size)
    if resumed_samples:
        print(
            f"[{split}] resume: found {shard_idx} complete shards, "
            f"skipping {resumed_samples} samples",
            flush=True,
        )
    if max_samples is not None and resumed_samples >= int(max_samples):
        print(f"[{split}] resume: max_samples already satisfied", flush=True)
        return {
            "split": split,
            "samples": resumed_samples,
            "resumed_samples": resumed_samples,
            "shard_size": shard_size,
            "num_shards": len(shard_rows),
            "label_counts": {},
            "manifest": str(manifest_path),
        }

    sample_in_shard = 0
    total = resumed_samples
    label_counts: Counter[str] = Counter()
    tar, tmp_path = _open_shard(split_dir, split, shard_idx)

    try:
        sample_iter = _iter_split_images(root, patient_rows, split, excluded_paths)
        indexed_iter = enumerate(sample_iter)
        if max_samples is not None:
            indexed_iter = ((idx, sample) for idx, sample in indexed_iter if resumed_samples <= idx < max_samples)
        elif resumed_samples:
            indexed_iter = ((idx, sample) for idx, sample in indexed_iter if idx >= resumed_samples)
        with ThreadPoolExecutor(max_workers=max(1, int(read_workers))) as pool:
            for chunk in _chunked(indexed_iter, max(1, int(prefetch_samples))):
                for key, png_bytes, json_bytes, label in pool.map(_read_sample_payload, chunk):
                    if sample_in_shard >= shard_size:
                        final = _finalize_shard(tar, tmp_path)
                        shard_rows.append({"shard": final.name, "samples": sample_in_shard})
                        shard_idx += 1
                        sample_in_shard = 0
                        tar, tmp_path = _open_shard(split_dir, split, shard_idx)

                    _add_bytes(tar, f"{key}.png", png_bytes)
                    _add_bytes(tar, f"{key}.json", json_bytes)
                    label_counts[label] += 1
                    total += 1
                    sample_in_shard += 1

                    if progress_every > 0 and total % progress_every == 0:
                        print(f"[{split}] packed {total} samples", flush=True)
    except Exception:
        tar.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if sample_in_shard > 0:
        final = _finalize_shard(tar, tmp_path)
        shard_rows.append({"shard": final.name, "samples": sample_in_shard})
    else:
        tar.close()
        if tmp_path.exists():
            tmp_path.unlink()

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard", "samples"])
        writer.writeheader()
        writer.writerows(shard_rows)

    report = {
        "split": split,
        "samples": total,
        "resumed_samples": resumed_samples,
        "shard_size": shard_size,
        "num_shards": len(shard_rows),
        "label_counts": dict(label_counts),
        "manifest": str(manifest_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Defaults to the immunocto dataset_dir in configs/datasets.yaml "
        "(override with PREPATH_DATASET_IMMUNOCTO_DIR).",
    )
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--patient-split", type=Path, default=None)
    parser.add_argument("--excluded-paths", type=Path, default=None)
    parser.add_argument("--compact-report", type=Path, default=None)
    parser.add_argument(
        "--write-shards",
        type=_str_to_bool,
        default=True,
        help="When false, only write compact split CSVs/report and skip tar shard creation.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the original large per-cell CSV after compact files are written.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--prefetch-samples", type=int, default=4096)
    parser.add_argument("--no-resume", action="store_true", help="Start from shard 0 instead of appending after existing shards.")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit per split.")
    args = parser.parse_args()

    root = (args.root or get_dataset_config("immunocto").dataset_dir).resolve()
    output_root = args.output_root or root / "immunocto_wds"
    source_csv = args.source_csv or root / "immunocto_split.csv"
    patient_split = args.patient_split or IMMUNOCTO_SPLITS / "patient_split.csv"
    excluded_paths_file = args.excluded_paths or IMMUNOCTO_SPLITS / "other_excluded.csv.gz"
    compact_report = args.compact_report or root / "immunocto_compact_manifest_report.json"

    compact_summary = None
    if source_csv.is_file():
        compact_summary = compact_manifest(
            root,
            source_csv,
            patient_split,
            excluded_paths_file,
            compact_report,
            delete_source=args.delete_source,
        )
        print(json.dumps({"compact_manifest": compact_summary}, indent=2, sort_keys=True))
    elif not patient_split.is_file():
        raise FileNotFoundError(
            f"Missing source split CSV {source_csv}; cannot create {patient_split}. "
            "Pass --source-csv or create the patient split first."
        )
    else:
        print(f"[compact] source CSV not found ({source_csv}); using existing {patient_split}")

    if not args.write_shards:
        if compact_summary is None:
            compact_summary = {
                "patient_split_csv": str(patient_split),
                "excluded_csv": str(excluded_paths_file),
                "source_csv": str(source_csv),
                "status": "existing_patient_split_reused",
            }
        print(json.dumps(compact_summary, indent=2, sort_keys=True))
        return

    patient_rows = _read_patient_split(patient_split)
    excluded_paths = _read_excluded_paths(excluded_paths_file)
    reports = []
    for split_raw in args.splits:
        split = _norm_split(split_raw)
        reports.append(
            build_split_shards(
                root,
                output_root,
                patient_rows,
                split,
                excluded_paths,
                shard_size=int(args.shard_size),
                progress_every=int(args.progress_every),
                read_workers=int(args.read_workers),
                prefetch_samples=int(args.prefetch_samples),
                resume=not args.no_resume,
                max_samples=args.max_samples,
            )
        )

    summary = {
        "root": str(root),
        "output_root": str(output_root),
        "source_csv": str(source_csv),
        "patient_split": str(patient_split),
        "excluded_paths": str(excluded_paths_file),
        "compact_report": str(compact_report),
        "compact_manifest": compact_summary,
        "reports": reports,
    }
    summary_path = output_root / "immunocto_wds_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
