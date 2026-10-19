"""Place extracted WSI bags onto the protocol split layout."""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from feature_extraction.protocol import DatasetProtocol, unique_wsi_h5


def _real(path: Path) -> Path | None:
    try:
        if path.is_symlink():
            target = path.resolve()
            return target if target.is_file() else None
        return path if path.is_file() else None
    except OSError:
        return None


def hardlink(src: Path, dst: Path, stats: dict) -> None:
    src_real = _real(src)
    if src_real is None:
        stats["skipped_broken"].append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        stats["already"] += 1
        return
    try:
        os.link(src_real, dst)
        stats["linked"] += 1
    except OSError:
        shutil.copy2(src_real, dst)
        stats["copied"] += 1


def index_wsi_bags(root: Path, model: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for path in root.rglob("*.h5"):
        real = _real(path)
        if real is None:
            continue
        out[path.stem] = real
    return out


def empty_stats() -> dict:
    return defaultdict(list, linked=0, copied=0, already=0, skipped_broken=[], missing_slides=[])


def place_cv_wsi(
    index: dict[str, Path],
    dest_layer: Path,
    model: str,
    fold_csvs: Iterable[Path],
    stats: dict | None = None,
    *,
    valid_equals_test: bool = True,
) -> dict:
    stats = stats if stats is not None else empty_stats()
    for fold, csv_path in enumerate(fold_csvs):
        frame = pd.read_csv(csv_path)
        slide_col = "slide_id" if "slide_id" in frame.columns else "sample_id"
        for split, sub in frame.groupby(frame["split"].str.lower()):
            split_name = str(split)
            if split_name == "valid":
                continue
            dest_splits = [split_name]
            if valid_equals_test and split_name == "test":
                dest_splits.append("valid")
            for slide_id in sub[slide_col].astype(str):
                src = index.get(slide_id)
                if src is None:
                    stats["missing_slides"].append(f"{dest_layer.name}/{Path(csv_path).name}:{slide_id}")
                    continue
                for dest_split in dest_splits:
                    dst = dest_layer / f"fold_{fold}" / dest_split / "h5_files" / model / f"{slide_id}.h5"
                    hardlink(src, dst, stats)
    return stats


def place_wsi_split(
    index: dict[str, Path],
    dest_layer: Path,
    model: str,
    split_csv: Path,
    stats: dict | None = None,
) -> dict:
    """Hardlink unique bags into ``train|test/h5_files/{model}/{slide}.h5``."""
    stats = stats if stats is not None else empty_stats()
    frame = pd.read_csv(split_csv)
    slide_col = "slide_id" if "slide_id" in frame.columns else "sample_id"
    for row in frame.itertuples(index=False):
        slide_id = str(getattr(row, slide_col))
        split_name = str(getattr(row, "split")).lower()
        if split_name not in {"train", "valid", "test"}:
            continue
        src = index.get(slide_id)
        if src is None:
            stats["missing_slides"].append(f"{dest_layer.name}:{slide_id}")
            continue
        dst = dest_layer / split_name / "h5_files" / model / f"{slide_id}.h5"
        hardlink(src, dst, stats)
    return stats


def merge_bag_index(roots: Iterable[Path], model: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for slide_id, path in index_wsi_bags(root, model).items():
            index.setdefault(slide_id, path)
    return index


def remove_fold_layout(layer_root: Path) -> list[str]:
    """Drop leftover pooled-fold / unique-dump dirs once train/test exist."""
    removed: list[str] = []
    for path in sorted(layer_root.iterdir()):
        if path.name in {"train", "test"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.name)
    return removed


def place_layer(
    layer_root: Path,
    model: str,
    dataset: DatasetProtocol,
    stats: dict | None = None,
) -> dict:
    stats = stats if stats is not None else empty_stats()
    index = index_wsi_bags(layer_root, model)
    if not index:
        return stats
    if dataset.kind == "wsi_cv":
        place_cv_wsi(
            index,
            layer_root,
            model,
            dataset.fold_csvs(),
            stats,
            valid_equals_test=dataset.valid_equals_test,
        )
    elif dataset.kind == "wsi_split" and dataset.split_csv is not None:
        place_wsi_split(index, layer_root, model, dataset.split_csv, stats)
    return stats


def cleanup_unique_wsi_files(layer_root: Path, model: str) -> int:
    """Drop leftover flat ``h5_files/{model}/*.h5`` after fold hardlinks exist."""
    unique_dir = layer_root / "h5_files" / model
    if not unique_dir.is_dir():
        return 0
    fold_stems = {path.stem for path in layer_root.glob(f"fold_*/**/h5_files/{model}/*.h5")}
    removed = 0
    for path in unique_dir.glob("*.h5"):
        if path.stem in fold_stems:
            path.unlink()
            removed += 1
    return removed


def unique_slide_rows(dataset: DatasetProtocol) -> list[dict[str, str]]:
    """Union of fold CSVs as ``sample_id,split`` rows (split=all)."""
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for csv_path in dataset.fold_csvs():
        frame = pd.read_csv(csv_path)
        slide_col = "slide_id" if "slide_id" in frame.columns else "sample_id"
        for slide_id in frame[slide_col].astype(str):
            if slide_id in seen:
                continue
            seen.add(slide_id)
            rows.append({"sample_id": slide_id, "split": "all"})
    return rows


def write_unique_selection_csv(dataset: DatasetProtocol, dest: Path) -> Path:
    rows = unique_slide_rows(dataset)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest, index=False)
    return dest


def wsi_bag_complete(layer_root: Path, model: str, slide_id: str) -> bool:
    if unique_wsi_h5(layer_root, model, slide_id).is_file():
        return True
    for split in ("train", "valid", "test"):
        if (layer_root / split / "h5_files" / model / f"{slide_id}.h5").is_file():
            return True
    return any(layer_root.glob(f"fold_*/**/h5_files/{model}/{slide_id}.h5"))
