#!/usr/bin/env python3
"""Make the leakage-free Immunocto assignment the one and only split.

1. Rewrite ``patient_split.csv`` to train/valid/test from
   :mod:`data.immunocto` (CRC02/11/19/30 test).
2. Recompute surplus-``other`` exclusions so each split keeps ``other`` count
   equal to that split's ``cd4`` count (seed 42, sorted paths). Freeze the list
   to ``other_excluded.csv.gz``.
3. Rewrite ``immunocto_split.csv`` split + ``offline_sampler_excluded`` to match.
4. Rebuild every linear-probe H5 tree: unique cells only (CRC29–32 are no
   longer copied into two splits), then drop the frozen exclusions.

Old catalog files are moved to ``Evaluation/linear_probe/_retired_old_split/``
(local scratch; not part of a paper rerun).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.immunocto import assignment, split_of  # noqa: E402

DATA = Path("/jhcnas5/houjunlin/multiplexed/Other/ImmunoctoV1")
SPLIT_DIR = REPO / "splits" / "single_cell_classification" / "immunocto"
SEED = 42
SPLITS = ("train", "valid", "test")
PF = REPO / "Pre_Features"
RETIRE = REPO / "Evaluation/linear_probe/_retired_old_split"

TREES = [
    ("uni2", "encoder", PF / "linear_probe_immunocto_uni2_20x_s32/encoder_z4"),
    ("rosie", "encoder", PF / "linear_probe_immunocto_rosie_40x_s64/encoder_stage3"),
    ("hemit", "encoder", PF / "linear_probe_immunocto_hemit_40x_s64/encoder_resnet_blocks"),
    ("mipheivit", "encoder", PF / "linear_probe_immunocto_mipheivit_20x_s32/encoder_z4"),
    ("gigatime", "encoder", PF / "linear_probe_immunocto_gigatime_20x_s32/encoder_x4_0"),
    ("histoplexer", "encoder", PF / "linear_probe_immunocto_histoplexer_40x_s64/encoder_bottleneck"),
    ("rosie", "output", PF / "linear_probe_immunocto_rosie_40x_s64/output_layers"),
    ("hemit", "output", PF / "linear_probe_immunocto_hemit_40x_s64/output_layers"),
    ("mipheivit", "output", PF / "linear_probe_immunocto_mipheivit_20x_s32/output_layers"),
    ("gigatime", "output", PF / "linear_probe_immunocto_gigatime_20x_s32/output_layers"),
    ("histoplexer", "output", PF / "linear_probe_immunocto_histoplexer_40x_s64/output_layers"),
]


def _rel_path(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if "CRC" in text:
        idx = text.find("CRC")
        text = text[idx:]
    return text.replace("\\", "/")


def _decode(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8")
    return str(v)


def freeze_catalog(split_csv: Path, excl_csv: Path, patient_csv: Path) -> dict:
    a = assignment()
    others: dict[str, list[str]] = {s: [] for s in SPLITS}
    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SPLITS}
    n_rows = 0
    with split_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            pid = str(row["patient_id"]).strip()
            lab = str(row["label"]).strip()
            rel = _rel_path(row.get("image_path") or "")
            if not rel:
                rel = f"{pid}/HE/{str(row['image_id']).strip()}.png"
            sp = split_of(pid)
            counts[sp][lab] += 1
            if lab == "other":
                others[sp].append(rel)

    rng = np.random.default_rng(SEED)
    excluded: list[str] = []
    quota = {}
    for sp in SPLITS:
        n_cd4 = int(counts[sp]["cd4"])
        paths = sorted(set(others[sp]))
        n_other = len(paths)
        keep_n = min(n_other, n_cd4)
        drop_n = n_other - keep_n
        if drop_n:
            drop_idx = np.sort(rng.choice(n_other, size=drop_n, replace=False))
            dropped = [paths[i] for i in drop_idx]
            excluded.extend(dropped)
        quota[sp] = {
            "n_cells": int(sum(counts[sp].values())),
            "cd4": n_cd4,
            "other_before": n_other,
            "other_kept": keep_n,
            "other_dropped": drop_n,
            "n_after": int(sum(counts[sp].values()) - drop_n),
            "by_label": dict(counts[sp]),
        }

    excluded_set = set(excluded)
    RETIRE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    for src, name in (
        (patient_csv, f"patient_split.legacy_{stamp}.csv"),
        (excl_csv, f"other_excluded.legacy_{stamp}.csv.gz"),
    ):
        if src.is_file():
            shutil.copy2(src, RETIRE / name)

    tmp_patient = patient_csv.with_suffix(".tmp")
    with tmp_patient.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["patient_id", "split"])
        w.writeheader()
        for sp in SPLITS:
            for pid in a[sp]:
                w.writerow({"patient_id": pid, "split": sp})
    os.replace(tmp_patient, patient_csv)

    tmp_excl = excl_csv.with_suffix(excl_csv.suffix + ".tmp")
    with gzip.open(tmp_excl, "wt", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["image_path"])
        w.writeheader()
        for p in sorted(excluded_set):
            w.writerow({"image_path": p})
    os.replace(tmp_excl, excl_csv)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_catalog_rows": n_rows,
        "assignment": {k: list(v) for k, v in a.items()},
        "quota": quota,
        "n_excluded": len(excluded_set),
        "patient_split_csv": str(patient_csv),
        "excluded_paths_csv": str(excl_csv),
        "rule": "per leakfree split, keep other count = cd4 count; drop the rest (seed 42, sorted paths)",
    }
    (RETIRE / "immunocto_split_quota.json").write_text(json.dumps(report, indent=2) + "\n")
    keep_path = REPO / "Pre_Features/immunocto_excluded_paths.txt"
    keep_path.write_text("\n".join(sorted(excluded_set)) + "\n")
    print(json.dumps({sp: quota[sp]["n_after"] for sp in SPLITS}, indent=2), flush=True)
    print(f"excluded {len(excluded_set)} surplus other cells", flush=True)
    return {"excluded": excluded_set, "report": report}


def rewrite_cell_csv(split_csv: Path, excluded: set[str]) -> None:
    tmp = split_csv.with_suffix(split_csv.suffix + ".canonical.tmp")
    n = 0
    with split_csv.open(newline="", encoding="utf-8-sig") as src, tmp.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            pid = str(row["patient_id"]).strip()
            rel = _rel_path(row.get("image_path") or "")
            row["split"] = split_of(pid)
            if "offline_sampler_excluded" in fields:
                row["offline_sampler_excluded"] = "True" if rel in excluded else "False"
            writer.writerow(row)
            n += 1
            if n % 1_000_000 == 0:
                print(f"  rewrote {n} catalog rows", flush=True)
    os.replace(tmp, split_csv)
    print(f"rewrote {split_csv} ({n} rows)", flush=True)


def _iter_h5(root: Path) -> list[Path]:
    hits = []
    if not root.is_dir():
        return hits
    for p in sorted(root.rglob("*.h5")):
        name = p.name
        if name.endswith(".bak") or "tmp" in name or "pre_" in name:
            continue
        hits.append(p)
    return hits


def rebuild_tree(dst: Path, excluded: set[str], *, dry_run: bool) -> dict:
    sources = []
    leak = dst.parent / f"{dst.name}_leakfree"
    for root in (dst, leak):
        sources.extend(_iter_h5(root))
    if not sources:
        raise FileNotFoundError(f"no H5 under {dst} or {leak}")

    # first-seen image_path wins: prefer leakfree files then original, valid>test>train
    def rank(path: Path) -> tuple[int, int]:
        text = str(path)
        pref = 0 if "_leakfree" in text else 1
        split_rank = 0
        for i, sp in enumerate(("valid", "test", "train")):
            if f"/{sp}/" in text or f"/{sp}_" in path.name or path.name.startswith(f"{sp}_"):
                split_rank = i
                break
        else:
            split_rank = 9
        return (pref, split_rank)

    sources = sorted(set(sources), key=rank)
    chosen: dict[str, tuple[Path, int, str]] = {}
    for path in sources:
        with h5py.File(path, "r") as handle:
            img = handle["image_paths"][:]
            cases = handle["case_ids"][:]
        for i, raw in enumerate(img):
            rel = _rel_path(_decode(raw))
            if rel in excluded or rel in chosen:
                continue
            case = _decode(cases[i]).upper()
            try:
                sp = split_of(case)
            except KeyError:
                continue
            chosen[rel] = (path, i, sp)

    by_split: dict[str, dict[Path, list[int]]] = {s: defaultdict(list) for s in SPLITS}
    for rel, (path, i, sp) in chosen.items():
        by_split[sp][path].append(i)

    report = {
        "dst": str(dst),
        "n_unique_kept": len(chosen),
        "n_per_split": {s: sum(len(v) for v in by_split[s].values()) for s in SPLITS},
        "n_source_files": len(sources),
    }
    print(f"{dst.name}: kept {report['n_unique_kept']} {report['n_per_split']}", flush=True)
    if dry_run:
        return report

    staging = dst.parent / f"{dst.name}.canonical_staging"
    if staging.exists():
        shutil.rmtree(staging)
    sample = sources[0].name
    # train_uni2_features.h5 -> uni2_features.h5
    for prefix in ("train_", "valid_", "test_"):
        if sample.startswith(prefix):
            sample = sample[len(prefix):]
            break

    for sp in SPLITS:
        file_idx = by_split[sp]
        n = sum(len(v) for v in file_idx.values())
        out_dir = staging / sp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sp}_{sample}"
        _write_split_h5(file_idx, out_path, n)
        print(f"  wrote {out_path.name} n={n}", flush=True)

    (staging / "split.json").write_text(json.dumps({
        "assignment": {k: list(v) for k, v in assignment().items()},
        "n_per_split": report["n_per_split"],
        "seed": SEED,
    }, indent=2) + "\n")

    retired = dst.parent / f"{dst.name}.pre_canonical"
    if dst.exists():
        if retired.exists():
            shutil.rmtree(retired)
        dst.rename(retired)
    staging.rename(dst)
    if leak.is_dir():
        shutil.rmtree(leak)
    if retired.exists():
        shutil.rmtree(retired)
        report["retired_deleted"] = True
    return report


def _write_split_h5(file_idx: dict[Path, list[int]], out_path: Path, n: int) -> None:
    if not file_idx:
        raise SystemExit(f"empty split for {out_path}")
    first = next(iter(file_idx))
    with h5py.File(first, "r") as proto, h5py.File(out_path, "w") as fout:
        for key, val in proto.attrs.items():
            fout.attrs[key] = val
        fout.attrs["immunocto_canonical_split"] = True
        fout.attrs["n_rows"] = n
        # features group
        feat_layers = list(proto["features"].keys())
        feat_shape = proto[f"features/{feat_layers[0]}"].shape[1:]
        feat_dtype = proto[f"features/{feat_layers[0]}"].dtype
        fg = fout.create_group("features")
        feat_dsets = {
            layer: fg.create_dataset(layer, shape=(n, *proto[f"features/{layer}"].shape[1:]),
                                     dtype=proto[f"features/{layer}"].dtype)
            for layer in feat_layers
        }
        other = {}
        for key in proto.keys():
            if key == "features":
                continue
            arr0 = proto[key]
            other[key] = fout.create_dataset(key, shape=(n, *arr0.shape[1:]), dtype=arr0.dtype)

    cursor = 0
    for path, idx in file_idx.items():
        order = np.asarray(idx, dtype=np.int64)
        with h5py.File(path, "r") as fin, h5py.File(out_path, "a") as fout:
            sl = slice(cursor, cursor + order.size)
            for layer in feat_layers:
                fout[f"features/{layer}"][sl] = fin[f"features/{layer}"][:][order]
            for key in other:
                fout[key][sl] = fin[key][:][order]
        cursor += order.size
    if cursor != n:
        raise SystemExit(f"row count mismatch {cursor} vs {n} for {out_path}")


def _split_h5(dst: Path, split: str) -> Path:
    hits = sorted(p for p in (dst / split).glob("*.h5") if p.is_file())
    if len(hits) != 1:
        raise SystemExit(f"expected 1 h5 under {dst / split}, got {[p.name for p in hits]}")
    return hits[0]


def _load_path_set(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        return {_rel_path(_decode(v)) for v in handle["image_paths"][:]}


def _filter_h5(src: Path, keep: set[str]) -> int:
    """Rewrite ``src`` in place, keeping only ``image_paths`` in ``keep``."""
    with h5py.File(src, "r") as handle:
        n_src = handle["image_paths"].shape[0]
        idx = [i for i, v in enumerate(handle["image_paths"][:]) if _rel_path(_decode(v)) in keep]
    if len(idx) == n_src:
        return n_src
    tmp = src.with_name(src.name + ".intersect_tmp")
    if tmp.exists():
        tmp.unlink()
    _write_split_h5({src: idx}, tmp, len(idx))
    os.replace(tmp, src)
    return len(idx)


def intersect_trees(*, dry_run: bool) -> dict:
    """Force every arm onto the same cells (GigaTIME extracted a smaller pool)."""
    per_split: dict[str, set[str] | None] = {s: None for s in SPLITS}
    tree_h5: dict[tuple[str, str], dict[str, Path]] = {}
    for model, route, dst in TREES:
        tree_h5[(route, model)] = {}
        for sp in SPLITS:
            h5 = _split_h5(dst, sp)
            tree_h5[(route, model)][sp] = h5
            paths = _load_path_set(h5)
            print(f"  {route}/{model} {sp} n={len(paths)}", flush=True)
            per_split[sp] = paths if per_split[sp] is None else (per_split[sp] & paths)
    report = {s: len(per_split[s] or ()) for s in SPLITS}
    print(f"intersection {report}", flush=True)
    if dry_run:
        return report
    for model, route, dst in TREES:
        n_written = {}
        for sp in SPLITS:
            n_written[sp] = _filter_h5(tree_h5[(route, model)][sp], per_split[sp] or set())
            print(f"  intersect {route}/{model} {sp} n={n_written[sp]}", flush=True)
        split_json = dst / "split.json"
        blob = json.loads(split_json.read_text()) if split_json.is_file() else {}
        blob["assignment"] = {k: list(v) for k, v in assignment().items()}
        blob["n_per_split"] = n_written
        blob["intersected"] = True
        blob["seed"] = SEED
        split_json.write_text(json.dumps(blob, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-catalog", action="store_true")
    ap.add_argument("--skip-h5", action="store_true")
    ap.add_argument("--intersect-only", action="store_true",
                    help="skip rebuild; intersect already-canonical trees")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-tree", type=str, default="", help="substring filter on feat dir")
    args = ap.parse_args()

    patient_csv = SPLIT_DIR / "patient_split.csv"
    excl_csv = SPLIT_DIR / "other_excluded.csv.gz"
    split_csv = DATA / "immunocto_split.csv"

    if args.intersect_only:
        print("\n=== intersect 11 trees onto shared cells ===", flush=True)
        intersect_trees(dry_run=args.dry_run)
        return

    if args.skip_catalog:
        excluded = set()
        with gzip.open(excl_csv, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                excluded.add(_rel_path(row["image_path"]))
        print(f"loaded frozen exclusion n={len(excluded)}", flush=True)
    else:
        blob = freeze_catalog(split_csv=split_csv, excl_csv=excl_csv, patient_csv=patient_csv)
        excluded = blob["excluded"]
        if not args.dry_run:
            rewrite_cell_csv(split_csv, excluded)

    if args.intersect_only:
        print("\n=== intersect 11 trees onto shared cells ===", flush=True)
        intersect_trees(dry_run=args.dry_run)
        return
    if args.skip_h5:
        return
    for model, route, dst in TREES:
        if args.only_tree and args.only_tree not in str(dst):
            continue
        print(f"\n=== {route}/{model} {dst} ===", flush=True)
        rebuild_tree(dst, excluded, dry_run=args.dry_run)
    if not args.only_tree:
        print("\n=== intersect 11 trees onto shared cells ===", flush=True)
        intersect_trees(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
