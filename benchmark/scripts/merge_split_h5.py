"""Concatenate sharded single-cell split H5 files into one protocol tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _copy_attrs(src: h5py.File, dst: h5py.File, n_rows: int) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = n_rows if key == "n_rows" else value


def _concat_dataset(handle: h5py.File, name: str, arrays: list[np.ndarray]) -> None:
    stacked = np.concatenate(arrays, axis=0)
    if stacked.dtype.kind in {"O", "U", "S"}:
        dt = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(name, data=np.asarray(stacked, dtype=object), dtype=dt)
        return
    handle.create_dataset(name, data=stacked, compression=None)


def merge_split(shard_files: list[Path], dest: Path) -> int:
    if not shard_files:
        raise SystemExit(f"no shard H5s for {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[np.ndarray]] = {}
    feat_tables: dict[str, list[np.ndarray]] = {}
    n_rows = 0
    first = shard_files[0]
    with h5py.File(first, "r") as src:
        keys = [k for k in src.keys() if k != "features"]
        feat_keys = list(src["features"].keys()) if "features" in src else []
    for path in shard_files:
        with h5py.File(path, "r") as src:
            n_rows += int(src.attrs.get("n_rows") or src["labels"].shape[0])
            for key in keys:
                tables.setdefault(key, []).append(src[key][()])
            for key in feat_keys:
                feat_tables.setdefault(key, []).append(src["features"][key][()])
    tmp = dest.with_name(dest.name + ".merge_tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(first, "r") as src, h5py.File(tmp, "w") as dst:
        _copy_attrs(src, dst, n_rows)
        for key, chunks in tables.items():
            _concat_dataset(dst, key, chunks)
        if feat_tables:
            grp = dst.create_group("features")
            for key, chunks in feat_tables.items():
                stacked = np.concatenate(chunks, axis=0)
                grp.create_dataset(key, data=stacked)
    tmp.replace(dest)
    print(f"merged {len(shard_files)} shards -> {dest} n={n_rows}")
    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--h5-name", default="train_mipheivit_features.h5")
    args = parser.parse_args()
    counts = {}
    for split in args.splits:
        name = f"{split}_mipheivit_features.h5"
        files = []
        for i in range(args.n_shards):
            path = args.shard_root / f"shard{i}" / "encoder_z4" / split / name
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            files.append(path)
        dest = args.dest_root / split / name
        counts[split] = merge_split(files, dest)
    split_json = args.dest_root / "split.json"
    payload = {"n_per_split": counts, "sharded": True, "n_shards": args.n_shards}
    split_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {split_json}")


if __name__ == "__main__":
    main()
