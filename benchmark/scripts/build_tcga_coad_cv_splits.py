#!/usr/bin/env python3
"""Build stratified k-fold split CSVs for TCGA-COAD prognosis MIL.

Each fold CSV uses an outer stratified patient fold as ``test`` and splits the
remaining patients into stratified ``train`` / ``valid`` subsets. Every patient
(and all associated slides) appears in exactly one outer test fold across the
full 5-fold run.

Example::

    python scripts/build_tcga_coad_cv_splits.py \\
      --base-split-csv ../splits/prognosis_os/tcga_coad_split.csv \\
      --n-folds 5 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def build_cv_splits(
    base_split_csv: str | Path,
    *,
    n_folds: int = 5,
    seed: int = 42,
    case_column: str = "case_id",
    event_column: str = "PFI.1",
    valid_fraction: float = 0.2,
    output_dir: str | Path | None = None,
    output_pattern: str = "tcga_coad_split_cv{n_folds}_outer_fold{fold}.csv",
) -> list[Path]:
    base_path = Path(base_split_csv).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base split CSV: {base_path}")

    frame = pd.read_csv(base_path)
    required = {case_column, "split", event_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Base split CSV missing columns: {sorted(missing)}")

    patients = frame.drop_duplicates(case_column).copy()
    patients[case_column] = patients[case_column].astype(str)
    patients[event_column] = pd.to_numeric(patients[event_column], errors="coerce")
    patients = patients.dropna(subset=[event_column]).reset_index(drop=True)
    if patients.empty:
        raise RuntimeError("No patients with valid event labels found for CV.")

    if len(patients) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} patients for CV, got {len(patients)}."
        )

    labels = patients[event_column].astype(int).to_numpy()
    split_indices = _stratified_kfold_indices(labels, n_folds=n_folds, seed=seed)

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else base_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    manifest_folds: list[dict[str, object]] = []

    for fold_idx, held_idx in enumerate(split_indices):
        held_set = set(int(idx) for idx in held_idx)
        train_valid_idx = [idx for idx in range(len(patients)) if idx not in held_set]
        held_ids = set(patients.iloc[held_idx][case_column].astype(str))
        train_valid = patients.iloc[train_valid_idx].reset_index(drop=True)
        test_ids = held_ids
        valid_ids = _stratified_valid_ids(
            train_valid,
            case_column=case_column,
            event_column=event_column,
            seed=seed + fold_idx,
            valid_fraction=valid_fraction,
        )

        rows: list[dict[str, str]] = []
        for _, row in patients.iterrows():
            pid = str(row[case_column])
            if pid in test_ids:
                split = "test"
            elif pid in valid_ids:
                split = "valid"
            else:
                split = "train"
            rows.extend(_patient_rows(frame, row, case_column=case_column, split=split))

        out_name = output_pattern.format(n_folds=n_folds, fold=fold_idx)
        out_path = out_dir / out_name
        _write_split_csv(out_path, rows, fieldnames=list(frame.columns))
        written.append(out_path)

        fold_patients = {r[case_column] for r in rows}
        fold_train = {r[case_column] for r in rows if r["split"] == "train"}
        fold_valid = {r[case_column] for r in rows if r["split"] == "valid"}
        fold_test = {r[case_column] for r in rows if r["split"] == "test"}
        manifest_folds.append(
            {
                "fold": fold_idx,
                "split_csv": str(out_path),
                "n_slides": len(rows),
                "n_patients": len(fold_patients),
                "n_train": len(fold_train),
                "n_valid": len(fold_valid),
                "n_test": len(fold_test),
                "train_events": int(
                    patients.loc[patients[case_column].isin(fold_train), event_column].sum()
                ),
                "valid_events": int(
                    patients.loc[patients[case_column].isin(fold_valid), event_column].sum()
                ),
                "test_events": int(
                    patients.loc[patients[case_column].isin(fold_test), event_column].sum()
                ),
            }
        )
        print(
            f"[fold {fold_idx}] patients train={len(fold_train)} valid={len(fold_valid)} "
            f"test={len(fold_test)} slides={len(rows)} -> {out_path.name}"
        )

    manifest = {
        "base_split_csv": str(base_path),
        "case_column": case_column,
        "event_column": event_column,
        "n_folds": n_folds,
        "seed": seed,
        "valid_fraction": valid_fraction,
        "output_pattern": output_pattern,
        "folds": manifest_folds,
    }
    manifest_path = out_dir / f"tcga_coad_split_cv{n_folds}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[manifest] {manifest_path}")
    return written


def _stratified_kfold_indices(labels: np.ndarray, *, n_folds: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(seed))
    folds: list[list[int]] = [[] for _ in range(int(n_folds))]
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for offset, idx in enumerate(indices.tolist()):
            folds[offset % int(n_folds)].append(int(idx))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def _stratified_valid_ids(
    patients: pd.DataFrame,
    *,
    case_column: str,
    event_column: str,
    seed: int,
    valid_fraction: float,
) -> set[str]:
    if patients.empty:
        return set()
    labels = patients[event_column].astype(int).to_numpy()
    counts = Counter(labels)
    n_valid = max(1, int(round(len(patients) * float(valid_fraction))))
    n_valid = min(n_valid, len(patients) - 1)
    valid_ids: set[str] = set()
    rng = pd.Series(range(len(patients))).sample(frac=1.0, random_state=seed).to_numpy()
    for label, count in counts.items():
        label_idx = [idx for idx in rng if labels[idx] == label]
        label_valid = max(1, int(round(n_valid * count / len(patients))))
        label_valid = min(label_valid, max(0, count - 1))
        for idx in label_idx[:label_valid]:
            valid_ids.add(str(patients.iloc[int(idx)][case_column]))
    if len(valid_ids) > n_valid:
        valid_ids = set(list(valid_ids)[:n_valid])
    return valid_ids


def _patient_rows(
    frame: pd.DataFrame,
    patient_row: pd.Series,
    *,
    case_column: str,
    split: str,
) -> list[dict[str, str]]:
    pid = str(patient_row[case_column])
    source_rows = frame.loc[frame[case_column].astype(str) == pid]
    rows: list[dict[str, str]] = []
    for _, row in source_rows.iterrows():
        out = {
            col: "" if pd.isna(row[col]) else str(row[col])
            for col in frame.columns
        }
        out["split"] = split
        rows.append(out)
    return rows


def _write_split_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCGA-COAD stratified k-fold split CSVs")
    parser.add_argument(
        "--base-split-csv",
        default="/data3/houjunlin/fxx/HE-mIF-Benchmark/splits/prognosis_os/tcga_coad_split.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--event-column", default="PFI.1")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output-pattern",
        default="tcga_coad_split_cv{n_folds}_outer_fold{fold}.csv",
    )
    args = parser.parse_args()
    build_cv_splits(
        args.base_split_csv,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
        case_column=str(args.case_column),
        event_column=str(args.event_column),
        valid_fraction=float(args.valid_fraction),
        output_dir=args.output_dir,
        output_pattern=args.output_pattern,
    )


if __name__ == "__main__":
    main()
