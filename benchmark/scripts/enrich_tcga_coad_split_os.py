#!/usr/bin/env python3
"""Add OS endpoint columns to an existing TCGA-COAD split CSV.

This preserves the existing patient-level train/valid/test assignment. It is
intended for paired endpoint checks, e.g. ROSIE PFI vs ROSIE OS on the same
slides and splits.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from build_tcga_coad_split import load_coad_prognosis

OS_COLUMNS = ("OS", "OS.time")


def enrich_split(
    split_csv: str | Path,
    *,
    cdr_xlsx: str | Path,
    backup: bool = True,
) -> Path:
    path = Path(split_csv).expanduser().resolve()
    cdr_path = Path(cdr_xlsx).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    if not cdr_path.is_file():
        raise FileNotFoundError(f"Missing TCGA-CDR workbook: {cdr_path}")

    prognosis = load_coad_prognosis(cdr_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"{path} has no header")
    if "case_id" not in fieldnames:
        raise ValueError(f"{path} must contain a case_id column")

    for col in OS_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    n_filled = 0
    n_missing = 0
    for row in rows:
        case_id = str(row.get("case_id", "")).strip().upper()
        endpoint = prognosis.get(case_id)
        if endpoint is None:
            n_missing += 1
            continue
        for col in OS_COLUMNS:
            value = endpoint.get(col)
            row[col] = "" if value is None else str(value)
        n_filled += 1

    if backup:
        backup_path = path.with_suffix(path.suffix + ".before_os")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {path}")
    print(f"  rows filled with OS endpoint: {n_filled}")
    print(f"  rows missing CDR endpoint: {n_missing}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-csv",
        default=str(Path(__file__).resolve().parents[2] / "splits/prognosis/tcga_coad/slide_split.csv"),
    )
    parser.add_argument(
        "--cdr-xlsx",
        default="/data3/houjunlin/fxx/datasets/TCGA_COAD/clinical.project-tcga-coad.2026-06-10/TCGA-CDR-SupplementalTableS1.xlsx",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    enrich_split(args.split_csv, cdr_xlsx=args.cdr_xlsx, backup=not args.no_backup)


if __name__ == "__main__":
    main()
