#!/usr/bin/env python3
"""Build TCGA-COAD ``tcga_coad_split.csv`` with a 7:1:2 train/valid/test split.

Prognosis features for stratification come from TCGA Pan-Cancer Clinical Data Resource
(Liu et al.) ``ExtraEndpoints`` sheet (``PFI.*``, ``PFS*``, ``*_cr``, and ``*.time*``).
Slides are assigned at **patient** level (``TCGA-XX-XXXX``) so multi-slide cases stay in
one split.

Example::

    python scripts/build_tcga_coad_split.py \\
      --dataset-root /data3/houjunlin/fxx/datasets/TCGA_COAD
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

PREPATH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PREPATH_ROOT))

from data.loader import build_wsi_path_map, get_dataset_config

CDR_MAIN_COLS = [
    "OS",
    "OS.time",
]

EXTRA_ENDPOINT_COLS = [
    "PFI.1",
    "PFI.time.1",
    "PFI.2",
    "PFI.time.2",
    "PFS",
    "PFS.time",
    "DSS_cr",
    "DSS.time.cr",
    "DFI.cr",
    "DFI.time.cr",
    "PFI.cr",
    "PFI.time.cr",
    "PFI.1.cr",
    "PFI.time.1.cr",
    "PFI.2.cr",
    "PFI.time.2.cr",
]

PROGNOSIS_COLS = [*CDR_MAIN_COLS, *EXTRA_ENDPOINT_COLS]

EVENT_COLS = {
    "OS",
    "PFI.1",
    "PFI.2",
    "PFS",
    "DSS_cr",
    "DFI.cr",
    "PFI.cr",
    "PFI.1.cr",
    "PFI.2.cr",
}

TIME_COLS = {c for c in PROGNOSIS_COLS if c not in EVENT_COLS}


def case_id_from_slide(slide_id: str) -> str:
    base = slide_id.split(".", 1)[0].upper()
    parts = base.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return base


def _parse_cdr_value(raw: Any) -> float | int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "#N/A", "NA", "nan", "None"):
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return int(v) if v.is_integer() else v
    except (TypeError, ValueError):
        return None


def _xlsx_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text = cell.findtext("main:is/main:t", default="", namespaces=ns)
        return text
    value = cell.findtext("main:v", default=None, namespaces=ns)
    if value is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (IndexError, ValueError):
            return value
    return value


def _load_xlsx_sheet_rows(xlsx_path: Path, sheet_name: str) -> list[list[Any]]:
    """Small XLSX reader fallback for environments without openpyxl."""
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//main:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
            if "Id" in rel.attrib and "Target" in rel.attrib
        }
        target = None
        for sheet in workbook.findall(".//main:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                target = rel_targets.get(sheet.attrib.get(f"{{{ns['rel']}}}id", ""))
                break
        if target is None:
            raise KeyError(f"{xlsx_path} missing {sheet_name} sheet")
        sheet_path = Path("xl") / target
        sheet_path = Path(*[part for part in sheet_path.parts if part != ".."])
        sheet_root = ET.fromstring(zf.read(str(sheet_path)))
        rows: list[list[Any]] = []
        for row in sheet_root.findall(".//main:sheetData/main:row", ns):
            values: list[Any] = []
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                idx = _xlsx_col_index(ref) if ref else len(values)
                while len(values) <= idx:
                    values.append(None)
                values[idx] = _xlsx_cell_value(cell, shared_strings, ns)
            rows.append(values)
        return rows


def _prognosis_from_rows(
    rows: list[list[Any]],
    *,
    sheet_name: str,
    columns: list[str],
) -> dict[str, dict[str, float | int | None]]:
    if not rows:
        raise RuntimeError(f"No rows found in {sheet_name}")
    header = rows[0]
    name_to_idx = {
        str(c).strip(): i for i, c in enumerate(header) if c is not None and str(c).strip()
    }
    for need in ("bcr_patient_barcode", "type", *columns):
        if need not in name_to_idx:
            raise KeyError(f"Column {need!r} missing from {sheet_name}")

    out: dict[str, dict[str, float | int | None]] = {}
    for row in rows[1:]:
        cancer = row[name_to_idx["type"]] if name_to_idx["type"] < len(row) else None
        if cancer is None or str(cancer).strip().upper() != "COAD":
            continue
        pid_raw = (
            row[name_to_idx["bcr_patient_barcode"]]
            if name_to_idx["bcr_patient_barcode"] < len(row)
            else None
        )
        pid = str(pid_raw).strip().upper()
        if not pid:
            continue
        out[pid] = {
            col: _parse_cdr_value(row[name_to_idx[col]] if name_to_idx[col] < len(row) else None)
            for col in columns
        }
    return out


def load_coad_prognosis(cdr_xlsx: Path) -> dict[str, dict[str, float | int | None]]:
    main = _prognosis_from_rows(
        _load_xlsx_sheet_rows(cdr_xlsx, "TCGA-CDR"),
        sheet_name="TCGA-CDR",
        columns=CDR_MAIN_COLS,
    )
    extra = _prognosis_from_rows(
        _load_xlsx_sheet_rows(cdr_xlsx, "ExtraEndpoints"),
        sheet_name="ExtraEndpoints",
        columns=EXTRA_ENDPOINT_COLS,
    )
    out: dict[str, dict[str, float | int | None]] = {}
    for pid, endpoints in extra.items():
        merged = {col: None for col in PROGNOSIS_COLS}
        merged.update(main.get(pid, {}))
        merged.update(endpoints)
        out[pid] = merged
    if not out:
        raise RuntimeError("No COAD prognosis rows found in TCGA-CDR workbook")
    return out


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    case_id: str
    wsi_path: Path


def _discover_slides(dataset_name: str = "tcga_coad") -> list[SlideRecord]:
    cfg = get_dataset_config(dataset_name)
    wsi_map = build_wsi_path_map(cfg)
    records = [
        SlideRecord(
            slide_id=sid,
            case_id=case_id_from_slide(sid),
            wsi_path=path.resolve(),
        )
        for sid, path in sorted(wsi_map.items())
    ]
    if not records:
        raise RuntimeError("No TCGA-COAD WSI files found under configured search roots.")
    return records


def _split_targets(n_patients: int) -> dict[str, int]:
    train = round(n_patients * 7 / 10)
    valid = round(n_patients * 1 / 10)
    test = n_patients - train - valid
    return {"train": train, "valid": valid, "test": test}


def _time_bucket(value: float | int | None) -> str:
    if not isinstance(value, (int, float)):
        return "NA"
    if value < 365:
        return "lt1y"
    if value < 1095:
        return "y1to3"
    if value < 1825:
        return "y3to5"
    return "ge5y"


def _stratum_key(patients: dict[str, dict[str, float | int | None]], case_id: str) -> tuple[str, ...]:
    prog = patients[case_id]

    def ev(col: str) -> str:
        val = prog.get(col)
        return "NA" if val is None else str(int(val))

    return (
        ev("OS"),
        ev("PFI.1"),
        ev("PFI.2"),
        ev("PFS"),
        ev("DSS_cr"),
        ev("PFI.cr"),
        _time_bucket(prog.get("PFI.time.1")),
        _time_bucket(prog.get("OS.time")),
        _time_bucket(prog.get("DSS.time.cr")),
    )


def _balance_cost(
    assignment: dict[str, str],
    patients: dict[str, dict[str, float | int | None]],
    targets: dict[str, int],
) -> float:
    buckets: dict[str, list[str]] = {k: [] for k in targets}
    for pid, split in assignment.items():
        buckets[split].append(pid)

    cost = 0.0
    for split, pids in buckets.items():
        cost += 1000.0 * abs(len(pids) - targets[split])
    for col in PROGNOSIS_COLS:
        if col in TIME_COLS:
            all_vals = [
                patients[p][col] for p in assignment if isinstance(patients[p].get(col), (int, float))
            ]
            global_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0
            for split, pids in buckets.items():
                vals = [
                    patients[p][col] for p in pids if isinstance(patients[p].get(col), (int, float))
                ]
                if not vals:
                    continue
                cost += abs((sum(vals) / len(vals)) - global_mean) / max(global_mean, 1.0)
        else:
            global_counts = Counter(
                "NA" if patients[p].get(col) is None else str(int(patients[p][col])) for p in assignment
            )
            n = len(assignment)
            for split, pids in buckets.items():
                if not pids:
                    continue
                local_counts = Counter(
                    "NA" if patients[p].get(col) is None else str(int(patients[p][col])) for p in pids
                )
                for key, gcount in global_counts.items():
                    cost += abs((local_counts.get(key, 0) / len(pids)) - (gcount / n))
    return cost


def _assign_splits(
    case_ids: list[str],
    patients: dict[str, dict[str, float | int | None]],
    *,
    seed: int,
) -> dict[str, str]:
    targets = _split_targets(len(case_ids))
    by_stratum: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for case_id in case_ids:
        by_stratum[_stratum_key(patients, case_id)].append(case_id)

    best: dict[str, str] | None = None
    best_cost = float("inf")

    for attempt in range(256):
        rng = random.Random(seed + attempt)
        assignment: dict[str, str] = {}

        for stratum_cases in by_stratum.values():
            members = list(stratum_cases)
            rng.shuffle(members)
            n = len(members)
            n_train = int(n * 7 / 10)
            n_valid = int(n * 1 / 10)
            n_test = n - n_train - n_valid
            for pid in members[:n_train]:
                assignment[pid] = "train"
            for pid in members[n_train : n_train + n_valid]:
                assignment[pid] = "valid"
            for pid in members[n_train + n_valid :]:
                assignment[pid] = "test"

        counts = Counter(assignment.values())
        for split, target in targets.items():
            delta = counts[split] - target
            if delta == 0:
                continue
            donor = split if delta > 0 else max(
                (s for s in targets if counts[s] > targets[s]), key=lambda s: counts[s] - targets[s]
            )
            receiver = split if delta < 0 else min(
                (s for s in targets if counts[s] < targets[s]), key=lambda s: targets[s] - counts[s]
            )
            movable = [
                pid
                for pid, current in assignment.items()
                if current == donor
            ]
            rng.shuffle(movable)
            for pid in movable:
                if counts[donor] <= targets[donor] and counts[receiver] >= targets[receiver]:
                    break
                if counts[donor] <= targets[donor]:
                    continue
                if counts[receiver] >= targets[receiver]:
                    continue
                assignment[pid] = receiver
                counts[donor] -= 1
                counts[receiver] += 1

        if len(assignment) != len(case_ids):
            continue
        if any(counts[s] != targets[s] for s in targets):
            continue
        cost = _balance_cost(assignment, patients, targets)
        if cost < best_cost:
            best = assignment
            best_cost = cost

    if best is None:
        raise RuntimeError("Failed to assign patient-level splits.")
    return best


def _summarize_split(
    name: str,
    case_ids: list[str],
    patients: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    if not case_ids:
        return {"split": name, "n_patients": 0}
    summary: dict[str, Any] = {"split": name, "n_patients": len(case_ids)}
    for col in PROGNOSIS_COLS:
        if col in TIME_COLS:
            vals = [
                patients[c][col] for c in case_ids if isinstance(patients[c].get(col), (int, float))
            ]
            summary[f"mean_{col}"] = round(sum(vals) / len(vals), 1) if vals else None
        else:
            counts = Counter(
                "NA" if patients[c].get(col) is None else str(int(patients[c][col])) for c in case_ids
            )
            summary[f"{col}_dist"] = dict(sorted(counts.items()))
    return summary


def build_tcga_coad_split(
    dataset_root: str | Path,
    output_csv: str | Path,
    *,
    cdr_xlsx: str | Path | None = None,
    seed: int = 42,
    dataset_name: str = "tcga_coad",
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    out = Path(output_csv).expanduser().resolve()
    clinical_dir = root / "clinical.project-tcga-coad.2026-06-10"
    cdr_path = (
        Path(cdr_xlsx).expanduser().resolve()
        if cdr_xlsx
        else clinical_dir / "TCGA-CDR-SupplementalTableS1.xlsx"
    )
    if not cdr_path.is_file():
        raise FileNotFoundError(
            f"Missing TCGA-CDR workbook at {cdr_path}. "
            "Place TCGA-CDR-SupplementalTableS1.xlsx (ExtraEndpoints sheet) there."
        )

    prognosis = load_coad_prognosis(cdr_path)
    slides = _discover_slides(dataset_name=dataset_name)

    by_case: dict[str, list[SlideRecord]] = defaultdict(list)
    for rec in slides:
        by_case[rec.case_id].append(rec)

    missing_prog = sorted({c for c in by_case if c not in prognosis})
    if missing_prog:
        print(
            f"[TCGA-COAD] warning: {len(missing_prog)} WSI cases lack prognosis in CDR "
            f"(e.g. {missing_prog[:3]}); excluding them."
        )
    case_ids = sorted(c for c in by_case if c in prognosis)
    if not case_ids:
        raise RuntimeError("No WSI cases with matching TCGA-CDR prognosis rows.")

    assignment = _assign_splits(case_ids, prognosis, seed=seed)

    rows: list[dict[str, str]] = []
    for case in case_ids:
        split = assignment[case]
        prog = prognosis[case]
        for rec in sorted(by_case[case], key=lambda r: r.slide_id):
            row = {
                "slide_id": rec.slide_id,
                "case_id": rec.case_id,
                "split": split,
                "label": "",
                "wsi_path": str(rec.wsi_path),
            }
            for col in PROGNOSIS_COLS:
                val = prog.get(col)
                row[col] = "" if val is None else str(val)
            rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slide_id", "case_id", "split", "label", "wsi_path", *PROGNOSIS_COLS]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    targets = _split_targets(len(case_ids))
    print(f"[TCGA-COAD] wrote {out}")
    print(f"  slides: {len(rows)}  patients: {len(case_ids)}")
    print(f"  target ratio 7:1:2 -> {targets}")
    for split in ("train", "valid", "test"):
        cases = [c for c, s in assignment.items() if s == split]
        n_slides = sum(len(by_case[c]) for c in cases)
        print(f"  {split}: {len(cases)} patients, {n_slides} slides")
        stats = _summarize_split(split, cases, prognosis)
        print(
            f"    mean OS.time={stats.get('mean_OS.time')}  "
            f"OS_dist={stats.get('OS_dist')}  "
            f"    mean PFI.time.1={stats.get('mean_PFI.time.1')}  "
            f"PFI.1_dist={stats.get('PFI.1_dist')}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCGA-COAD tcga_coad_split.csv")
    parser.add_argument(
        "--dataset-root",
        default="/data3/houjunlin/fxx/datasets/TCGA_COAD",
        help="TCGA-COAD dataset root.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV (default: HE-mIF-Benchmark/splits/prognosis_os/tcga_coad_split.csv).",
    )
    parser.add_argument(
        "--cdr-xlsx",
        default=None,
        help="TCGA-CDR Supplemental Table S1 xlsx (default: clinical.project-*/TCGA-CDR-SupplementalTableS1.xlsx).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    out = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else Path("/data3/houjunlin/fxx/HE-mIF-Benchmark/splits/prognosis_os/tcga_coad_split.csv")
    )
    build_tcga_coad_split(
        root,
        out,
        cdr_xlsx=args.cdr_xlsx,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
