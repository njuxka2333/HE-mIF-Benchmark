#!/usr/bin/env python3
"""Patient-level 5-fold LUAD OS splits, seed 42, balanced event + time.

Train/test only (Shazam protocol: valid_split=test). All slides of a case
stay in the same fold. Stratify on OS event crossed with OS.time quartiles
so each held-out fold has similar event counts and survival-time mix.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SPLITS_ROOT  # noqa: E402

LUAD_DIR = SPLITS_ROOT / "prognosis_os"
SRC = LUAD_DIR / "tcga_luad_split.csv"
N_FOLDS = 5
SEED = 42
N_TIME_BINS = 4
MIN_PER_STRATUM = N_FOLDS


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, g in df.groupby("case_id", sort=True):
        events = pd.to_numeric(g["OS"], errors="coerce")
        times = pd.to_numeric(g["OS.time"], errors="coerce")
        event = int((events.fillna(0) == 1).any())
        time = float(times.dropna().median()) if times.notna().any() else 0.0
        if not np.isfinite(time):
            time = 0.0
        rows.append({"case_id": case_id, "event": event, "time": time, "n_slides": len(g)})
    return pd.DataFrame(rows)


def _time_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    try:
        bins = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
        return np.asarray(bins, dtype=int)
    except ValueError:
        return np.zeros(len(values), dtype=int)


def make_strata(patients: pd.DataFrame) -> pd.Series:
    event = patients["event"].to_numpy(dtype=int)
    time = patients["time"].to_numpy(dtype=float)
    raw = np.empty(len(patients), dtype=object)
    for e in (0, 1):
        idx = np.where(event == e)[0]
        bins = _time_bin(time[idx], N_TIME_BINS)
        for i, b in zip(idx, bins):
            raw[i] = f"{e}_{int(b)}"
    counts = Counter(raw)
    mapping = {k: k for k in counts}
    # Merge rare time bins into the previous bin of the same event.
    for e in (0, 1):
        keys = sorted(k for k in counts if k.startswith(f"{e}_"))
        for key in keys:
            if counts[mapping[key]] >= MIN_PER_STRATUM:
                continue
            prev = [k for k in keys if k < key]
            target = prev[-1] if prev else (keys[1] if len(keys) > 1 and keys[0] == key else key)
            if target == key:
                continue
            counts[mapping[target]] += counts[mapping[key]]
            counts[mapping[key]] = 0
            for src, dst in list(mapping.items()):
                if dst == key:
                    mapping[src] = mapping[target]
    return pd.Series([mapping[s] for s in raw], index=patients.index, name="stratum")


def split_stats(df: pd.DataFrame, split: str) -> dict:
    sub = df[df["split"] == split]
    patients = sub.drop_duplicates("case_id")
    event = pd.to_numeric(patients["OS"], errors="coerce").fillna(0).eq(1)
    time = pd.to_numeric(patients["OS.time"], errors="coerce")
    return {
        "n_slides": int(len(sub)),
        "n_patients": int(len(patients)),
        "n_events": int(event.sum()),
        "event_rate": float(event.mean()) if len(patients) else 0.0,
        "time_median": float(time.median()) if len(patients) else 0.0,
        "time_mean": float(time.mean()) if len(patients) else 0.0,
        "time_p25": float(time.quantile(0.25)) if len(patients) else 0.0,
        "time_p75": float(time.quantile(0.75)) if len(patients) else 0.0,
    }


def write_folds(df: pd.DataFrame, patients: pd.DataFrame, assignments: dict[str, int]) -> list[dict]:
    records = []
    for fold in range(N_FOLDS):
        out = df.copy()
        test_cases = {cid for cid, f in assignments.items() if f == fold}
        out["split"] = np.where(out["case_id"].isin(test_cases), "test", "train")
        path = LUAD_DIR / f"tcga_luad_split_cv5_os_balanced_fold{fold}.csv"
        out.to_csv(path, index=False)
        rec = {
            "fold": fold,
            "split_csv": str(path),
            "train": split_stats(out, "train"),
            "test": split_stats(out, "test"),
        }
        records.append(rec)
        t = rec["test"]
        print(
            f"fold {fold}: test {t['n_patients']} pts / {t['n_events']} events "
            f"(rate {t['event_rate']:.3f}), time median {t['time_median']:.1f} "
            f"[{t['time_p25']:.1f}, {t['time_p75']:.1f}] -> {path}",
            flush=True,
        )
    return records


def main() -> None:
    df = pd.read_csv(SRC)
    patients = _patient_table(df)
    patients["stratum"] = make_strata(patients)
    print("strata:", dict(Counter(patients["stratum"])), flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    assignments: dict[str, int] = {}
    X = np.zeros(len(patients))
    y = patients["stratum"].to_numpy()
    for fold, (_train_idx, test_idx) in enumerate(skf.split(X, y)):
        for cid in patients.iloc[test_idx]["case_id"]:
            assignments[cid] = fold
    assert len(assignments) == len(patients)
    folds = write_folds(df, patients, assignments)
    manifest = {
        "base_split_csv": str(SRC),
        "case_column": "case_id",
        "event_column": "OS",
        "time_column": "OS.time",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_time_bins": N_TIME_BINS,
        "min_per_stratum": MIN_PER_STRATUM,
        "protocol": "patient_level_5fold_train_test",
        "stratify": "OS event x OS.time quartile (merged if <5 patients)",
        "n_patients": int(len(patients)),
        "n_slides": int(len(df)),
        "n_events": int(patients["event"].sum()),
        "folds": folds,
    }
    man_path = LUAD_DIR / "tcga_luad_split_cv5_os_balanced_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {man_path}", flush=True)


if __name__ == "__main__":
    main()
