#!/usr/bin/env python3
"""Recompute Cell-niches significance from the 4 µm cache. No re-inference.

Adds AUPRC shuffle-null + Otsu-positive prevalence, strips leftover P/R/F1@0.5
keys, copies JSON into results/, and rebuilds the three official CSVs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

from training.postprocess import postprocess_virtual_staining
from metrics.vstain import (
    PATCH_UM,
    PROTOCOL_ID,
    REPORT_BIN_UM,
    otsu_threshold,
    pool_to_um,
    ranking_block,
)
from metrics.vstain import (
    DEFAULT_N_BOOT,
    DEFAULT_N_PERM,
    compute_marker_significance,
    significance_ranking_block,
)

CACHE_DIR = REPO / "Evaluation" / "virtual_staining" / "cell_niches" / "cache_4um"
RESULTS = REPO / "results"
MODELS = ("mipheivit", "hemit", "histoplexer", "gigatime", "rosie")
_PRF = (
    "precision_at_0p5_mean_patch",
    "recall_at_0p5_mean_patch",
    "f1_at_0p5_mean_patch",
)
_JSON_NAME = {
    "mipheivit": "mipheivit_vstain_shared_metrics.json",
    "hemit": "vstain_shared_metrics.json",
    "histoplexer": "vstain_shared_metrics.json",
    "gigatime": "vstain_shared_metrics.json",
    "rosie": "vstain_shared_metrics.json",
}


def _pos_frac(gt4: np.ndarray) -> tuple[float, float]:
    fracs = []
    n_pos = n_pix = 0
    for t4 in gt4:
        t8 = pool_to_um(t4, PATCH_UM, REPORT_BIN_UM)
        y = (t8.ravel() > otsu_threshold(t8)).astype(np.int8)
        fracs.append(float(y.mean()))
        n_pos += int(y.sum())
        n_pix += int(y.size)
    mean_p = float(np.mean(fracs)) if fracs else float("nan")
    glob = float(n_pos) / n_pix if n_pix else float("nan")
    return mean_p, glob


def _cd3_patches(npz: np.lib.npyio.NpzFile):
    pred4 = npz["CD3_pred"]
    gt4 = npz["CD3_gt"]
    groups = list(npz["groups"])
    patches = []
    for i in range(pred4.shape[0]):
        patches.append(
            (
                pool_to_um(pred4[i], PATCH_UM, REPORT_BIN_UM),
                pool_to_um(gt4[i], PATCH_UM, REPORT_BIN_UM),
            )
        )
    return patches, groups


def _rescore_model(name: str, n_boot: int, n_perm: int, seed: int, cache_dir: Path) -> Path:
    src = cache_dir / name
    npz_path = src / "pooled_4um.npz"
    json_path = src / _JSON_NAME[name]
    print(f"==== {name}  cache={npz_path}", flush=True)
    npz = np.load(npz_path, allow_pickle=True)
    patches, groups = _cd3_patches(npz)
    significance = compute_marker_significance(
        {"CD3": patches},
        {"CD3": groups},
        markers=("CD3",),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    blob = json.loads(json_path.read_text())
    per_marker = blob.get("per_marker") or {}
    for marker, vals in list(per_marker.items()):
        per_marker[marker] = {k: v for k, v in vals.items() if k not in _PRF}
        gt_key = f"{marker}_gt"
        if gt_key in npz.files:
            mean_p, glob = _pos_frac(npz[gt_key])
            per_marker[marker]["positive_frac_mean_patch"] = mean_p
            per_marker[marker]["positive_frac_global_pixels"] = glob
    ranking = ranking_block(per_marker)
    blob["protocol"] = PROTOCOL_ID
    blob["per_marker"] = per_marker
    blob["ranking"] = ranking
    blob["significance"] = significance
    blob["significance_ranking"] = significance_ranking_block(significance)
    blob["intensity"] = (
        "4 µm mean-pool cache, 8 µm report; Spearman on pooled ranks; "
        "AUPRC vs per-patch Otsu of pooled GT; chance = mean Otsu-positive fraction"
    )
    blob["notes"] = (
        "Compare models on ranking.cd3_spearman and ranking.cd3_auprc only. "
        "SSIM/PSNR and overall_*/core4_* are not cross-model columns. "
        "AUPRC chance is ranking.cd3_positive_frac. Shuffle-null is reported "
        "for Spearman and AUPRC; a p at 1/(n_perm+1) is the permutation floor "
        "and does not rank models. Do not publish P/R/F1@0.5."
    )
    json_path.write_text(json.dumps(blob, indent=2) + "\n")
    csv_name = json_path.with_suffix(".csv")
    pd.DataFrame([{"marker": k, **v} for k, v in per_marker.items()]).to_csv(
        csv_name, index=False
    )
    dest_dir = RESULTS / "virtual_staining" / "cell_niches" / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, dest_dir / json_path.name)
    shutil.copy2(csv_name, dest_dir / csv_name.name)
    print(f"wrote {json_path} -> {dest_dir / json_path.name}", flush=True)
    return json_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Directory of per-model pooled_4um.npz from the frozen shared-protocol run.",
    )
    args = p.parse_args()
    for name in args.models:
        _rescore_model(name, args.n_boot, args.n_perm, args.seed, args.cache_dir)
    postprocess_virtual_staining(SimpleNamespace(results_root=RESULTS))


if __name__ == "__main__":
    main()
