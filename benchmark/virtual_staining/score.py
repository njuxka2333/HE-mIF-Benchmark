#!/usr/bin/env python3
"""Score frozen HE→mIF output on Cell-niches (virtual staining).

Same generator as output-layer training (``models.mif_output``). This runner
only loads patches, runs the model, then calls metrics:

* every model: ``metrics.vstain`` (``cell_niches_shared``)
* MIPHEI-ViT extra: Table 4 pixel metrics in the same module

Entry: ``python -m training.train --config .../virtual_staining/...``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from PIL import Image
from tqdm import tqdm

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from data.marker_registry import (  # noqa: E402
    MODEL_GEOMETRY,
    MODEL_PANELS,
    canonical_marker,
    resolve_channel_map,
)
from metrics.vstain import (  # noqa: E402
    DEFAULT_N_BOOT,
    DEFAULT_N_PERM,
    IF_SPAN,
    PATCH_UM,
    PROTOCOL_ID,
    SIGNIFICANCE_MARKERS,
    SharedVStainAccumulator,
    VStainMetrics,
    collect_fov_p99,
    compute_marker_significance,
    describe_geometry,
    filter_kept_manifest,
    pred_to_if,
    ranking_block,
    raw_to_if_orion,
    significance_ranking_block,
    tile_coords,
    tile_pearson,
    write_per_patch_spearman,
    write_pooled_4um_cache,
)
from utils.marker_stack import align_prediction_to_gt  # noqa: E402
from models.mif_feed import apply_feed_env, model_feed, prepare_he_feed, stitch_predictions  # noqa: E402
from models.mif_output import load_vstain_generator  # noqa: E402
from utils.hf_auth import login_huggingface  # noqa: E402
from utils.paths import datasets_root, relocate_dataset_path  # noqa: E402
from utils.resize_ops import resize_continuous  # noqa: E402

MODELS = ("rosie", "hemit", "histoplexer", "gigatime", "mipheivit")
DEFAULT_PATCHES = datasets_root() / "cell_niches_paired_patches"

SHARED_NOTES = (
    "Output channel order is the official training/eval emission list "
    "({source}). Compare models on ranking.cd3_spearman "
    "and ranking.cd3_auprc only. SSIM/PSNR are appendix-only and are not "
    "cross-model columns. ranking.overall_* averages this model's own "
    "markers (coverage 2/12 to 11/12). Cell-niches is a 20-FOV zero-shot "
    "cross-organ cohort. --significance keeps the CD3 shuffled-pairing "
    "null for Spearman and AUPRC (no Spearman CI ranking) and FOV-bootstraps "
    "AUPRC. Report the Otsu-positive fraction as the AUPRC chance line. "
    "A shuffle p at 1/(n_perm+1) is the permutation floor and does not "
    "rank models. Do not publish P/R/F1 at the absolute cutoff 0.5."
)
MIPHEI_SHARED_NOTES = (
    "Shared ruler for cross-model ranking. Orion Table-4 numbers remain in "
    "mipheivit_vstain_metrics.json and are not the comparison protocol. "
    "Compare models on ranking.cd3_spearman and ranking.cd3_auprc at 8 µm. "
    "--significance keeps the CD3 shuffle null for Spearman and AUPRC "
    "(no Spearman CI ranking) and FOV-bootstraps AUPRC. Report the "
    "Otsu-positive fraction as the AUPRC chance line. A shuffle p at "
    "1/(n_perm+1) is the permutation floor and does not rank models."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Downstream YAML under configs/downstream/virtual_staining/cell_niches/output/",
    )
    p.add_argument("--model", default=None, choices=MODELS)
    p.add_argument("--patches-dir", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--patch-ids",
        type=Path,
        default=None,
        help="Optional text/CSV of patch_id values; score only those kept patches.",
    )
    p.add_argument(
        "--no-significance",
        action="store_true",
        help="Skip FOV bootstrap/null even if the YAML sets training.significance.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Metrics output dir (YAML run.output_dir, else Evaluation/...)",
    )
    p.add_argument(
        "--significance",
        action="store_true",
        help=(
            "Also compute the CD3 shuffled-pairing null and FOV-bootstrap AUPRC "
            "(metrics.vstain significance metrics). Spearman has no CI. Point estimates "
            "for every overlap marker are still written."
        ),
    )
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--sig-seed", type=int, default=0)
    return p.parse_args(argv)


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    raw: dict = {}
    output_dir = None
    if args.config is not None:
        from training.config import load_eval_config

        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        output_dir = load_eval_config(args.config).output_dir
    exp = raw.get("experiment") or {}
    data = raw.get("data") or {}
    training = raw.get("training") or {}
    args.model = args.model or exp.get("model")
    if args.model not in MODELS:
        raise SystemExit("pass --config or --model (rosie/hemit/histoplexer/gigatime/mipheivit)")
    args.patches_dir = relocate_dataset_path(
        args.patches_dir or Path(data.get("patches_dir") or DEFAULT_PATCHES)
    )
    default_batch = 4 if args.model == "mipheivit" else 1
    if args.batch_size is None:
        args.batch_size = int(training.get("batch_size") or default_batch)
    if args.out_dir is None:
        args.out_dir = output_dir
    if args.no_significance:
        args.significance = False
    elif not args.significance and bool(training.get("significance", False)):
        args.significance = True
    if args.n_boot == DEFAULT_N_BOOT and training.get("n_boot") is not None:
        args.n_boot = int(training["n_boot"])
    if args.n_perm == DEFAULT_N_PERM and training.get("n_perm") is not None:
        args.n_perm = int(training["n_perm"])
    if args.sig_seed == 0 and training.get("sig_seed") is not None:
        args.sig_seed = int(training["sig_seed"])
    return args


def _match_records(matches) -> list[dict]:
    return [
        {
            "canonical": m.canonical,
            "model_index": m.model_index,
            "dataset_index": m.dataset_index,
            "model_name": m.model_name,
            "dataset_name": m.dataset_name,
        }
        for m in matches
    ]


def _write_miphei_table4(
    *,
    out_dir: Path,
    n_ok: int,
    device,
    model_mpp: float,
    tile_px: int,
    marker_names: list[str],
    tile_pearson_vals: dict[str, list[float]],
    metrics: VStainMetrics,
    matches,
) -> dict:
    results = metrics.results()
    per_marker = {}
    for name in marker_names:
        vals = np.asarray(tile_pearson_vals[name], dtype=np.float64)
        per_marker[name] = {
            "pearson_r": results[name]["pearson_r"],
            "pearson_r_global_pooled": results[name]["pearson_r"],
            "pearson_r_mean_tile": float(np.nanmean(vals)) if vals.size else float("nan"),
            "pearson_r_std_tile": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "psnr": results[name]["psnr"],
            "ssim": results[name]["ssim"],
            "n_pixels": results[name]["n_pixels"],
            "n_tiles": results[name]["n_tiles"],
        }
    pearsons = {k: v["pearson_r"] for k, v in per_marker.items()}
    tile_pearsons = {k: v["pearson_r_mean_tile"] for k, v in per_marker.items()}
    mean_r = float(np.nanmean(list(pearsons.values()))) if pearsons else float("nan")
    mean_tile = float(np.nanmean(list(tile_pearsons.values()))) if tile_pearsons else float("nan")
    summary = {
        "model": "mipheivit",
        "dataset": "cell_niches",
        "n_patches": n_ok,
        "device": str(device),
        "channel_order_verified": True,
        "protocol": "miphei-vit Table 4 / BaseEvaluator._eval_split",
        "geometry": {
            "model_mpp": model_mpp,
            "tile_px": tile_px,
            "patch_um": 256.0,
            "mosaic": "2x2 of 128 µm tiles at 0.5 mpp (training / eval operating point)",
            "scored_grid": "256 px @ 0.5 mpp; GT downsampled from native mIF, not pred upsampled",
        },
        "pred_mapping": "clamp(x, -0.9, 0.9)",
        "gt_mapping": (
            "per-FOV Orion p99 histogram, clip(log1p(x/p99),0,1)*1.8-0.9 "
            "(no AF subtraction)"
        ),
        "psnr_ssim_data_range": IF_SPAN,
        "pearson_aggregation": (
            "global pooled pixels, then mean over overlapping markers "
            "(MIPHEI-ViT Table 4). Mean-of-tiles kept as pearson_r_mean_tile."
        ),
        "mean_pearson_r": mean_r,
        "mean_pearson_r_mean_tile": mean_tile,
        "per_marker": per_marker,
        "matches": _match_records(matches),
        "notes": (
            "Primary Pearson is MIPHEI Table 4: IF-space [-0.9, 0.9], one r over all "
            "eval pixels, then mean over the 9 overlapping markers. Previous "
            "mean-of-tiles / per-tile p1-p99.9 GT protocol is not comparable to "
            "Orion 0.470 / PathoCell 0.336 / HEMIT 0.538."
        ),
    }
    out_json = out_dir / "mipheivit_vstain_metrics.json"
    out_csv = out_dir / "mipheivit_vstain_metrics.csv"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{"marker": k, **v} for k, v in per_marker.items()]).to_csv(out_csv, index=False)
    print(
        json.dumps(
            {
                "mean_pearson_r": mean_r,
                "mean_pearson_r_mean_tile": mean_tile,
                "per_marker_pearson": pearsons,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {out_json} and {out_csv}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _apply_config(parse_args(argv))
    name = args.model
    patches_dir = args.patches_dir
    manifest = pd.read_csv(patches_dir / "patches.csv")
    manifest = manifest.loc[manifest["keep"]].reset_index(drop=True)
    manifest = filter_kept_manifest(manifest, args.patch_ids, args.limit)
    if manifest.empty:
        raise SystemExit(f"no kept patches in {patches_dir / 'patches.csv'}")

    matches = resolve_channel_map(name, "cell_niches", allow_unverified=True)
    if not matches:
        raise SystemExit(f"no overlapping markers between {name} and cell_niches")
    marker_names = [m.canonical for m in matches]
    geom = MODEL_GEOMETRY[name]
    model_mpp = float(geom["mpp"])
    native_px = int(geom["input_size"])
    feed = model_feed(name, geom)

    apply_feed_env(name, native_px)
    login_huggingface()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    generator, transform = load_vstain_generator(name, device, gpu_num=1)
    generator.eval()

    geo0 = describe_geometry(
        model=name,
        model_mpp=model_mpp,
        input_size=native_px,
        feed=feed,
        patch_um=PATCH_UM,
        source_he_px=int(manifest.iloc[0]["w_he"]),
        source_he_mpp=float(manifest.iloc[0]["he_mpp"]),
    )
    print(
        f"model={name} device={device} n_patches={len(manifest)} "
        f"protocol={PROTOCOL_ID} {geo0.get('notes', '')} overlaps={marker_names}",
        flush=True,
    )
    if name == "rosie" and feed == "whole":
        print(
            "rosie inference=paper 8px stride stitch "
            "(128 context, 8×-downsampled map; not evaluate.py gaussian splat)",
            flush=True,
        )
    elif name == "rosie":
        print(
            "rosie inference=128 px @ 0.5 mpp mosaic "
            "(ConvNeXt 128→224 is inside the backbone; no FOV stride-8 stitch)",
            flush=True,
        )
    if name == "mipheivit":
        print(
            f"tile={native_px}px@{model_mpp}mpp (2×2 mosaic at 0.5 mpp)",
            flush=True,
        )

    records = manifest.to_dict(orient="records")
    shared = SharedVStainAccumulator(
        marker_names,
        cache_patches=args.significance,
        cache_markers=SIGNIFICANCE_MARKERS,
    )
    per_patch_ids: list[str] = []
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            BENCHMARK_ROOT.parent
            / "Evaluation"
            / "virtual_staining"
            / "cell_niches"
            / "output"
            / name
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_range = {"min": None, "max": None, "n_ch": None}
    first_pred_hw = None
    first_gt_hw = None
    table4_metrics = None
    tile_pearson_vals: dict[str, list[float]] = {}
    fov_p99: dict[str, dict[int, float]] = {}
    if name == "mipheivit":
        gt_channels = sorted({m.dataset_index for m in matches})
        print("collecting per-FOV Orion p99 on GT…", flush=True)
        fov_p99 = collect_fov_p99(patches_dir, records, gt_channels, n_channels=12)
        table4_metrics = VStainMetrics(marker_names, data_range=IF_SPAN)
        tile_pearson_vals = {m: [] for m in marker_names}

    n_ok = 0

    def update_range(pred: np.ndarray) -> None:
        lo, hi = float(np.nanmin(pred)), float(np.nanmax(pred))
        pred_range["n_ch"] = int(pred.shape[0])
        pred_range["min"] = lo if pred_range["min"] is None else min(pred_range["min"], lo)
        pred_range["max"] = hi if pred_range["max"] is None else max(pred_range["max"], hi)

    def score_pred(rec: dict, pred_chw: np.ndarray) -> None:
        nonlocal first_pred_hw, first_gt_hw
        update_range(pred_chw)
        need = max(m.model_index for m in matches) + 1
        if pred_chw.shape[0] < need:
            raise RuntimeError(
                f"{name} emitted {pred_chw.shape[0]} channels, need index {need - 1}"
            )
        gt = tifffile.imread(patches_dir / rec["mif_relpath"])
        if gt.ndim != 3:
            raise RuntimeError(f"GT {rec['mif_relpath']} shape {gt.shape}")
        if name == "mipheivit":
            p99 = fov_p99[str(rec["fov_id"])]
            phys_px = int(pred_chw.shape[-1])
            pred_if = pred_to_if(pred_chw)
            gt_if = np.empty((gt.shape[0], phys_px, phys_px), dtype=np.float32)
            for ch in range(gt.shape[0]):
                mapped = raw_to_if_orion(gt[ch], p99.get(ch, 1.0))
                gt_if[ch] = resize_continuous(mapped, (phys_px, phys_px))
            for y, x in tile_coords(phys_px, native_px):
                for match in matches:
                    p_tile = pred_if[match.model_index, y : y + native_px, x : x + native_px]
                    g_tile = gt_if[match.dataset_index, y : y + native_px, x : x + native_px]
                    table4_metrics.update(match.canonical, p_tile, g_tile)
                    tile_pearson_vals[match.canonical].append(tile_pearson(p_tile, g_tile))
        first_pred_hw = first_pred_hw or (int(pred_chw.shape[-2]), int(pred_chw.shape[-1]))
        first_gt_hw = first_gt_hw or (int(gt.shape[-2]), int(gt.shape[-1]))
        pred_on_gt = align_prediction_to_gt(pred_chw, gt)
        gt_f32 = np.asarray(gt, dtype=np.float32)
        for match in matches:
            shared.update(
                match.canonical,
                pred_on_gt[match.model_index],
                gt_f32[match.dataset_index],
                group=rec.get("fov_id"),
            )
        per_patch_ids.append(str(rec["patch_id"]))

    pending_x: list[torch.Tensor] = []
    pending_meta: list[tuple[int, tuple[int, int]]] = []
    patch_bufs: dict[int, dict] = {}

    def infer_pending() -> dict[int, list[tuple[tuple[int, int], np.ndarray]]]:
        nonlocal pending_x, pending_meta
        if not pending_x:
            return {}
        x = torch.stack(pending_x, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            pred = generator(x).float().cpu().numpy()
        grouped: dict[int, list[tuple[tuple[int, int], np.ndarray]]] = {}
        for i, (slot, coord) in enumerate(pending_meta):
            grouped.setdefault(slot, []).append((coord, pred[i]))
        pending_x, pending_meta = [], []
        return grouped

    def flush_slot(slot: int, tile_out: list[tuple[tuple[int, int], np.ndarray]]) -> None:
        nonlocal n_ok
        rec = patch_bufs.pop(slot)
        tile_px = int(rec["tile_px"])
        pred_chw = stitch_predictions(
            name, [t[1] for t in tile_out], [t[0] for t in tile_out], tile_px, rec["phys_px"]
        )
        score_pred(rec, pred_chw)
        n_ok += 1
        if n_ok == 1 and name != "mipheivit":
            print(f"first pred shape={tuple(pred_chw.shape)} range={pred_range}", flush=True)

    bar_desc = "MIPHEI" if name == "mipheivit" else name
    for slot, rec in enumerate(tqdm(records, desc=bar_desc)):
        he = np.array(Image.open(patches_dir / rec["he_relpath"]).convert("RGB"))
        patch_um = float(rec["size_um"])
        if name != "mipheivit" and abs(patch_um - PATCH_UM) > 1e-3:
            raise RuntimeError(f"{rec['patch_id']} size_um={patch_um}, expected {PATCH_UM}")
        tiles, coords, tile_px, phys_px = prepare_he_feed(
            name,
            he,
            patch_um,
            model_mpp=model_mpp,
            tile_px=native_px,
            feed=feed,
        )
        patch_bufs[slot] = {**rec, "phys_px": phys_px, "tile_px": tile_px, "n_tiles": len(tiles)}
        for tile, coord in zip(tiles, coords):
            pending_x.append(transform(Image.fromarray(tile)))
            pending_meta.append((slot, coord))
            if len(pending_x) >= args.batch_size:
                for finished_slot, tile_out in infer_pending().items():
                    rec_n = patch_bufs[finished_slot]["n_tiles"]
                    patch_bufs[finished_slot].setdefault("tiles", []).extend(tile_out)
                    if len(patch_bufs[finished_slot]["tiles"]) >= rec_n:
                        flush_slot(finished_slot, patch_bufs[finished_slot]["tiles"])
    for finished_slot, tile_out in infer_pending().items():
        rec_n = patch_bufs[finished_slot]["n_tiles"]
        patch_bufs[finished_slot].setdefault("tiles", []).extend(tile_out)
        if len(patch_bufs[finished_slot]["tiles"]) >= rec_n:
            flush_slot(finished_slot, patch_bufs[finished_slot]["tiles"])
    if patch_bufs:
        raise RuntimeError(f"unscored patches left: {list(patch_bufs)}")

    table4_summary = None
    if name == "mipheivit" and table4_metrics is not None:
        table4_summary = _write_miphei_table4(
            out_dir=out_dir,
            n_ok=n_ok,
            device=device,
            model_mpp=model_mpp,
            tile_px=native_px,
            marker_names=marker_names,
            tile_pearson_vals=tile_pearson_vals,
            metrics=table4_metrics,
            matches=matches,
        )

    per_marker = shared.per_marker()
    ranking = ranking_block(per_marker)
    if name == "mipheivit":
        geo0 = describe_geometry(
            model="mipheivit",
            model_mpp=model_mpp,
            input_size=native_px,
            feed="mosaic",
            source_he_px=int(manifest.iloc[0]["w_he"]),
            source_he_mpp=float(manifest.iloc[0]["he_mpp"]),
            pred_hw=(native_px * 2, native_px * 2),
            gt_hw=first_gt_hw,
        )
        shared_json = out_dir / "mipheivit_vstain_shared_metrics.json"
        shared_csv = out_dir / "mipheivit_vstain_shared_metrics.csv"
    else:
        geo0["pred_hw"] = list(first_pred_hw) if first_pred_hw else None
        geo0["gt_hw"] = list(first_gt_hw) if first_gt_hw else None
        shared_json = out_dir / "vstain_shared_metrics.json"
        shared_csv = out_dir / "vstain_shared_metrics.csv"
    write_per_patch_spearman(
        out_dir / "per_patch_spearman.csv",
        per_patch_ids,
        shared.spearman_patches,
        auprc_patches=shared.auprc_patches,
        spearman_4um_patches=shared.spearman_4um_patches,
    )
    write_pooled_4um_cache(
        out_dir / "pooled_4um.npz",
        per_patch_ids,
        shared.pooled_4um,
        groups=(shared.patch_groups.get("CD3") or None),
    )

    def write_summary(significance, significance_ranking) -> None:
        if name == "mipheivit":
            summary = {
                "model": "mipheivit",
                "dataset": "cell_niches",
                "n_patches": n_ok,
                "device": str(device),
                "channel_order_verified": True,
                "protocol": PROTOCOL_ID,
                "geometry": geo0,
                "intensity": (
                    "4 µm mean-pool cache, 8 µm report; Spearman on pooled ranks; "
                    "AUPRC vs per-patch Otsu of pooled GT"
                ),
                "ranking": ranking,
                "per_marker": per_marker,
                "significance": significance,
                "significance_ranking": significance_ranking,
                "matches": (table4_summary or {}).get("matches") or _match_records(matches),
                "notes": MIPHEI_SHARED_NOTES,
            }
            print_payload = {
                "shared_ranking": ranking,
                "shared_significance_ranking": significance_ranking,
            }
        else:
            summary = {
                "model": name,
                "dataset": "cell_niches",
                "n_patches": n_ok,
                "device": str(device),
                "channel_order_verified": bool(MODEL_PANELS[name].verified),
                "protocol": PROTOCOL_ID,
                "geometry": geo0,
                "pred_range": pred_range,
                "intensity": (
                    "4 µm mean-pool cache, 8 µm report; Spearman on pooled ranks; "
                    "AUPRC vs per-patch Otsu of pooled GT"
                ),
                "ranking": ranking,
                "per_marker": per_marker,
                "significance": significance,
                "significance_ranking": significance_ranking,
                "matches": _match_records(matches),
                "notes": SHARED_NOTES.format(source=MODEL_PANELS[name].source),
            }
            print_payload = {
                "ranking": ranking,
                "significance_ranking": significance_ranking,
                "pred_range": pred_range,
            }
        shared_json.write_text(json.dumps(summary, indent=2) + "\n")
        pd.DataFrame([{"marker": k, **v} for k, v in per_marker.items()]).to_csv(shared_csv, index=False)
        print(json.dumps(print_payload, indent=2), flush=True)
        print(f"wrote {shared_json} and {shared_csv}", flush=True)

    write_summary(None, None)
    if args.significance:
        significance = compute_marker_significance(
            shared.raw_patches,
            shared.patch_groups,
            markers=SIGNIFICANCE_MARKERS,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.sig_seed,
        )
        write_summary(significance, significance_ranking_block(significance))

if __name__ == "__main__":
    main()
