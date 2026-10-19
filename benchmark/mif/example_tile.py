"""Official Cell-niches CD3 example tile for virtual staining.

The paper panel is one frozen 256 µm patch (median of 5-model mean CD3
Spearman). Scoring dumps that patch's CD3 map; the pipeline composes the
shared-colormap figure after all generators have run.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("ROSIE_INFER_BATCH", "1024")

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from data.marker_registry import MODEL_GEOMETRY, canonical_marker, resolve_channel_map
from mif.preview import MARKER_CMAP_STOPS, colorize_marker, display_like_gt
from models.mif_feed import apply_feed_env, model_feed, prepare_he_feed, stitch_predictions
from utils.paths import REPO_ROOT, datasets_root, evaluation_root, relocate_dataset_path

EXAMPLE_PATCH_ID = "0f90ab3d_x2303_y13818"
MODELS = ("gigatime", "mipheivit", "hemit", "rosie", "histoplexer")
LABELS = {
    "gigatime": "GigaTIME",
    "mipheivit": "MIPHEI-ViT",
    "hemit": "HEMIT",
    "rosie": "ROSIE",
    "histoplexer": "HistoPlexer",
}


def patches_dir() -> Path:
    return relocate_dataset_path(datasets_root() / "cell_niches_paired_patches")


def example_tile_dir() -> Path:
    return evaluation_root() / "virtual_staining" / "example_tile"


def maps_dir() -> Path:
    return example_tile_dir() / "maps"


def cd3_index(model: str) -> int:
    matches = resolve_channel_map(model, "cell_niches", allow_unverified=True)
    hit = next(m for m in matches if canonical_marker(m.canonical) == "CD3")
    return int(hit.model_index)


def maybe_save_scored_cd3(model: str, patch_id: str, pred_on_gt: np.ndarray, matches) -> None:
    """Persist the official tile's CD3 channel during scoring (already on the GT grid)."""
    if str(patch_id) != EXAMPLE_PATCH_ID:
        return
    hit = next((m for m in matches if canonical_marker(m.canonical) == "CD3"), None)
    if hit is None:
        return
    dest = maps_dir()
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / f"{model}_cd3.npy", np.asarray(pred_on_gt[hit.model_index], dtype=np.float32))


def _eval_roots() -> tuple[Path, ...]:
    """Prefer this rerun's Evaluation, then the frozen results snapshot."""
    seen: list[Path] = []
    for root in (evaluation_root(), REPO_ROOT / "results"):
        if root not in seen:
            seen.append(root)
    return tuple(seen)


def _per_patch_csv(model: str) -> Path | None:
    candidates: list[Path] = []
    for root in _eval_roots():
        candidates.append(root / "virtual_staining" / "cell_niches" / "output" / model / "per_patch_spearman.csv")
        candidates.append(root / "virtual_staining" / "cell_niches" / "cache_4um" / model / "per_patch_spearman.csv")
    candidates.append(REPO_ROOT / "results" / "virtual_staining" / "cell_niches" / model / "per_patch_spearman.csv")
    for csv in candidates:
        if csv.is_file():
            return csv
    return None


def collect_cd3_ranking(models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """Join per-model CD3 Spearman / AUPRC into the example-tile ranking table."""
    parts: list[pd.DataFrame] = []
    for name in models:
        csv = _per_patch_csv(name)
        if csv is None:
            continue
        df = pd.read_csv(csv)
        if "patch_id" not in df.columns or "CD3_spearman" not in df.columns:
            continue
        keep = {"patch_id": df["patch_id"].astype(str)}
        keep[f"{name}_spearman"] = df["CD3_spearman"].astype(float)
        if "CD3_auprc" in df.columns:
            keep[f"{name}_auprc"] = df["CD3_auprc"].astype(float)
        parts.append(pd.DataFrame(keep).drop_duplicates("patch_id"))
    if not parts:
        raise FileNotFoundError(
            "no per_patch_spearman.csv with CD3; run virtual staining scoring first"
        )
    out = parts[0]
    for extra in parts[1:]:
        out = out.merge(extra, on="patch_id", how="outer")
    spearman_cols = [c for c in out.columns if c.endswith("_spearman")]
    auprc_cols = [c for c in out.columns if c.endswith("_auprc")]
    out["mean_spearman"] = out[spearman_cols].mean(axis=1, skipna=True)
    out["min_spearman"] = out[spearman_cols].min(axis=1, skipna=True)
    out["mean_auprc"] = out[auprc_cols].mean(axis=1, skipna=True) if auprc_cols else np.nan
    return out


def _infer_cd3(name: str, rec: dict, he: np.ndarray, device) -> np.ndarray:
    import torch
    from models.mif_output import load_vstain_generator

    geom = MODEL_GEOMETRY[name]
    model_mpp = float(geom["mpp"])
    native_px = int(geom["input_size"])
    feed = model_feed(name, geom)
    apply_feed_env(name, native_px)
    generator, transform = load_vstain_generator(name, device, gpu_num=1)
    generator.eval()
    tiles, coords, tile_px, phys_px = prepare_he_feed(
        name,
        he,
        float(rec["size_um"]),
        model_mpp=model_mpp,
        tile_px=native_px,
        feed=feed,
    )
    xs = torch.stack([transform(Image.fromarray(t)) for t in tiles]).to(device)
    with torch.inference_mode():
        pred = generator(xs).float().cpu().numpy()
    pred_chw = stitch_predictions(name, list(pred), coords, tile_px, phys_px)
    del generator, xs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from mif.resample import align_prediction_to_gt

    gt = tifffile.imread(patches_dir() / rec["mif_relpath"])
    aligned = align_prediction_to_gt(pred_chw, gt)
    return np.asarray(aligned[cd3_index(name)], dtype=np.float32)


def write_cd3_figure(he, gt_cd3, cd3_maps, row, rec, scores, out_dir: Path | None = None) -> Path:
    """Paint every CD3 map with GT's histogram and one colormap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    dest = Path(out_dir) if out_dir is not None else example_tile_dir()
    dest.mkdir(parents=True, exist_ok=True)

    rgb_gt = None
    rgb_models = {}
    for name in MODELS:
        if name not in cd3_maps:
            continue
        gt01, pred01 = display_like_gt(cd3_maps[name], gt_cd3)
        if rgb_gt is None:
            rgb_gt = colorize_marker(gt01)
        rgb_models[name] = colorize_marker(pred01)
    if rgb_gt is None:
        gt01, _ = display_like_gt(gt_cd3, gt_cd3)
        rgb_gt = colorize_marker(gt01)

    panels = [("H&E", he), ("GT CD3", rgb_gt)]
    for name in MODELS:
        if name not in rgb_models:
            continue
        rho = float(row[f"{name}_spearman"]) if f"{name}_spearman" in row.index else float("nan")
        ap = float(row[f"{name}_auprc"]) if f"{name}_auprc" in row.index else float("nan")
        panels.append((f"{LABELS[name]}\nρ={rho:.3f}  AP={ap:.3f}", rgb_models[name]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n + 0.35, 3.05), dpi=300, layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=7, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cmap = LinearSegmentedColormap.from_list("marker", MARKER_CMAP_STOPS)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.018, pad=0.012)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("CD3", fontsize=7)
    rank = int(scores["mean_spearman"].rank(ascending=False, method="first")[row.name])
    n_tiles = int(len(scores))
    fig.suptitle(
        f"CD3, one colormap; predictions histogram-matched to GT  ·  "
        f"{EXAMPLE_PATCH_ID}  ·  {rec['entity']}\n"
        f"ρ / AUPRC are this tile.  tile mean ρ={float(row['mean_spearman']):.3f}",
        fontsize=8,
    )
    fig.savefig(dest / "panel_cd3.png", bbox_inches="tight", facecolor="white")
    fig.savefig(dest / "panel_cd3.pdf", bbox_inches="tight")
    plt.close(fig)

    Image.fromarray(he).save(dest / "panel_he.png")
    Image.fromarray(he).save(dest / "he.png")
    Image.fromarray(rgb_gt).save(dest / "panel_gt_cd3.png")
    for name, rgb in rgb_models.items():
        Image.fromarray(rgb).save(dest / f"panel_{name}_cd3.png")

    meta = {
        "patch_id": EXAMPLE_PATCH_ID,
        "selection_rule": "median of 5-model mean CD3 Spearman over 477 kept patches",
        "entity": rec["entity"],
        "fov_id": rec["fov_id"],
        "rank_by_mean_cd3_spearman": rank,
        "n_tiles": n_tiles,
        "tissue_frac": float(rec["tissue_frac"]),
        "cd3_spearman": {
            m: float(row[f"{m}_spearman"]) for m in MODELS if f"{m}_spearman" in row.index
        },
        "cd3_auprc": {m: float(row[f"{m}_auprc"]) for m in MODELS if f"{m}_auprc" in row.index},
        "mean_cd3_spearman": float(row["mean_spearman"]),
        "min_cd3_spearman": float(row["min_spearman"]) if "min_spearman" in row.index else None,
        "mean_cd3_auprc": float(row["mean_auprc"]) if "mean_auprc" in row.index else None,
        "display": (
            "Shared marker colormap. Each prediction is rank-preserving "
            "histogram-matched to GT CD3 so brightness style matches GT; "
            "spatial pattern is the model's."
        ),
        "note": "Annotated ρ and AUPRC on the panel are this tile's, not cohort means.",
        "he": str(patches_dir() / rec["he_relpath"]),
        "mif": str(patches_dir() / rec["mif_relpath"]),
    }
    (dest / "pick.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest, flush=True)
    return dest


def render_vstain_example_tile(
    *,
    models: tuple[str, ...] | None = None,
    device=None,
    infer_missing: bool = True,
) -> Path:
    """Compose the official CD3 panel. Infers any model whose scored map is missing."""
    import torch
    from utils.hf_auth import login_huggingface

    names = tuple(models) if models else MODELS
    names = tuple(n for n in names if n in MODELS) or MODELS
    dest = example_tile_dir()
    dest.mkdir(parents=True, exist_ok=True)
    ranking = collect_cd3_ranking(MODELS)
    dest_rank = dest / "candidate_cd3_ranking.csv"
    ranking.to_csv(dest_rank, index=False)

    manifest = pd.read_csv(patches_dir() / "patches.csv")
    rec = manifest.loc[manifest["patch_id"] == EXAMPLE_PATCH_ID].iloc[0].to_dict()
    he = np.array(Image.open(patches_dir() / rec["he_relpath"]).convert("RGB"))
    gt = tifffile.imread(patches_dir() / rec["mif_relpath"])
    gt_cd3 = np.asarray(gt[0], dtype=np.float32)  # cell_niches channel 0 is CD3
    if EXAMPLE_PATCH_ID not in set(ranking["patch_id"].astype(str)):
        raise SystemExit(f"example patch {EXAMPLE_PATCH_ID} missing from ranking")
    row = ranking.loc[ranking["patch_id"].astype(str) == EXAMPLE_PATCH_ID].iloc[0]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    cd3_maps = {}
    need_infer = []
    for name in names:
        cached = maps_dir() / f"{name}_cd3.npy"
        if cached.is_file():
            cd3_maps[name] = np.load(cached)
        else:
            need_infer.append(name)
    if need_infer and infer_missing:
        login_huggingface()
        print(f"device={device} patch={EXAMPLE_PATCH_ID} infer={need_infer}", flush=True)
        for name in need_infer:
            print(f"---- infer {name}", flush=True)
            cd3 = _infer_cd3(name, rec, he, device)
            cd3_maps[name] = cd3
            maps_dir().mkdir(parents=True, exist_ok=True)
            np.save(maps_dir() / f"{name}_cd3.npy", cd3)
            print(f"    CD3 {cd3.shape} range=[{cd3.min():.4g},{cd3.max():.4g}]", flush=True)
    elif need_infer:
        print(f"missing CD3 maps (skip infer): {need_infer}", flush=True)

    write_cd3_figure(he, gt_cd3, cd3_maps, row, rec, ranking, out_dir=dest)
    return dest
