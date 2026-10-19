"""Extract a frozen HE–mIF physical-window patch list for Cell-niches Task 1.

Filter once on native coordinates. All five generators consume the same
``patches.csv``. Do not re-filter after model-specific resize.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import box
from tqdm import tqdm

from .clam_he import (
    is_black_patch,
    is_white_patch,
    patch_in_tissue_contour,
    segment_tissue,
    tissue_fraction,
)
from .io import PyramidTIFF, as_gray, as_rgb_u8, index_wsis
from .mif_qc import build_mif_qc, map_level0_window_to_thumb, saturation_fraction
from .pairing import FOVPair, MIF_MPP_UM, PANEL_CHANNELS, load_fov_pairs

Image.MAX_IMAGE_PIXELS = None


@dataclass
class FilterConfig:
    patch_size_um: float = 256.0
    stride_um: float = 256.0
    mif_mpp_um: float = MIF_MPP_UM
    thumb_max_side: int = 1024
    # CLAM create_patches.py defaults
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    a_t: int = 100
    a_h: int = 16
    max_n_holes: int = 8
    white_thresh: int = 5
    black_thresh: int = 40
    contour_fn: str = "four_pt"
    tissue_min_frac: float = 0.5
    aligned_min_frac: float = 0.95
    stitch_z_thresh: float = 10.0
    stitch_dilate_px: int = 2
    stitch_max_frac: float = 0.03
    dapi_fg_min: float = 0.1
    min_dynamic_range: float = 200.0
    sat_frac_max: float = 0.01
    sat_value: int = 65535
    jpeg_quality: int = 95


def _aligned_to_he_px(aligned, he_mpp_x: float, he_mpp_y: float):
    return affine_transform(aligned, [1000.0 / he_mpp_x, 0.0, 0.0, 1000.0 / he_mpp_y, 0.0, 0.0])


def _iter_polygon_exteriors(geom):
    if geom is None or geom.is_empty:
        return
    geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for g in geoms:
        if g.geom_type != "Polygon" or g.is_empty:
            continue
        yield g.exterior


def _reason_and_keep(row: dict, cfg: FilterConfig) -> tuple[str, bool]:
    if not row["in_bounds_he"]:
        return "oob_he", False
    if not row["in_bounds_mif"]:
        return "oob_mif", False
    if not row["markers_complete"]:
        return "missing_marker", False
    if row["aligned_frac"] < cfg.aligned_min_frac:
        return "aligned", False
    if not row["in_contour"]:
        return "not_in_contour", False
    if row["is_white"]:
        return "white", False
    if row["is_black"]:
        return "black", False
    if row["tissue_frac"] < cfg.tissue_min_frac:
        return "low_tissue", False
    if row["stitch_frac"] > cfg.stitch_max_frac:
        return "stitch", False
    if row["tissue_frac"] >= cfg.tissue_min_frac and row["dapi_fg"] < cfg.dapi_fg_min:
        return "he_mif_disagree", False
    if row["dynamic_range"] < cfg.min_dynamic_range:
        return "low_dynamic_range", False
    if row["sat_frac"] > cfg.sat_frac_max:
        return "saturated", False
    return "keep", True


def _draw_overlay(
    he_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    records: list[dict],
    downsample: tuple[float, float],
    aligned_he,
) -> np.ndarray:
    vis = as_rgb_u8(he_rgb).copy()
    tint = vis.copy()
    tint[tissue_mask > 0] = (0.65 * tint[tissue_mask > 0] + np.array([0, 80, 0])).clip(0, 255)
    vis = tint.astype(np.uint8)
    for ring in _iter_polygon_exteriors(aligned_he):
        xs, ys = ring.xy
        pts = np.stack(
            [np.array(xs) / downsample[0], np.array(ys) / downsample[1]],
            axis=1,
        ).astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 0), 1)
    colors = {
        "keep": (0, 220, 0),
        "stitch": (255, 0, 255),
        "he_mif_disagree": (255, 140, 0),
        "low_tissue": (255, 255, 0),
        "not_in_contour": (255, 255, 0),
        "aligned": (160, 160, 160),
        "white": (255, 255, 255),
        "black": (40, 40, 40),
        "low_dynamic_range": (0, 180, 255),
        "saturated": (255, 0, 0),
    }
    for rec in records:
        x0, y0, x1, y1 = map_level0_window_to_thumb(
            rec["x_he"], rec["y_he"], rec["w_he"], rec["h_he"], downsample, vis.shape[:2]
        )
        color = colors.get(rec["reject_reason"], (255, 0, 0))
        thickness = 2 if rec["keep"] else 1
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    return vis


def _save_example(path: Path, he_crop: np.ndarray, dapi_crop: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    he = as_rgb_u8(he_crop)
    he_s = cv2.resize(he, (128, 128), interpolation=cv2.INTER_AREA)
    if dapi_crop is None or dapi_crop.size == 0:
        Image.fromarray(he_s).save(path)
        return
    d = dapi_crop.astype(np.float32)
    lo, hi = np.percentile(d, (1, 99.9))
    if hi <= lo:
        d8 = np.zeros_like(d, np.uint8)
    else:
        d8 = np.clip((d - lo) / (hi - lo), 0, 1)
        d8 = (d8 * 255).astype(np.uint8)
    d8 = cv2.resize(d8, (128, 128), interpolation=cv2.INTER_AREA)
    d_rgb = np.stack([d8, d8, d8], axis=-1)
    Image.fromarray(np.concatenate([he_s, d_rgb], axis=1)).save(path)


def process_fov(
    fov: FOVPair,
    cfg: FilterConfig,
    out_dir: Path,
    *,
    write_patches: bool,
    max_examples_per_reason: int,
) -> list[dict]:
    if fov.missing_markers:
        raise FileNotFoundError(
            f"{fov.fov_id} missing marker TIFFs: {fov.missing_markers}"
        )
    if fov.dapi_path is None:
        raise FileNotFoundError(f"{fov.fov_id} has no DAPI1/2/3")

    he = PyramidTIFF(fov.he_path)
    dapi = PyramidTIFF(fov.dapi_path)
    ck = PyramidTIFF(fov.marker_paths["CK"]) if "CK" in fov.marker_paths else None
    marker_handles: dict[str, PyramidTIFF] = {}
    try:
        he_level = he.choose_level(cfg.thumb_max_side)
        he_thumb = as_rgb_u8(he.read_level(he_level))
        he_ds = he.downsample_of(he_level)
        tissue = segment_tissue(
            he_thumb,
            downsample=he_ds,
            sthresh=cfg.sthresh,
            mthresh=cfg.mthresh,
            close=cfg.close,
            use_otsu=cfg.use_otsu,
            filter_params={"a_t": cfg.a_t, "a_h": cfg.a_h, "max_n_holes": cfg.max_n_holes},
        )

        dapi_level = dapi.choose_level(cfg.thumb_max_side)
        dapi_thumb = as_gray(dapi.read_level(dapi_level)).astype(np.float32)
        dapi_ds = dapi.downsample_of(dapi_level)
        ck_thumb = None
        if ck is not None:
            ck_level = ck.choose_level(cfg.thumb_max_side)
            ck_thumb = as_gray(ck.read_level(ck_level)).astype(np.float32)
            if ck_thumb.shape != dapi_thumb.shape:
                ck_thumb = cv2.resize(
                    ck_thumb,
                    (dapi_thumb.shape[1], dapi_thumb.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
        stitch_gray = ck_thumb if ck_thumb is not None else dapi_thumb
        mif_qc = build_mif_qc(
            dapi_thumb,
            downsample=dapi_ds,
            ck_thumb=ck_thumb,
            stitch_z_thresh=cfg.stitch_z_thresh,
            stitch_dilate_px=cfg.stitch_dilate_px,
            stitch_gray=stitch_gray,
        )

        scale_x = he.width / float(dapi.width)
        scale_y = he.height / float(dapi.height)
        he_mpp_x = cfg.mif_mpp_um / scale_x
        he_mpp_y = cfg.mif_mpp_um / scale_y
        he_mpp = 0.5 * (he_mpp_x + he_mpp_y)
        size_he = int(round(cfg.patch_size_um / he_mpp))
        stride_he = int(round(cfg.stride_um / he_mpp))
        if size_he < 8 or stride_he < 1:
            raise ValueError(f"degenerate window size_he={size_he} stride_he={stride_he}")
        size_mif_w = int(round(size_he / scale_x))
        size_mif_h = int(round(size_he / scale_y))
        aligned_he = _aligned_to_he_px(fov.aligned, he_mpp_x, he_mpp_y)

        qc_dir = out_dir / "qc" / "overlays"
        qc_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(he_thumb).save(qc_dir / f"{fov.fov_id}_he.png")
        Image.fromarray(tissue.tissue_mask).save(qc_dir / f"{fov.fov_id}_clam_mask.png")
        Image.fromarray(mif_qc.dapi_otsu_u8).save(qc_dir / f"{fov.fov_id}_dapi_u8.png")
        Image.fromarray(mif_qc.dapi_fg_mask).save(qc_dir / f"{fov.fov_id}_dapi_fg.png")
        Image.fromarray(mif_qc.stitch_mask).save(qc_dir / f"{fov.fov_id}_stitch.png")

        records: list[dict] = []
        example_counts: dict[str, int] = {}
        he_dir = out_dir / "he"
        mif_dir = out_dir / "mif"
        if write_patches:
            he_dir.mkdir(parents=True, exist_ok=True)
            mif_dir.mkdir(parents=True, exist_ok=True)

        ys = range(0, he.height - size_he + 1, stride_he)
        xs = range(0, he.width - size_he + 1, stride_he)
        for y_he in ys:
            for x_he in xs:
                w_he = size_he
                h_he = size_he
                x_mif = int(round(x_he / scale_x))
                y_mif = int(round(y_he / scale_y))
                in_bounds_he = x_he >= 0 and y_he >= 0 and x_he + w_he <= he.width and y_he + h_he <= he.height
                in_bounds_mif = (
                    x_mif >= 0
                    and y_mif >= 0
                    and x_mif + size_mif_w <= dapi.width
                    and y_mif + size_mif_h <= dapi.height
                )
                patch_box = box(x_he, y_he, x_he + w_he, y_he + h_he)
                if aligned_he.is_empty or patch_box.area <= 0:
                    aligned_frac = 0.0
                else:
                    aligned_frac = float(aligned_he.intersection(patch_box).area / patch_box.area)

                in_contour = patch_in_tissue_contour(
                    x_he, y_he, size_he, tissue, contour_fn=cfg.contour_fn
                )
                tfrac = tissue_fraction(x_he, y_he, w_he, h_he, tissue)

                hx0, hy0, hx1, hy1 = map_level0_window_to_thumb(
                    x_he, y_he, w_he, h_he, he_ds, he_thumb.shape[:2]
                )
                he_win = he_thumb[hy0:hy1, hx0:hx1]
                white = bool(he_win.size) and is_white_patch(he_win, cfg.white_thresh)
                black = bool(he_win.size) and is_black_patch(he_win, cfg.black_thresh)

                mx0, my0, mx1, my1 = map_level0_window_to_thumb(
                    x_mif, y_mif, size_mif_w, size_mif_h, dapi_ds, dapi_thumb.shape[:2]
                )
                stitch_crop = mif_qc.stitch_mask[my0:my1, mx0:mx1]
                stitch_frac = float(stitch_crop.mean()) / 255.0 if stitch_crop.size else 1.0
                dapi_fg_crop = mif_qc.dapi_fg_mask[my0:my1, mx0:mx1]
                dapi_fg = float(dapi_fg_crop.mean()) / 255.0 if dapi_fg_crop.size else 0.0
                dapi_raw = dapi_thumb[my0:my1, mx0:mx1]
                if dapi_raw.size:
                    p1, p99 = np.percentile(dapi_raw, (1.0, 99.0))
                    dapi_range = float(p99 - p1)
                else:
                    dapi_range = 0.0
                ck_range = 0.0
                if ck_thumb is not None:
                    ck_win = ck_thumb[my0:my1, mx0:mx1]
                    if ck_win.size:
                        c1, c99 = np.percentile(ck_win, (1.0, 99.0))
                        ck_range = float(c99 - c1)
                dynamic_range = max(dapi_range, ck_range)
                sat_frac = saturation_fraction(dapi_raw, cfg.sat_value) if dapi_raw.size else 1.0

                rec = {
                    "fov_id": fov.fov_id,
                    "entity": fov.entity,
                    "x_he": int(x_he),
                    "y_he": int(y_he),
                    "w_he": int(w_he),
                    "h_he": int(h_he),
                    "x_mif": int(x_mif),
                    "y_mif": int(y_mif),
                    "w_mif": int(size_mif_w),
                    "h_mif": int(size_mif_h),
                    "size_um": float(cfg.patch_size_um),
                    "stride_um": float(cfg.stride_um),
                    "he_mpp": float(he_mpp),
                    "he_mpp_x": float(he_mpp_x),
                    "he_mpp_y": float(he_mpp_y),
                    "mif_mpp": float(cfg.mif_mpp_um),
                    "scale_x": float(scale_x),
                    "scale_y": float(scale_y),
                    "in_bounds_he": bool(in_bounds_he),
                    "in_bounds_mif": bool(in_bounds_mif),
                    "markers_complete": True,
                    "aligned_frac": aligned_frac,
                    "in_contour": bool(in_contour),
                    "tissue_frac": tfrac,
                    "is_white": white,
                    "is_black": black,
                    "stitch_frac": stitch_frac,
                    "dapi_fg": dapi_fg,
                    "dapi_dynamic_range": dapi_range,
                    "ck_dynamic_range": ck_range,
                    "dynamic_range": dynamic_range,
                    "sat_frac": sat_frac,
                    "dapi_stain": fov.dapi_stain,
                    "he_path": str(fov.he_path),
                }
                rec["patch_id"] = f"{fov.fov_id[:8]}_x{x_he}_y{y_he}"
                reason, keep = _reason_and_keep(rec, cfg)
                rec["reject_reason"] = reason
                rec["keep"] = keep
                rec["he_relpath"] = ""
                rec["mif_relpath"] = ""

                if keep and write_patches:
                    he_native = as_rgb_u8(he.crop_level0(x_he, y_he, w_he, h_he))
                    if is_white_patch(he_native, cfg.white_thresh):
                        rec["is_white"] = True
                        rec["reject_reason"] = "white"
                        rec["keep"] = False
                    elif is_black_patch(he_native, cfg.black_thresh):
                        rec["is_black"] = True
                        rec["reject_reason"] = "black"
                        rec["keep"] = False
                    else:
                        if not marker_handles:
                            marker_handles = {
                                name: PyramidTIFF(path) for name, path in fov.marker_paths.items()
                            }
                        stack = []
                        for name in PANEL_CHANNELS:
                            ch = as_gray(marker_handles[name].crop_level0(x_mif, y_mif, size_mif_w, size_mif_h))
                            if ch.shape[0] != size_mif_h or ch.shape[1] != size_mif_w:
                                rec["reject_reason"] = "oob_mif"
                                rec["keep"] = False
                                rec["in_bounds_mif"] = False
                                break
                            stack.append(ch.astype(np.uint16, copy=False))
                        if rec["keep"]:
                            vol = np.stack(stack, axis=0)
                            if float(vol.max()) < 1:
                                rec["reject_reason"] = "all_zero"
                                rec["keep"] = False
                            else:
                                he_rel = Path("he") / f"{rec['patch_id']}.png"
                                mif_rel = Path("mif") / f"{rec['patch_id']}.tif"
                                Image.fromarray(he_native).save(out_dir / he_rel)
                                tifffile.imwrite(
                                    out_dir / mif_rel,
                                    vol,
                                    photometric="minisblack",
                                    metadata={
                                        "axes": "CYX",
                                        "channel_names": list(PANEL_CHANNELS),
                                    },
                                )
                                rec["he_relpath"] = str(he_rel)
                                rec["mif_relpath"] = str(mif_rel)

                reason_key = rec["reject_reason"]
                n_ex = example_counts.get(reason_key, 0)
                if n_ex < max_examples_per_reason:
                    example_counts[reason_key] = n_ex + 1
                    _save_example(
                        out_dir / "qc" / "examples" / reason_key / f"{rec['patch_id']}.png",
                        he_win,
                        dapi_raw if dapi_raw.size else None,
                    )
                records.append(rec)

        overlay = _draw_overlay(he_thumb, tissue.tissue_mask, records, he_ds, aligned_he)
        Image.fromarray(overlay).save(qc_dir / f"{fov.fov_id}_overlay.png")
        return records
    finally:
        he.close()
        dapi.close()
        if ck is not None:
            ck.close()
        for handle in marker_handles.values():
            handle.close()


def extract_cohort(
    root: Path,
    out_dir: Path,
    cfg: FilterConfig,
    *,
    write_patches: bool = False,
    fov_ids: list[str] | None = None,
    max_examples_per_reason: int = 12,
) -> pd.DataFrame:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_index = index_wsis(root / "tier1/wsis")
    fovs = load_fov_pairs(root, wsi_index=wsi_index)
    if fov_ids:
        want = {s.lower() for s in fov_ids}
        fovs = [f for f in fovs if f.fov_id.lower() in want or any(f.fov_id.lower().startswith(s) for s in want)]
        if not fovs:
            raise ValueError(f"no FOVs matched {fov_ids}")

    (out_dir / "filter_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (out_dir / "channel_order.json").write_text(
        json.dumps({"channels": list(PANEL_CHANNELS), "dapi": "DAPI1 then DAPI2/3 for QC only"}, indent=2)
        + "\n"
    )

    all_rows: list[dict] = []
    for fov in tqdm(fovs, desc="FOVs"):
        all_rows.extend(
            process_fov(
                fov,
                cfg,
                out_dir,
                write_patches=write_patches,
                max_examples_per_reason=max_examples_per_reason,
            )
        )

    candidates = pd.DataFrame(all_rows)
    candidates_path = out_dir / "candidates.csv"
    patches_path = out_dir / "patches.csv"
    candidates.to_csv(candidates_path, index=False)
    kept = candidates.loc[candidates["keep"]].copy()
    kept.to_csv(patches_path, index=False)

    summary = {
        "n_fov": int(candidates["fov_id"].nunique()) if len(candidates) else 0,
        "n_candidates": int(len(candidates)),
        "n_keep": int(len(kept)),
        "reject_reason_counts": candidates["reject_reason"].value_counts().to_dict() if len(candidates) else {},
        "keep_per_fov": kept.groupby("fov_id").size().to_dict() if len(kept) else {},
        "candidates_csv": str(candidates_path),
        "patches_csv": str(patches_path),
        "write_patches": write_patches,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return candidates
