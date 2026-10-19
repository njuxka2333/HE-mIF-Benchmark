"""Official PathoCell example crop for cell semantic segmentation.

The crop is centered on the ROSIE Dendritic-cell true positive in ``reg005_B``.
Each encoder UNETR infers native windows; softmax probabilities are averaged
inside each GT instance and painted back. Called from ``training.pipeline``.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from data.hooknet_tls_wsd import patch_to_tensor
from data.loader import coerce_label_dict_keys
from data.pathocell_hdf import load_pathocell_volume, pathocell_tile_geometry
from metrics import get_instance_level_labels
from training.example_tile_encoders import bind_no_z0, build_encoder, load_checkpoint, state_has_z0
from training.online_segmentation_train import (
    build_decoder,
    encoder_features,
    forward_segmentation_head,
    load_eval_config,
    load_experiment_config,
    resolve_num_classes,
)
from utils.hf_auth import login_huggingface
from utils.paths import REPO_ROOT, datasets_root, evaluation_root, relocate_dataset_path
from utils.resize_ops import resize_image_np, resize_mask

HDF = relocate_dataset_path(datasets_root() / "PathoCell" / "pathocell_hdf" / "reg005_B.hdf")
FROZEN_CKPT_ROOT = REPO_ROOT / "results" / "cell_semantic_seg" / "pathocell" / "encoder_layer"
NATIVE_MPP = 0.377
CANONICAL_MPP = 0.5
CANONICAL_SIZE = 224
# 224 × 0.5 µm = 112 µm → 297 native pixels at 0.377 mpp, centered on DCs.
DISPLAY_Y0, DISPLAY_X0 = 29, 813
DISPLAY_NATIVE = int(round(CANONICAL_SIZE * CANONICAL_MPP / NATIVE_MPP))
FOV_ID = "reg005_B.hdf"
DC_ID = 4
ALPHA = 0.42
MODELS = ("uni2", "mipheivit", "rosie", "hemit", "gigatime", "histoplexer")
LABELS = {
    "uni2": "UNI2",
    "mipheivit": "MIPHEI-ViT",
    "rosie": "ROSIE",
    "hemit": "HEMIT",
    "gigatime": "GigaTIME",
    "histoplexer": "HistoPlexer",
}
YAML = {
    name: REPO_ROOT / "configs" / "downstream" / "cellseg" / "pathocell" / "encoder" / f"{name}.yaml"
    for name in MODELS
}
CLASS_NAMES = {
    1: "B cells",
    2: "Macrophages/Monocytes",
    3: "Adipocytes",
    4: "Dendritic cells",
    5: "T cells",
    6: "Granulocytes",
    7: "NK cells",
    8: "Nerves",
    9: "Plasma cells",
    10: "Smooth muscle",
    11: "Stroma",
    12: "Tumor cells",
    13: "Vasculature/Lymphatics",
    14: "Other cells",
}
# High-separation hues on pink H&E: DC is cyan, granulocytes stay red.
COLORS = {
    1: (20, 80, 220),      # B cells — blue
    2: (255, 140, 0),      # Macrophages — orange
    3: (255, 220, 0),      # Adipocytes — gold
    4: (0, 230, 255),      # Dendritic cells — cyan
    5: (40, 180, 40),      # T cells — green
    6: (220, 30, 30),      # Granulocytes — red
    7: (160, 50, 210),     # NK cells — purple
    8: (130, 80, 40),      # Nerves — brown
    9: (255, 90, 170),     # Plasma cells — pink
    10: (150, 150, 255),   # Smooth muscle — periwinkle
    11: (140, 140, 140),   # Stroma — gray
    12: (120, 0, 50),      # Tumor cells — maroon
    13: (180, 230, 255),   # Vasculature — ice blue
    14: (170, 170, 40),    # Other cells — olive
}


def _mask_to_color(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb


def _semantic_edges(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    edge = (m != np.roll(m, 1, axis=0)) | (m != np.roll(m, 1, axis=1))
    edge[0, :] = False
    edge[:, 0] = False
    return edge & (m > 0)


def make_overlay(he: np.ndarray, mask: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    base = np.asarray(he, dtype=np.float32)
    over = _mask_to_color(mask).astype(np.float32)
    out = base.copy()
    fg = np.asarray(mask) > 0
    out[fg] = (1.0 - alpha) * base[fg] + alpha * over[fg]
    edge = _semantic_edges(mask)
    out[edge] = 0.35 * out[edge] + 0.65 * np.array([20, 20, 20], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_canonical(he_native: np.ndarray, mask_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    he = resize_image_np(he_native, (CANONICAL_SIZE, CANONICAL_SIZE))
    mask = resize_mask(mask_native, (CANONICAL_SIZE, CANONICAL_SIZE))
    return he, mask


def _save_legend(path: Path) -> None:
    sw, sh, pad, text_w = 28, 22, 8, 280
    h = pad + len(CLASS_NAMES) * (sh + 6) + pad
    w = pad + sw + 10 + text_w + pad
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    y = pad
    for cls, name in CLASS_NAMES.items():
        draw.rectangle([pad, y, pad + sw, y + sh], fill=COLORS[cls], outline=(30, 30, 30))
        draw.text((pad + sw + 10, y + 3), name, fill=(20, 20, 20), font=font)
        y += sh + 6
    img.save(path)


def paint_instance_pred(
    probs_chw: np.ndarray,
    semantic_hw: np.ndarray,
    instance_hw: np.ndarray,
    label_dict: dict,
) -> np.ndarray:
    """PFMB protocol: average class probs inside each GT instance, then argmax."""
    instance = np.asarray(instance_hw)
    out = np.zeros(instance.shape, dtype=np.uint8)
    ids = np.unique(instance)
    ids = ids[ids != 0]
    if ids.size == 0:
        return out
    _, pred_lab = get_instance_level_labels(
        np.asarray(probs_chw, dtype=np.float32),
        np.asarray(semantic_hw),
        instance,
        label_dict,
    )
    pred_lab = np.asarray(pred_lab).reshape(-1)
    if pred_lab.size != ids.size:
        allowed = {int(k) for k in label_dict}
        for iid in ids:
            sel = instance == iid
            cls = int(np.argmax(probs_chw[:, sel].mean(axis=1)))
            out[sel] = cls if cls in allowed else 0
        return out
    for iid, cls in zip(ids, pred_lab):
        out[instance == iid] = int(cls)
    return out


def example_tile_dir() -> Path:
    return evaluation_root() / "cell_semantic_seg" / "pathocell" / "example_tile"


def _ckpt_path(name: str) -> Path:
    cfg = load_eval_config(YAML[name])
    candidates = [
        Path(cfg.output_dir) / "best_model.pt",
        evaluation_root() / "cellseg" / "pathocell" / "encoder" / name / "best_model.pt",
        FROZEN_CKPT_ROOT / name / "best_model.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _patch_hemit_pos_drop(module: torch.nn.Module) -> None:
    """Newer timm Swin drops ``pos_drop``; PathoCell layer hooks still read it."""
    for child in module.modules():
        if type(child).__name__ == "SwinTransformer" and not hasattr(child, "pos_drop"):
            child.pos_drop = torch.nn.Identity()


def _build_encoder(exp, device: torch.device):
    encoder, transform = build_encoder(exp, device)
    if str(exp.model).lower() in {"hemit", "hemmit"}:
        _patch_hemit_pos_drop(encoder)
    return encoder, transform


def _axis_starts(origin: int, length: int, win: int) -> list[int]:
    if win >= length:
        return [origin - (win - length) // 2]
    starts = list(range(origin, origin + length - win + 1, win))
    last = origin + length - win
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_model_window(
    image_hwc: np.ndarray, ty0: int, tx0: int, read_size: int, output_size: int
) -> np.ndarray:
    height, width = image_hwc.shape[:2]
    ty1, tx1 = ty0 + read_size, tx0 + read_size
    sy0, sx0 = max(0, ty0), max(0, tx0)
    sy1, sx1 = min(height, ty1), min(width, tx1)
    he_win = image_hwc[sy0:sy1, sx0:sx1]
    pad = ((sy0 - ty0, ty1 - sy1), (sx0 - tx0, tx1 - sx1), (0, 0))
    if any(v != 0 for pair in pad[:2] for v in pair):
        he_win = np.pad(he_win, pad, mode="constant")
    if he_win.shape[0] != output_size or he_win.shape[1] != output_size:
        he_win = resize_image_np(he_win, (output_size, output_size))
    return he_win


def _softmax_np(logits_chw: np.ndarray) -> np.ndarray:
    x = logits_chw - logits_chw.max(axis=0, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)


def _infer_model(
    name: str,
    image_hwc: np.ndarray,
    display: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    """Return softmax probabilities on the native display crop, shape (C, H, W)."""
    yaml_path = YAML[name]
    exp = load_experiment_config(yaml_path)
    cfg = load_eval_config(yaml_path)
    ckpt_path = _ckpt_path(name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing PathoCell encoder checkpoint: {ckpt_path}")

    encoder, transform = _build_encoder(exp, device)
    native_mpp = float(exp.dataset_config.native_mpp or NATIVE_MPP)
    read_size, output_size = pathocell_tile_geometry(
        int(exp.target_patch_size),
        float(exp.target_mpp) if exp.use_mpp else None,
        native_mpp=native_mpp,
    )
    y0, x0, dsize = display
    y_starts = _axis_starts(y0, dsize, read_size)
    x_starts = _axis_starts(x0, dsize, read_size)
    windows = [(ty, tx) for ty in y_starts for tx in x_starts]
    print(
        f"    {name}: read={read_size} out={output_size} n_windows={len(windows)}",
        flush=True,
    )

    dummy = torch.zeros(1, 3, output_size, output_size)
    first = encoder_features(encoder, dummy, device=device, use_amp=True)
    num_classes = resolve_num_classes(cfg, exp)
    decoder = build_decoder(cfg, exp, first, num_classes=num_classes).to(device).eval()
    blob = load_checkpoint(ckpt_path)
    sd = blob["model_state_dict"]
    if blob.get("feature_layers"):
        cfg.head_config["feature_layers"] = list(blob["feature_layers"])
    if not state_has_z0(sd):
        bind_no_z0(decoder)
        print(f"    {name}: checkpoint has no z0; decoder-blocks-only forward", flush=True)
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"    {name}: load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    dummy_mask = torch.zeros(1, output_size, output_size, dtype=torch.long)
    use_amp = bool((cfg.training or {}).get("amp", True))
    canvas = np.zeros((num_classes, dsize, dsize), dtype=np.float64)
    weight = np.zeros((dsize, dsize), dtype=np.float64)

    for ty0, tx0 in windows:
        he_win = _extract_model_window(image_hwc, ty0, tx0, read_size, output_size)
        images = patch_to_tensor(he_win, transform).unsqueeze(0)
        feats = encoder_features(encoder, images, device=device, use_amp=use_amp)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"
        ):
            logits = forward_segmentation_head(
                decoder, feats, dummy_mask, images=images.to(device, non_blocking=True)
            )
        logits = logits.float()
        if tuple(logits.shape[-2:]) != (read_size, read_size):
            logits = F.interpolate(
                logits, size=(read_size, read_size), mode="bilinear", align_corners=False
            )
        probs = _softmax_np(logits[0].detach().cpu().numpy())
        iy0, ix0 = max(y0, ty0), max(x0, tx0)
        iy1, ix1 = min(y0 + dsize, ty0 + read_size), min(x0 + dsize, tx0 + read_size)
        if iy0 >= iy1 or ix0 >= ix1:
            continue
        canvas[:, iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] += probs[
            :, iy0 - ty0 : iy1 - ty0, ix0 - tx0 : ix1 - tx0
        ]
        weight[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] += 1.0

    if not np.all(weight > 0):
        raise RuntimeError(f"{name}: display crop not fully covered by model windows")
    canvas /= np.clip(weight[None, ...], 1e-8, None)
    del encoder, decoder, images, first, feats, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return canvas.astype(np.float32)


def render_pathocell_example_tile(
    *,
    models: tuple[str, ...] | None = None,
    device=None,
    force: bool = False,
) -> Path:
    OUT = example_tile_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    names = tuple(models) if models else MODELS
    names = tuple(n for n in names if n in MODELS) or MODELS
    assert DISPLAY_NATIVE == 297, DISPLAY_NATIVE
    image_hwc, semantic_hw, instance_hw = load_pathocell_volume(HDF)
    if semantic_hw is None or instance_hw is None:
        raise SystemExit(f"missing semantic/instance mask in {HDF}")
    y0, x0, dsize = DISPLAY_Y0, DISPLAY_X0, DISPLAY_NATIVE
    he_native = image_hwc[y0 : y0 + dsize, x0 : x0 + dsize]
    gt_native = semantic_hw[y0 : y0 + dsize, x0 : x0 + dsize].astype(np.uint8)
    inst_native = instance_hw[y0 : y0 + dsize, x0 : x0 + dsize]
    he, gt = _to_canonical(he_native, gt_native)

    Image.fromarray(he).save(OUT / "he.png")
    Image.fromarray(gt).save(OUT / "gt_mask.png")
    Image.fromarray(make_overlay(he, gt)).save(OUT / "gt_overlay.png")
    _save_legend(OUT / "legend.png")
    present = {int(c): int(n) for c, n in zip(*np.unique(gt, return_counts=True))}
    print(
        f"crop native=({y0}:{y0+dsize},{x0}:{x0+dsize}) display={CANONICAL_SIZE}@{CANONICAL_MPP}mpp "
        f"dc_px={present.get(DC_ID, 0)} classes={sorted(k for k in present if k)}",
        flush=True,
    )

    login_huggingface()
    if device is None:
        device_s = os.environ.get("PATHOCELL_TILE_DEVICE")
        if device_s:
            device = torch.device(device_s)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f"device={device} out={OUT}", flush=True)

    pred_counts = {}
    for name in names:
        overlay_path = OUT / f"{name}_overlay.png"
        pred_path = OUT / f"{name}_pred.png"
        if not force and overlay_path.is_file() and pred_path.is_file():
            pred = np.asarray(Image.open(pred_path), dtype=np.uint8)
            print(f"---- skip {name} (exists)", flush=True)
        else:
            print(f"---- infer {name}", flush=True)
            exp = load_experiment_config(YAML[name])
            label_dict = coerce_label_dict_keys(dict(exp.dataset_config.label_dict))
            probs = _infer_model(name, image_hwc, (y0, x0, dsize), device)
            pred_native = paint_instance_pred(probs, gt_native, inst_native, label_dict)
            _, pred = _to_canonical(he_native, pred_native)
            Image.fromarray(pred).save(pred_path)
            Image.fromarray(make_overlay(he, pred)).save(overlay_path)
            n_inst = int((np.unique(inst_native) != 0).sum())
            print(f"    {name}: instance-averaged {n_inst} cells", flush=True)
        pred_counts[name] = {int(c): int(n) for c, n in zip(*np.unique(pred, return_counts=True))}
        print(f"    {name} classes={sorted(k for k in pred_counts[name] if k)}", flush=True)

    manifest = {
        "fov": FOV_ID,
        "canonical_px": CANONICAL_SIZE,
        "canonical_mpp": CANONICAL_MPP,
        "native_mpp": NATIVE_MPP,
        "native_crop_yx": [y0, x0, dsize, dsize],
        "alpha": ALPHA,
        "gt_class_pixels": present,
        "pred_class_pixels": pred_counts,
        "note": "Predictions are instance-averaged: softmax mean inside each GT instance, then argmax painted back onto the instance mask.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", OUT, flush=True)
    return OUT


if __name__ == "__main__":
    render_pathocell_example_tile()
