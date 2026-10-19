"""Official TLS example tile, called from ``training.pipeline`` after postprocess."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.hooknet_tls import IGNORE_LABEL, make_overlay
from data.hooknet_tls_sampler import extract_tls_pfmb_tile, tls_source_read_size
from data.hooknet_tls_wsd import patch_to_tensor
from data.pathocell_hdf import pfmb_linspace_tiles
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
from utils.paths import REPO_ROOT, evaluation_root, tls_fixed_grid_root

TILE_ID = "TCGA-NC-A5HO-01Z-00-DX1_ix90_iy26"
REFERENCE_MPP = 0.25
CANONICAL_SIZE = 1024
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
    name: REPO_ROOT / "configs" / "downstream" / "tls" / "encoder" / f"{name}.yaml"
    for name in MODELS
}
FROZEN_CKPT_ROOT = REPO_ROOT / "results" / "tls_seg" / "encoder_layer"


def example_tile_dir() -> Path:
    return evaluation_root() / "tls_seg" / "example_tile"


def _unpad_center(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    read_h, read_w = arr.shape[-2:]
    pad_top = max(0, (read_h - orig_h) // 2)
    pad_left = max(0, (read_w - orig_w) // 2)
    return arr[..., pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def _ckpt_path(name: str) -> Path:
    cfg = load_eval_config(YAML[name])
    candidates = [
        Path(cfg.output_dir) / "best_model.pt",
        evaluation_root() / "tls" / "encoder" / name / "best_model.pt",
        FROZEN_CKPT_ROOT / name / "best_model.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _infer_model(
    name: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    yaml_path = YAML[name]
    exp = load_experiment_config(yaml_path)
    cfg = load_eval_config(yaml_path)
    ckpt_path = _ckpt_path(name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing TLS encoder checkpoint: {ckpt_path}")

    encoder, transform = build_encoder(exp, device)
    read_size = tls_source_read_size(
        int(exp.target_patch_size),
        float(exp.target_mpp),
        reference_mpp=REFERENCE_MPP,
    )
    tiles = pfmb_linspace_tiles(CANONICAL_SIZE, CANONICAL_SIZE, read_size)
    output_size = int(exp.target_patch_size)

    tensors = []
    crops = []
    for tile in tiles:
        image_crop, _mask_crop = extract_tls_pfmb_tile(
            rgb, mask, tile, read_size, output_size
        )
        if transform is None:
            tensor = torch.from_numpy(np.array(image_crop, copy=True)).permute(2, 0, 1).float() / 255.0
        else:
            tensor = patch_to_tensor(image_crop, transform)
        tensors.append(tensor)
        y0, y1, x0, x1 = tile
        crops.append((int(y0), int(y1), int(x0), int(x1)))

    images = torch.stack(tensors, dim=0)
    batch_size = max(1, int((cfg.data or {}).get("test_batch_size") or 1))
    num_classes = resolve_num_classes(cfg, exp)
    use_amp = bool((cfg.training or {}).get("amp", True))

    first = encoder_features(
        encoder, images[:1], device=device, use_amp=use_amp
    )
    decoder = build_decoder(cfg, exp, first, num_classes=num_classes).to(device).eval()
    blob = load_checkpoint(ckpt_path)
    sd = blob["model_state_dict"]
    if blob.get("feature_layers"):
        cfg.head_config["feature_layers"] = list(blob["feature_layers"])
    if not state_has_z0(sd):
        bind_no_z0(decoder)
        print(f"    {name}: checkpoint has no z0; using decoder-blocks-only forward", flush=True)
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"    {name}: load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    canvas = np.zeros((num_classes, CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
    dummy = torch.zeros(1, output_size, output_size, dtype=torch.long)
    n = images.shape[0]
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        batch = images[sl]
        feats = encoder_features(encoder, batch, device=device, use_amp=use_amp)
        masks = dummy.expand(batch.shape[0], -1, -1)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"
        ):
            logits = forward_segmentation_head(
                decoder, feats, masks, images=batch.to(device, non_blocking=True)
            )
        logits = logits.float()
        if tuple(logits.shape[-2:]) != (read_size, read_size):
            logits = F.interpolate(
                logits, size=(read_size, read_size), mode="bilinear", align_corners=False
            )
        logits_np = logits.detach().cpu().numpy()
        for i, (y0, y1, x0, x1) in enumerate(crops[sl]):
            crop = _unpad_center(logits_np[i], y1 - y0, x1 - x0)
            canvas[:, y0:y1, x0:x1] = crop

    pred = np.argmax(canvas, axis=0).astype(np.uint8)
    del encoder, decoder, images, first, feats, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def render_tls_example_tile(
    *,
    models: tuple[str, ...] | None = None,
    device=None,
    force: bool = False,
) -> Path:
    out = example_tile_dir()
    out.mkdir(parents=True, exist_ok=True)
    names = tuple(models) if models else MODELS
    names = tuple(n for n in names if n in MODELS) or MODELS
    grid = tls_fixed_grid_root()
    image_path = grid / "test" / "images" / f"{TILE_ID}.png"
    mask_path = grid / "test" / "masks" / f"{TILE_ID}.png"
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"missing HE/GT PNG for {TILE_ID} under {grid}")

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path), dtype=np.int64)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if rgb.shape[:2] != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise RuntimeError(f"expected {CANONICAL_SIZE}² HE, got {rgb.shape[:2]}")

    Image.fromarray(rgb).save(out / "he.png")
    Image.fromarray(mask.astype(np.uint8)).save(out / "gt_mask.png")
    Image.fromarray(make_overlay(rgb, mask)).save(out / "gt_overlay.png")
    counts = {cls: int(np.count_nonzero(mask == cls)) for cls in (1, 2, 3, IGNORE_LABEL)}
    print(
        f"tile={TILE_ID} he={rgb.shape} tls={counts[1]} gc={counts[2]} "
        f"rest={counts[3]} ignore={counts[IGNORE_LABEL]}",
        flush=True,
    )

    login_huggingface()
    if device is None:
        device_s = os.environ.get("TLS_TILE_DEVICE")
        if device_s:
            device = torch.device(device_s)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f"device={device} out={out}", flush=True)

    pred_counts = {}
    for name in names:
        pred_path = out / f"{name}_pred.png"
        overlay_path = out / f"{name}_overlay.png"
        if not force and pred_path.is_file() and overlay_path.is_file():
            pred = np.asarray(Image.open(pred_path), dtype=np.uint8)
            print(f"---- skip {name} (exists)", flush=True)
        else:
            print(f"---- infer {name}", flush=True)
            pred = _infer_model(name, rgb, mask, device)
            vis = pred.astype(np.int64, copy=True)
            vis[mask == IGNORE_LABEL] = IGNORE_LABEL
            Image.fromarray(pred).save(pred_path)
            Image.fromarray(make_overlay(rgb, vis)).save(overlay_path)
        pred_counts[name] = {
            "tls": int(np.count_nonzero(pred == 1)),
            "gc": int(np.count_nonzero(pred == 2)),
            "rest": int(np.count_nonzero(pred == 3)),
            "other": int(np.count_nonzero((pred != 1) & (pred != 2) & (pred != 3))),
        }
        print(f"    pred counts {pred_counts[name]}", flush=True)

    meta = {
        "tile_id": TILE_ID,
        "split": "test",
        "grid": str(grid),
        "gt_counts": {
            "tls": counts[1],
            "gc": counts[2],
            "rest": counts[3],
            "ignore": counts[IGNORE_LABEL],
        },
        "pred_counts": pred_counts,
        "models": {name: LABELS[name] for name in names},
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out
