"""
ROSIE baseline aligned with the official repo (rosie/) and the paper.

Two inference modes:

1. **Patch / bag / cell** (H,W ≤ 128): one ConvNeXt classifier forward per
   image → 50-D head, broadcast to ``[B, 50, H, W]``. Matches ``train.py``
   (128→224→50-D).

2. **Large FOV / WSI** (H or W > 128): paper-standard **8 px stride stitch**.
   Each 128×128 context predicts the center 8×8 mean; those predictions are
   assembled into an 8×-downsampled map (1024 H&E → 128 pred). This is the
   setting used for Pearson / SSIM in Nature Communications 2025, not the
   denser Gaussian splat in ``evaluate.py`` (overlap_stride = stride//2).

Input to wrapper: ``[B, 3, H, W]`` in [0, 1] (``get_trans()`` = ToTensor).
Output: ``[B, 50, H, W]`` (≤128) or ``[B, 50, ceil(H/8), ceil(W/8)]`` (FOV).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from models.patch_compat import log_native_64_patch, spatial_hw

from utils.paths import reference_model_path

_ROSIE_ROOT = reference_model_path("rosie")
if str(_ROSIE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROSIE_ROOT))

PATCH_SIZE = 128
PATCH_HALF = PATCH_SIZE // 2
CNN_INPUT_SIZE = 224
NUM_CHANNELS = 50
# Paper analyses: 8 px non-overlapping stride, one pred per 8×8 block.
GRID_STRIDE = 8
INFER_BATCH = int(os.environ.get("ROSIE_INFER_BATCH", "512"))
_STITCH_LOGGED = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pad_patch(
    patch: np.ndarray,
    original_size: Tuple[int, int],
    x_center: int,
    y_center: int,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]
    if current_height == patch_size and current_width == patch_size:
        return patch

    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)
    pad_shape = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
        (0, 0),
    )
    padded_patch = np.pad(patch, pad_shape, mode="constant", constant_values=0)
    return padded_patch[:patch_size, :patch_size]


def _build_weight_kernel(stride_size: int = GRID_STRIDE) -> np.ndarray:
    """Gaussian blend kernel; size follows official ``kernel_size = stride_size * 2``."""
    kernel_size = stride_size * 2
    y_grid, x_grid = np.mgrid[0:kernel_size, 0:kernel_size]
    center = kernel_size // 2
    return np.exp(
        -((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * (kernel_size / 4) ** 2)
    ).astype(np.float32)


def _load_rosie_backbone(ckpt_path: str, num_outputs: int) -> nn.Module:
    model = models.convnext_small(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "ROSIE checkpoint loaded with non-strict key differences: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model


def _tensor_to_hwc_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert [3, H, W] tensor in [0, 1] to HxWx3 float numpy in ~[0, 255]."""
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() > 1.5:
        arr = img.permute(1, 2, 0).cpu().numpy()
    else:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255)
    return arr


def _prep_he_tensor_for_backbone(he_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8-ish HWC → normalized CHW 224 (official evaluate transforms)."""
    resize = transforms.Resize(CNN_INPUT_SIZE, antialias=True)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    patch_tensor = torch.from_numpy(np.asarray(he_hwc)).permute(2, 0, 1).float() / 255.0
    return normalize(resize(patch_tensor)).to(device)


def _predict_head_batch(
    backbone: nn.Module,
    batch_hwc: List[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Run ConvNeXt classifier on a list of HWC patches → [N, C] numpy."""
    tensors = [_prep_he_tensor_for_backbone(p, device) for p in batch_hwc]
    batch = torch.stack(tensors, dim=0)
    backbone.eval()
    with torch.no_grad():
        return backbone(batch).detach().cpu().numpy()


def _head_map_single_image(
    backbone: nn.Module,
    img_hwc: np.ndarray,
    device: torch.device,
    num_channels: int,
) -> torch.Tensor:
    """
    Official training-style inference for one patch-sized image.

    One center context crop (full image if ≤128) → head → broadcast to HxW.
    GAP of the result equals the 50-D classifier output.
    """
    height, width = img_hwc.shape[0], img_hwc.shape[1]
    context = min(PATCH_SIZE, height, width)
    patch_half = context // 2
    y_center, x_center = height // 2, width // 2
    channels = [img_hwc[:, :, c] for c in range(3)]
    b = np.clip(y_center - patch_half, 0, height)
    t = np.clip(y_center + patch_half, 0, height)
    l = np.clip(x_center - patch_half, 0, width)
    r = np.clip(x_center + patch_half, 0, width)
    he_patch = np.stack([ch[b:t, l:r] for ch in channels], axis=-1)
    he_patch = _pad_patch(he_patch, (height, width), x_center, y_center, patch_size=context)
    pred = _predict_head_batch(backbone, [he_patch], device)[0]
    out = np.empty((num_channels, height, width), dtype=np.float32)
    out[:] = pred[:, None, None]
    return torch.from_numpy(out).unsqueeze(0)


def _paper_stitch_tensor(
    backbone: nn.Module,
    x: torch.Tensor,
    num_channels: int,
    stride_size: int = GRID_STRIDE,
    context_patch_size: int = PATCH_SIZE,
    infer_batch: int = INFER_BATCH,
) -> torch.Tensor:
    """Paper-standard 8 px stitch on a GPU batch ``[B, 3, H, W]`` in ``[0, 1]``.

    Matches Nature Communications 2025: 128 context, stride 8, output is the
    8×-downsampled map (each 8×8 block → one pixel). ConvNeXt sees 224² after
    bilinear resize and ImageNet normalize, same as ``train.py`` eval transforms.
    """
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"ROSIE stitch expects [B, 3, H, W], got {tuple(x.shape)}")
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.float()
    bsz, _, height, width = x.shape
    half = context_patch_size // 2
    n_h = (height - 1) // stride_size + 1
    n_w = (width - 1) // stride_size + 1
    n_win = bsz * n_h * n_w
    infer_batch = max(1, int(os.environ.get("ROSIE_INFER_BATCH", str(infer_batch))))
    global _STITCH_LOGGED
    if not _STITCH_LOGGED:
        n_fwd = (n_win + infer_batch - 1) // infer_batch
        print(
            f"rosie stitch {n_h}x{n_w}={n_win} windows (spatial grid, not per-channel); "
            f"ConvNeXt head emits {num_channels} markers each forward; "
            f"batch={infer_batch} -> {n_fwd} forwards/FOV",
            flush=True,
        )
        _STITCH_LOGGED = True
    xpad = F.pad(x, (half, half, half, half), mode="constant", value=0.0)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    preds = x.new_empty((n_win, num_channels))
    k = context_patch_size
    use_amp = bool(x.is_cuda)
    backbone.eval()
    # Stream grid-rows so we never materialize all 16k 128² crops at once.
    windows_per_row = bsz * n_w
    rows_per = max(1, infer_batch // max(windows_per_row, 1))
    filled = 0
    with torch.inference_mode():
        for row0 in range(0, n_h, rows_per):
            n_rows = min(rows_per, n_h - row0)
            y0 = row0 * stride_size
            block_h = k + (n_rows - 1) * stride_size
            windows = (
                xpad[:, :, y0 : y0 + block_h, :]
                .unfold(2, k, stride_size)
                .unfold(3, k, stride_size)[:, :, :, :n_w]
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(-1, 3, k, k)
            )
            for start in range(0, windows.shape[0], infer_batch):
                sl = windows[start : start + infer_batch]
                sl = F.interpolate(
                    sl,
                    size=CNN_INPUT_SIZE,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                sl = (sl - mean) / std
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = backbone(sl)
                else:
                    out = backbone(sl)
                n = int(out.shape[0])
                preds[filled : filled + n] = out.float()
                filled += n
    if filled != n_win:
        raise RuntimeError(f"ROSIE stitch filled {filled} preds, expected {n_win}")
    return preds.view(bsz, n_h, n_w, num_channels).permute(0, 3, 1, 2).contiguous()


def _splat_single_image(
    backbone: nn.Module,
    img_hwc: np.ndarray,
    device: torch.device,
    num_channels: int,
    stride_size: int,
    weight_kernel: np.ndarray,
    context_patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """
    Official ``evaluate.process_image`` splat for a large image.

    Grid uses ``overlap_stride = max(1, stride_size // 2)``; kernel uses
    ``stride_size * 2`` (same as evaluate.py).
    """
    height, width = img_hwc.shape[0], img_hwc.shape[1]
    patch_half = context_patch_size // 2
    channels = [img_hwc[:, :, c] for c in range(3)]
    # Match evaluate.py: overlap_stride = max(1, stride_size // 2)
    overlap_stride = max(1, stride_size // 2)

    coords: List[Tuple[int, int]] = []
    for y in range(0, height, overlap_stride):
        for x in range(0, width, overlap_stride):
            coords.append((x, y))

    raw_output = np.zeros((num_channels, height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    half_size = (stride_size * 2) // 2

    backbone.eval()
    with torch.no_grad():
        for start in range(0, len(coords), INFER_BATCH):
            batch_coords = coords[start : start + INFER_BATCH]
            batch_hwc = []
            for x_center, y_center in batch_coords:
                b = np.clip(y_center - patch_half, 0, height)
                t = np.clip(y_center + patch_half, 0, height)
                l = np.clip(x_center - patch_half, 0, width)
                r = np.clip(x_center + patch_half, 0, width)
                he_patch = np.stack([ch[b:t, l:r] for ch in channels], axis=-1)
                he_patch = _pad_patch(
                    he_patch, (height, width), x_center, y_center, patch_size=context_patch_size
                )
                batch_hwc.append(he_patch)

            predictions = _predict_head_batch(backbone, batch_hwc, device)

            for pred, (x_center, y_center) in zip(predictions, batch_coords):
                t = np.clip(y_center - half_size, 0, height)
                b = np.clip(y_center + half_size, 0, height)
                l = np.clip(x_center - half_size, 0, width)
                r = np.clip(x_center + half_size, 0, width)
                kernel_h, kernel_w = b - t, r - l
                weight = weight_kernel[:kernel_h, :kernel_w]
                for c in range(num_channels):
                    raw_output[c, t:b, l:r] += pred[c] * weight
                weight_map[t:b, l:r] += weight

    weight_map = np.maximum(weight_map, 1e-8)
    for c in range(num_channels):
        raw_output[c] /= weight_map

    return torch.from_numpy(raw_output).unsqueeze(0)


class ROSIESpatialWrapper(nn.Module):
    """
    ROSIE virtual-stain wrapper.

    - Patch-sized inputs (≤128): single head forward (train.py semantics).
    - Larger inputs: paper 8 px stride stitch (8×-downsampled 50-channel map).
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_channels: int = NUM_CHANNELS,
        stride_size: int = GRID_STRIDE,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels
        self.stride_size = stride_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"ROSIE expects [B, 3, H, W], got {tuple(x.shape)}")

        log_native_64_patch("ROSIE", x)
        device = x.device
        h, w = spatial_hw(x)
        use_head_only = h <= PATCH_SIZE and w <= PATCH_SIZE

        if use_head_only:
            outputs = []
            for b in range(x.shape[0]):
                img_hwc = _tensor_to_hwc_numpy(x[b])
                outputs.append(
                    _head_map_single_image(self.backbone, img_hwc, device, self.num_channels)
                )
            return torch.cat(outputs, dim=0).to(device=device, dtype=x.dtype)

        return _paper_stitch_tensor(
            self.backbone,
            x,
            self.num_channels,
            stride_size=self.stride_size,
        ).to(device=device, dtype=x.dtype)


def get_model(
    device: torch.device,
    ckpt_path: str,
    gpu_num: int = 1,
    num_channels: int = NUM_CHANNELS,
) -> nn.Module:
    del gpu_num
    backbone = _load_rosie_backbone(ckpt_path, num_channels)
    model = ROSIESpatialWrapper(backbone, num_channels=num_channels)
    return model.to(device).eval()


def get_trans():
    # Full patch retained; wrapper resizes context to 224 internally.
    return transforms.Compose([transforms.ToTensor()])
