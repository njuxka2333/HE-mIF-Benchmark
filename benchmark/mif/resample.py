"""Resample generated mIF stacks to a unified UNet grid.

Matches the image convention in ``utils.resize_ops``:

* downsample → area (OpenCV ``INTER_AREA`` / ``F.interpolate(..., mode="area")``)
* upsample   → bicubic (OpenCV ``INTER_CUBIC`` / ``mode="bicubic"``)

Masks stay on the label-safe path in ``resize_mask`` (nearest). This module is
only for float marker stacks (3–50 channels), which ``resize_image`` cannot take
because it forces RGB uint8 via PIL.

Protocol (virtual-stain segmentation)
-------------------------------------
1. Run the HE→mIF foundation model at **its** expected size / mpp.
2. Resample the generated stack to the **task-unified** UNet size
   (cellseg 256, TLS 512) **before** z-scoring.
3. Train the UNet on that unified grid; masks are resized with ``resize_mask``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "CELLSEG_UNET_SIZE",
    "TLS_UNET_SIZE",
    "align_prediction_to_gt",
    "resample_mif_stack",
    "resample_mif_stack_np",
]

# Task-unified UNet spatial sizes (after generation / FOV crop).
CELLSEG_UNET_SIZE = 256
TLS_UNET_SIZE = 512


def resample_mif_stack(
    stack: torch.Tensor,
    target_size: int | tuple[int, int] | None,
    *,
    upsample_mode: str = "bicubic",
    clamp: Sequence[float] | None = None,
) -> torch.Tensor:
    """Resample ``[B,C,H,W]`` (or ``[C,H,W]``) float stack to ``target_size``.

    ``None`` / matching size → identity. Downsample uses ``area``; upsample uses
    ``upsample_mode`` (``bicubic`` default, or ``bilinear`` to avoid overshoot).
    Optional ``clamp=(lo, hi)`` after upsample (e.g. ``(-1, 1)`` for tanh models).
    """
    if target_size is None:
        return stack
    if isinstance(target_size, int):
        th = tw = int(target_size)
    else:
        th, tw = int(target_size[0]), int(target_size[1])
    if th <= 0 or tw <= 0:
        raise ValueError(f"invalid target_size={target_size!r}")

    squeezed = False
    if stack.dim() == 3:
        stack = stack.unsqueeze(0)
        squeezed = True
    if stack.dim() != 4:
        raise ValueError(f"expected [B,C,H,W] or [C,H,W], got {tuple(stack.shape)}")

    h, w = int(stack.shape[-2]), int(stack.shape[-1])
    if h == th and w == tw:
        return stack.squeeze(0) if squeezed else stack

    if h > th or w > tw:
        # Shrinking either side → area (anti-aliased). If one side grows and the
        # other shrinks, area is still the conservative choice for anti-aliasing.
        out = F.interpolate(stack, size=(th, tw), mode="area")
    else:
        mode = str(upsample_mode).lower()
        if mode not in {"bicubic", "bilinear", "nearest"}:
            raise ValueError(f"unsupported upsample_mode={upsample_mode!r}")
        kwargs = {"align_corners": False} if mode in {"bicubic", "bilinear"} else {}
        out = F.interpolate(stack, size=(th, tw), mode=mode, **kwargs)
        if clamp is not None:
            lo, hi = float(clamp[0]), float(clamp[1])
            out = out.clamp_(lo, hi)
    return out.squeeze(0) if squeezed else out


def resample_mif_stack_np(
    stack: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    upsample_interp: int | None = None,
) -> np.ndarray:
    """NumPy CHW float resize with the same area-down / cubic-up convention.

    Used by offline FOV tile crops (CPU). Prefer ``resample_mif_stack`` on GPU
    mid-forward when the tensor is already on device.
    """
    import cv2

    if stack.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {stack.shape}")
    target_h = max(1, int(target_h))
    target_w = max(1, int(target_w))
    c, h, w = stack.shape
    if h == target_h and w == target_w:
        return stack

    if h > target_h or w > target_w:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC if upsample_interp is None else int(upsample_interp)

    src = stack.astype(np.float32, copy=False)
    # OpenCV's packed resize only handles ≤4 channels; marker stacks are often 5–50.
    if c <= 4:
        hwc = np.transpose(src, (1, 2, 0))
        resized = cv2.resize(hwc, (target_w, target_h), interpolation=interp)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return np.ascontiguousarray(np.transpose(resized, (2, 0, 1)).astype(np.float32))
    planes = [
        cv2.resize(src[i], (target_w, target_h), interpolation=interp) for i in range(c)
    ]
    return np.ascontiguousarray(np.stack(planes, axis=0).astype(np.float32))


def align_prediction_to_gt(pred_chw: np.ndarray, gt_chw: np.ndarray) -> np.ndarray:
    """Resample generated markers onto the GT spatial grid before scoring.

    Virtual-staining generators emit at their own operating point (ROSIE 128,
    HistoPlexer 256, MIPHEI/GigaTIME 512, HEMIT 1024). Cell-niches GT is the
    native mIF crop (typically 788 px at 0.325 mpp for a 256 µm window). Metrics
    compare against that GT tensor, so the prediction is warped onto ``gt``
    (area down / cubic up), not the other way around.
    """
    pred = np.asarray(pred_chw)
    gt = np.asarray(gt_chw)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"expected CHW pred/gt, got pred={pred.shape} gt={gt.shape}")
    _pc, gh, gw = int(gt.shape[0]), int(gt.shape[-2]), int(gt.shape[-1])
    if gh < 1 or gw < 1:
        raise ValueError(f"invalid GT spatial size {(gh, gw)}")
    if pred.shape[-2] == gh and pred.shape[-1] == gw:
        return np.ascontiguousarray(pred, dtype=np.float32)
    return resample_mif_stack_np(pred.astype(np.float32, copy=False), gh, gw)
