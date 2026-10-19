"""
following PrePATH style.

Paired image / mask resizing for MPP adjustment and patch finalization.

Images
  downsample: linear kernel + area interpolation  (OpenCV INTER_AREA)
  upsample:   linear kernel + bicubic interpolation (OpenCV INTER_CUBIC)

Masks (label-safe)
  downsample: nearest-neighbor for label maps; binary maps use area then threshold
  upsample:   nearest-neighbor on the same output grid as the upsampled image
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

SizeWH = Tuple[int, int]


def _size_wh(width: int, height: int) -> SizeWH:
    return int(width), int(height)


def is_downsample(src_wh: SizeWH, dst_wh: SizeWH) -> bool:
    return dst_wh[0] < src_wh[0] or dst_wh[1] < src_wh[1]


def interpolation_for_size(src_wh: SizeWH, dst_wh: SizeWH) -> int:
    """Continuous maps: INTER_AREA when shrinking, INTER_CUBIC when growing."""
    if is_downsample(src_wh, dst_wh):
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


def resize_continuous(arr: np.ndarray, size: SizeWH) -> np.ndarray:
    """Resize a 2-D continuous map (mIF / intensity) with area-down / cubic-up."""
    src = np.asarray(arr)
    if src.ndim != 2:
        raise ValueError(f"resize_continuous expects HxW, got {src.shape}")
    height, width = src.shape
    src_wh = _size_wh(width, height)
    if src_wh == size:
        return np.ascontiguousarray(src, dtype=np.float32)
    out = cv2.resize(
        src.astype(np.float32, copy=False),
        size,
        interpolation=interpolation_for_size(src_wh, size),
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def resize_image(
    image: Union[Image.Image, np.ndarray],
    size: SizeWH,
) -> Image.Image:
    """Resize RGB image with area (down) or bicubic (up) interpolation."""
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
        arr = np.array(pil)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = Image.fromarray(arr)

    src_wh = _size_wh(pil.size[0], pil.size[1])
    if src_wh == size:
        return pil

    out = cv2.resize(arr, size, interpolation=interpolation_for_size(src_wh, size))
    return Image.fromarray(out)


def resize_mask(
    mask: Union[Image.Image, np.ndarray],
    size: SizeWH,
) -> np.ndarray:
    """Resize a single-channel label mask; preserves discrete labels where required."""
    if isinstance(mask, Image.Image):
        arr = np.array(mask)
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        arr = arr[..., 0]

    h, w = arr.shape[:2]
    src_wh = _size_wh(w, h)
    if src_wh == size:
        return arr

    unique_count = len(np.unique(arr))
    is_label_map = arr.dtype in (np.uint16, np.int32, np.int64) or unique_count > 2

    if is_downsample(src_wh, size):
        if is_label_map:
            out = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
        else:
            resized = cv2.resize(arr.astype(np.float32), size, interpolation=cv2.INTER_AREA)
            out = (resized >= 0.5).astype(arr.dtype if arr.dtype != np.bool_ else np.uint8)
    else:
        out = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)

    return out


def resize_image_mask_pair(
    image: Union[Image.Image, np.ndarray],
    mask: Union[Image.Image, np.ndarray, None],
    size: SizeWH,
) -> Tuple[Image.Image, np.ndarray | None]:
    """Apply paired resize so image and mask share identical output dimensions."""
    img_out = resize_image(image, size)
    if mask is None:
        return img_out, None
    mask_out = resize_mask(mask, size)
    return img_out, mask_out


def minimal_cover_positions(length: int, window_size: int) -> List[int]:
    """
    Return the fewest start positions needed to cover a 1D source.

    When ``window_size`` exceeds ``length``, return one centered negative start;
    callers can pad the missing context symmetrically.
    """
    length = max(1, int(length))
    window_size = max(1, int(window_size))
    if window_size >= length:
        return [(length - window_size) // 2]

    count = int(np.ceil(length / window_size))
    if count <= 1:
        return [0]
    max_start = length - window_size
    positions = [int(round(i * max_start / (count - 1))) for i in range(count)]
    return sorted(dict.fromkeys(positions))


def minimal_cover_grid(
    height: int,
    width: int,
    window_size: int,
) -> List[Tuple[int, int]]:
    """Return top-left ``(x, y)`` starts for minimal-overlap full coverage."""
    xs = minimal_cover_positions(width, window_size)
    ys = minimal_cover_positions(height, window_size)
    return [(x, y) for y in ys for x in xs]


def crop_or_pad_image_mask(
    img: Image.Image,
    mask: Optional[np.ndarray],
    left: int,
    top: int,
    crop_size: int,
    *,
    image_fill: Tuple[int, int, int] = (255, 255, 255),
    mask_fill: int = 0,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Crop a square region; pad missing context when the region crosses bounds."""
    crop_size = max(1, int(crop_size))
    left = int(left)
    top = int(top)
    width, height = img.size
    right = left + crop_size
    bottom = top + crop_size

    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, width)
    src_bottom = min(bottom, height)
    dst_left = src_left - left
    dst_top = src_top - top

    canvas = Image.new("RGB", (crop_size, crop_size), image_fill)
    if src_right > src_left and src_bottom > src_top:
        patch = img.crop((src_left, src_top, src_right, src_bottom)).convert("RGB")
        canvas.paste(patch, (dst_left, dst_top))

    if mask is None:
        return canvas, None

    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    out_mask = np.full((crop_size, crop_size), mask_fill, dtype=mask_arr.dtype)
    if src_right > src_left and src_bottom > src_top:
        mask_patch = mask_arr[src_top:src_bottom, src_left:src_right]
        out_mask[
            dst_top : dst_top + mask_patch.shape[0],
            dst_left : dst_left + mask_patch.shape[1],
        ] = mask_patch
    return canvas, out_mask


def mpp_patch_sizes(
    target_patch_size: int,
    target_mpp: Optional[float],
    *,
    native_mpp: float,
    default_patch_size: int,
) -> Tuple[int, int]:
    """
    Return ``(native_read_px, output_px)`` so output is ``output_px`` at ``target_mpp``.

    Physical FOV = output_px * target_mpp = native_read_px * native_mpp.
    """
    if target_patch_size > 0:
        output_size = int(target_patch_size)
    else:
        output_size = int(default_patch_size)

    if target_mpp is not None and target_mpp > 0:
        read_size = max(1, int(round(output_size * float(target_mpp) / float(native_mpp))))
    else:
        read_size = output_size
    return read_size, output_size


def center_crop_image_mask(
    img: Image.Image,
    mask: Optional[np.ndarray],
    crop_size: int,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Center-crop a square window of ``crop_size`` from image and optional mask."""
    width, height = img.size
    crop_size = min(int(crop_size), width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    if mask is not None:
        mask = mask[top : top + crop_size, left : left + crop_size]
    return img, mask


def apply_mpp_patch_geometry(
    img: Image.Image,
    mask: Optional[np.ndarray],
    read_size: int,
    output_size: int,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Center-crop up to ``read_size`` at native resolution, then resize to ``output_size``.

    When the source is smaller than ``read_size``, use the largest square crop available
    (no zero-padding).
    """
    read_size = max(1, int(read_size))
    output_size = max(1, int(output_size))
    crop_size = min(read_size, img.size[0], img.size[1])
    img, mask = center_crop_image_mask(img, mask, crop_size)

    if img.size[0] != output_size or img.size[1] != output_size:
        img, mask = resize_image_mask_pair(img, mask, (output_size, output_size))
    return img, mask


def apply_mpp_patch_geometry_at(
    img: Image.Image,
    mask: Optional[np.ndarray],
    left: int,
    top: int,
    read_size: int,
    output_size: int,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Crop/pad a positioned native read window, then resize to model input size."""
    img, mask = crop_or_pad_image_mask(img, mask, left, top, read_size)
    output_size = max(1, int(output_size))
    if img.size[0] != output_size or img.size[1] != output_size:
        img, mask = resize_image_mask_pair(img, mask, (output_size, output_size))
    return img, mask


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def resize_image_np(image_hwc: np.ndarray, size: SizeWH) -> np.ndarray:
    """Resize RGB HWC uint8 image with area (down) or bicubic (up) interpolation."""
    arr = _as_hwc_uint8(image_hwc)
    height, width = arr.shape[:2]
    src_wh = _size_wh(width, height)
    if src_wh == size:
        return arr

    return cv2.resize(arr, size, interpolation=interpolation_for_size(src_wh, size))


def crop_or_pad_image_mask_np(
    image_hwc: np.ndarray,
    mask: Optional[np.ndarray],
    left: int,
    top: int,
    crop_size: int,
    *,
    image_fill: Tuple[int, int, int] = (255, 255, 255),
    mask_fill: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """NumPy crop/pad using native volume bounds (no PIL roundtrip)."""
    image_hwc = _as_hwc_uint8(image_hwc)
    crop_size = max(1, int(crop_size))
    left = int(left)
    top = int(top)
    height, width = image_hwc.shape[:2]
    right = left + crop_size
    bottom = top + crop_size

    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, width)
    src_bottom = min(bottom, height)
    dst_left = src_left - left
    dst_top = src_top - top

    canvas = np.full((crop_size, crop_size, 3), image_fill, dtype=np.uint8)
    if src_right > src_left and src_bottom > src_top:
        patch = image_hwc[src_top:src_bottom, src_left:src_right]
        canvas[
            dst_top : dst_top + patch.shape[0],
            dst_left : dst_left + patch.shape[1],
        ] = patch

    if mask is None:
        return canvas, None

    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    out_mask = np.full((crop_size, crop_size), mask_fill, dtype=mask_arr.dtype)
    if src_right > src_left and src_bottom > src_top:
        mask_patch = mask_arr[src_top:src_bottom, src_left:src_right]
        out_mask[
            dst_top : dst_top + mask_patch.shape[0],
            dst_left : dst_left + mask_patch.shape[1],
        ] = mask_patch
    return canvas, out_mask


def apply_mpp_patch_geometry_at_np(
    image_hwc: np.ndarray,
    mask: Optional[np.ndarray],
    left: int,
    top: int,
    read_size: int,
    output_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Crop/pad a positioned native read window, then resize to model input size."""
    image_hwc, mask = crop_or_pad_image_mask_np(image_hwc, mask, left, top, read_size)
    output_size = max(1, int(output_size))
    if image_hwc.shape[0] != output_size or image_hwc.shape[1] != output_size:
        image_hwc = resize_image_np(image_hwc, (output_size, output_size))
        if mask is not None:
            mask = resize_mask(mask, (output_size, output_size))
    return image_hwc, mask


def apply_mpp_patch_geometry_triple_at_np(
    image_hwc: np.ndarray,
    semantic_hw: Optional[np.ndarray],
    instance_hw: Optional[np.ndarray],
    left: int,
    top: int,
    read_size: int,
    output_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply identical MPP geometry to RGB + semantic + instance masks in one pass."""
    image_hwc_full = _as_hwc_uint8(image_hwc)
    image_crop, semantic = crop_or_pad_image_mask_np(
        image_hwc_full,
        semantic_hw,
        left,
        top,
        read_size,
    )
    _, instance = crop_or_pad_image_mask_np(
        image_hwc_full,
        instance_hw,
        left,
        top,
        read_size,
    )
    output_size = max(1, int(output_size))
    if image_crop.shape[0] != output_size or image_crop.shape[1] != output_size:
        image_crop = resize_image_np(image_crop, (output_size, output_size))
        if semantic is not None:
            semantic = resize_mask(semantic, (output_size, output_size))
        if instance is not None:
            instance = resize_mask(instance, (output_size, output_size))
    return image_crop, semantic, instance


def tensor_from_image_hwc(image_hwc: np.ndarray, transform) -> "torch.Tensor":
    """Convert HWC uint8 RGB to a model input tensor (single conversion point)."""
    import torch
    from PIL import Image

    image_hwc = _as_hwc_uint8(image_hwc)
    if transform is not None:
        return transform(Image.fromarray(image_hwc, mode="RGB"))
    return torch.from_numpy(image_hwc).permute(2, 0, 1).float() / 255.0
