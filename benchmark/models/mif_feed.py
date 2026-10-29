"""Paper-native HE feeds for frozen HE→mIF generators.

Virtual-staining scoring and the standalone example renderer use this. Output-layer
segmentation still runs one tile at a time through ``FrozenMifGenerator``; the
geometry here is how a 256 µm Cell-niches window is presented to each model.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import cv2
import numpy as np

from utils.resize_ops import resize_image_np

__all__ = [
    "apply_feed_env",
    "he_at_model_mpp",
    "he_mosaic_at_model_mpp",
    "model_feed",
    "mosaic_tiles",
    "prepare_he_feed",
    "purge_histoplexer_src_shadow",
    "stitch_predictions",
]


def expected_he_px(patch_um: float, model_mpp: float) -> int:
    return max(1, int(round(float(patch_um) / float(model_mpp))))


def model_feed(model_name: str, geom: dict) -> str:
    name = str(model_name).strip().lower()
    default = "whole" if name in ("hemit", "histoplexer") else "mosaic"
    return str(geom.get("feed") or default)


def apply_feed_env(model_name: str, native_px: int) -> None:
    """Wrapper-side canvas sizes. MIPHEI-ViT does not use these env vars."""
    name = str(model_name).strip().lower()
    if name == "mipheivit":
        return
    os.environ["HEMIT_TARGET_SIZE"] = str(native_px)
    os.environ["HISTOPLEXER_TARGET_SIZE"] = str(native_px)
    if name == "gigatime":
        os.environ["GIGATIME_WINDOW_SIZE"] = "256"


def purge_histoplexer_src_shadow() -> None:
    """HistoPlexer lives at ``HistoPlexer/src.models``; MIPHEI-ViT has ``src/models.py``."""
    from utils.paths import reference_model_path

    root = str(reference_model_path("histoplexer"))
    drop = []
    for path in sys.path:
        if "MIPHEI-ViT" in str(path).replace("\\", "/"):
            drop.append(path)
    for path in drop:
        sys.path.remove(path)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


# Back-compat name used by older scripts.
_purge_src_shadow = purge_histoplexer_src_shadow


def he_at_model_mpp(he_rgb: np.ndarray, patch_um: float, model_mpp: float) -> tuple[np.ndarray, int]:
    phys_px = expected_he_px(patch_um, model_mpp)
    phys = resize_image_np(he_rgb, (phys_px, phys_px))
    return phys, phys_px


def mosaic_tiles(phys: np.ndarray, tile_px: int) -> tuple[list[np.ndarray], list[tuple[int, int]], int]:
    h, _w = phys.shape[:2]
    phys_px = h
    pad = (tile_px - phys_px % tile_px) % tile_px
    if pad:
        phys = cv2.copyMakeBorder(phys, 0, pad, 0, pad, cv2.BORDER_REFLECT)
    tiles, coords = [], []
    ph, pw = phys.shape[:2]
    for y in range(0, ph, tile_px):
        for x in range(0, pw, tile_px):
            tiles.append(phys[y : y + tile_px, x : x + tile_px])
            coords.append((y, x))
    return tiles, coords, phys_px


def he_mosaic_at_model_mpp(
    he_rgb: np.ndarray,
    *,
    patch_um: float,
    model_mpp: float,
    tile_px: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]], int]:
    """MIPHEI-ViT: resize the 256 µm window to 0.5 mpp and split into 256 px tiles."""
    phys_px = max(tile_px, int(round(float(patch_um) / float(model_mpp))))
    phys = resize_image_np(he_rgb, (phys_px, phys_px))
    pad = (tile_px - phys_px % tile_px) % tile_px
    if pad:
        phys = cv2.copyMakeBorder(phys, 0, pad, 0, pad, cv2.BORDER_REFLECT)
    tiles: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    h, w = phys.shape[:2]
    for y in range(0, h, tile_px):
        for x in range(0, w, tile_px):
            tiles.append(phys[y : y + tile_px, x : x + tile_px])
            coords.append((y, x))
    return tiles, coords, phys_px


def stitch_predictions(
    model_name: str,
    tile_preds: Sequence[np.ndarray],
    coords: Sequence[tuple[int, int]],
    tile_px: int,
    phys_px: int,
) -> np.ndarray:
    """Stitch CHW tiles. HistoPlexer/HEMIT whole-window: a single tile is already complete."""
    name = str(model_name).strip().lower()
    if name != "mipheivit" and len(tile_preds) == 1:
        return np.asarray(tile_preds[0], dtype=np.float32)
    n_ch, height, width = tile_preds[0].shape
    if name != "mipheivit" and (height != tile_px or width != tile_px):
        raise RuntimeError(f"tile pred {tile_preds[0].shape} != tile_px={tile_px}")
    padded = max(c[0] for c in coords) + tile_px
    canvas = np.zeros((n_ch, padded, padded), dtype=np.float32)
    for pred, (y, x) in zip(tile_preds, coords):
        canvas[:, y : y + tile_px, x : x + tile_px] = pred
    return canvas[:, :phys_px, :phys_px]


def prepare_he_feed(
    model_name: str,
    he_rgb: np.ndarray,
    patch_um: float,
    *,
    model_mpp: float,
    tile_px: int,
    feed: str,
) -> tuple[list[np.ndarray], list[tuple[int, int]], int, int]:
    """Return ``(tiles, coords, tile_px, phys_px)`` for one Cell-niches window."""
    name = str(model_name).strip().lower()
    if name == "mipheivit":
        tiles, coords, phys_px = he_mosaic_at_model_mpp(
            he_rgb, patch_um=patch_um, model_mpp=model_mpp, tile_px=tile_px
        )
        return tiles, coords, tile_px, phys_px
    phys, phys_px = he_at_model_mpp(he_rgb, patch_um, model_mpp)
    expect_px = expected_he_px(patch_um, model_mpp)
    if phys_px != expect_px:
        raise RuntimeError(f"{name} resampled {phys_px} px, expected {expect_px}")
    if feed == "whole":
        return [phys], [(0, 0)], phys_px, phys_px
    tiles, coords, phys_px = mosaic_tiles(phys, tile_px)
    return tiles, coords, tile_px, phys_px
