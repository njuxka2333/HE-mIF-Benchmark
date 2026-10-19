"""Pyramid TIFF IO for Cell-niches libvips tiles (no OpenSlide)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def index_wsis(wsi_dir: Path) -> dict[str, Path]:
    """Map trailing UUID in ``{prefix}_{uuid}.tiff`` to the file path."""
    out: dict[str, Path] = {}
    for path in sorted(Path(wsi_dir).glob("*.tiff")):
        uid = path.stem.rsplit("_", 1)[-1]
        out[uid] = path
    return out


def as_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            vmax = float(np.max(rgb)) or 1.0
            rgb = np.clip(rgb.astype(np.float32) * (255.0 / vmax), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)
    raise ValueError(f"unsupported RGB shape {arr.shape}")


def as_gray(arr: np.ndarray) -> np.ndarray:
    """Cell-niches mIF TIFFs are often (H, W, 3) with R=G=B copies."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., 0]
    raise ValueError(f"unsupported gray shape {arr.shape}")


class PyramidTIFF:
    """Random-access crops from a multi-page libvips pyramid TIFF."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._tf = tifffile.TiffFile(self.path)
        if not self._tf.series:
            raise ValueError(f"no TIFF series in {self.path}")
        self.series = self._tf.series[0]
        self.n_levels = len(self.series.levels)
        shape = self.series.levels[0].shape
        self.height = int(shape[0])
        self.width = int(shape[1])
        self.dtype = self.series.levels[0].dtype
        self._level_cache: dict[int, np.ndarray] = {}

    def level_shape(self, level: int) -> tuple[int, int]:
        h, w = self.series.levels[level].shape[:2]
        return int(h), int(w)

    def downsample_of(self, level: int) -> tuple[float, float]:
        h, w = self.level_shape(level)
        return self.width / float(w), self.height / float(h)

    def choose_level(self, target_max_side: int = 1024) -> int:
        best_i = 0
        best_d = float("inf")
        for i in range(self.n_levels):
            h, w = self.level_shape(i)
            d = abs(max(h, w) - target_max_side)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def read_level(self, level: int) -> np.ndarray:
        if level not in self._level_cache:
            self._level_cache[level] = self.series.levels[level].asarray()
        return self._level_cache[level]

    def crop_level0(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, x0 + int(w))
        y1 = min(self.height, y0 + int(h))
        return self.series.levels[0].asarray()[y0:y1, x0:x1]

    def close(self) -> None:
        self._level_cache.clear()
        self._tf.close()

    def __enter__(self) -> "PyramidTIFF":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
