"""TLS PNG sampler: PFMB tiles over the canonical 1024 px @ 0.25 mpp cell.

Each stored patch is the same 256 µm physical cell. A model-native window on
that PNG is ``round(target_patch_size * target_mpp / 0.25)`` px. Those windows
cover the 1024 PNG with PathoCell PFMB linspace tiles (no overlap). Short edge
tiles are center-padded, then resized to ``target_patch_size``.

Center-cropping a single native FOV out of the 1024 cell is not used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from data.hooknet_tls import (
    IGNORE_LABEL,
    PatchPlanRecord,
    load_selected_tiles_csv,
)
from data.hooknet_tls_wsd import patch_to_tensor
from data.pathocell_hdf import (
    PathoCellTile,
    center_pad_2d,
    center_pad_hwc,
    pfmb_linspace_tiles,
)
from utils.resize_ops import resize_image_np, resize_mask

LABEL_TO_CLASS_ID = {"tls": 1, "gc": 2, "rest": 3}
OFFICIAL_TRAIN_LABELS = ("tls", "gc", "rest")
CANONICAL_PROTOCOL = "canonical_physical_grid_v1"


def tls_source_read_size(
    target_patch_size: int,
    target_mpp: float,
    *,
    reference_mpp: float,
) -> int:
    """Window size on the canonical PNG that equals ``target_patch_size`` @ ``target_mpp``."""
    if target_patch_size <= 0 or target_mpp <= 0 or reference_mpp <= 0:
        raise ValueError(
            f"invalid TLS geometry patch={target_patch_size} mpp={target_mpp} "
            f"reference_mpp={reference_mpp}"
        )
    return max(1, int(round(float(target_patch_size) * float(target_mpp) / float(reference_mpp))))


def extract_tls_pfmb_tile(
    rgb: np.ndarray,
    mask: np.ndarray,
    tile: PathoCellTile,
    read_size: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop one PFMB tile, center-pad to ``read_size``, resize to model input."""
    y0, y1, x0, x1 = tile
    image_crop = np.asarray(rgb[y0:y1, x0:x1])
    mask_crop = np.asarray(mask[y0:y1, x0:x1])
    image_crop = center_pad_hwc(image_crop, read_size, read_size, fill=0)
    mask_crop = center_pad_2d(mask_crop, read_size, read_size, fill=IGNORE_LABEL)
    if image_crop.shape[0] != output_size or image_crop.shape[1] != output_size:
        image_crop = resize_image_np(image_crop, (output_size, output_size))
        mask_crop = resize_mask(mask_crop, (output_size, output_size))
    return image_crop, mask_crop.astype(np.int64, copy=False)


@dataclass(frozen=True)
class _TileRef:
    record: PatchPlanRecord
    tile: PathoCellTile
    image_path: Path
    mask_path: Path


class FixedGridIndexSampler:
    """Finite-index PFMB tiles over HookNet-TLS canonical PNGs."""

    def __init__(
        self,
        exp: Any,
        samples: Any,
        official: Any,
        transform: Any,
        *,
        split: str = "train",
        device: Any = None,
        backend: str = "openslide",
        seed_offset: int = 0,
        patch_workers: int | None = None,
        uniform_class_sampling: bool = False,
        patch_weight_mode: str | None = None,
        class_balanced: bool | None = None,
    ) -> None:
        del samples, backend, patch_weight_mode, device
        if official is None or getattr(official, "fixed_grid_root", None) is None:
            raise ValueError("FixedGridIndexSampler requires official_sampling.fixed_grid_root")

        self.exp = exp
        self.official = official
        self.transform = transform
        self.split = str(split)
        self.class_balanced = bool(
            uniform_class_sampling if class_balanced is None else class_balanced
        )
        self.min_selected_class_pixels = max(
            1, int(getattr(official, "fixed_grid_min_selected_class_pixels", 1) or 1)
        )
        self.max_resample_attempts = max(
            1, int(getattr(official, "fixed_grid_max_resample_attempts", 32) or 32)
        )
        self.batch_size = max(1, int(getattr(official, "sampler_batch_size", 1) or 1))

        root = Path(official.fixed_grid_root).resolve()
        if not root.is_dir():
            raise ValueError(f"fixed_grid_root is not a directory: {root}")
        self.root = root
        summary_path = root / "manifest_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"fixed-grid root summary not found: {summary_path}")
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("geometry_protocol") != CANONICAL_PROTOCOL:
            raise ValueError(
                "Fixed-grid training requires canonical_physical_grid_v1 data. "
                "Rebuild the dataset with rebuild_hooknet_tls.py."
            )
        self.reference_mpp = float(summary["reference_mpp"])
        self.canonical_size = int(summary["output_size"])
        target_mpp = exp.target_mpp if exp.target_mpp is not None else self.reference_mpp
        self.target_mpp = float(target_mpp)
        self.target_size = int(exp.target_patch_size)
        self.read_size = tls_source_read_size(
            self.target_size, self.target_mpp, reference_mpp=self.reference_mpp
        )
        if self.read_size > self.canonical_size:
            raise ValueError(
                f"Target geometry {self.target_size}px @ {self.target_mpp} mpp requires "
                f"{self.read_size}px from canonical {self.canonical_size}px @ "
                f"{self.reference_mpp} mpp; cannot derive this view."
            )
        self.tiles: list[PathoCellTile] = pfmb_linspace_tiles(
            self.canonical_size, self.canonical_size, self.read_size
        )
        if not self.tiles:
            raise ValueError("PFMB tiling produced no TLS tiles")

        catalog = root / "patches.csv"
        if not catalog.is_file():
            raise FileNotFoundError(
                f"TLS patch catalog missing: {catalog}. "
                "Expected patches.csv (same crop columns as Cell-niches)."
            )
        records = [rec for rec in load_selected_tiles_csv(catalog) if rec.split == self.split]
        self._index: list[_TileRef] = []
        self._label_pools: dict[str, list[int]] = {name: [] for name in OFFICIAL_TRAIN_LABELS}
        for record in records:
            he_rel = record.image_relpath
            mask_rel = record.mask_relpath or he_rel
            if not he_rel:
                continue
            image_path = root / he_rel if "/" in he_rel.replace("\\", "/") else root / self.split / "images" / Path(he_rel).name
            mask_path = root / mask_rel if "/" in mask_rel.replace("\\", "/") else root / self.split / "masks" / Path(mask_rel).name
            parent_idx = len(self._index)
            counts = {"tls": record.count_tls, "gc": record.count_gc, "rest": record.count_rest}
            for tile in self.tiles:
                self._index.append(
                    _TileRef(
                        record=record,
                        tile=tile,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
            for label, count in counts.items():
                if count > 0:
                    # Parent-level pool: tile membership is checked after PFMB extract.
                    self._label_pools[label].extend(range(parent_idx, parent_idx + len(self.tiles)))

        if not self._index:
            raise FileNotFoundError(f"No readable fixed-grid patches under {root / self.split}")

        seed = int(getattr(official, "seed", 42) or 42) + int(seed_offset)
        self._rng = np.random.default_rng(seed)
        self._sequential = False
        self._order = np.arange(len(self._index), dtype=np.int64)
        self._cursor = 0
        workers = patch_workers if patch_workers is not None else int(getattr(exp, "iterator_cpus", 1) or 1)
        self.patch_workers = max(1, int(workers))
        self._pool = (
            ThreadPoolExecutor(max_workers=self.patch_workers, thread_name_prefix="tls_fg_patch")
            if self.patch_workers > 1
            else None
        )

    def __len__(self) -> int:
        return len(self._index)

    def begin_full_pass(self) -> None:
        """Cover every PFMB tile once (full_grid eval)."""
        self._sequential = True
        self.reset_epoch()

    def reset_epoch(self, epoch: int | None = None) -> None:
        del epoch
        n = len(self._index)
        if self._sequential:
            self._order = np.arange(n, dtype=np.int64)
        else:
            self._order = self._rng.permutation(n) if n else np.zeros(0, dtype=np.int64)
        self._cursor = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def next_batch(self) -> dict[str, Any]:
        if self._sequential:
            indices = self._take_sequential()
        elif self.class_balanced:
            indices = [self._sample_class_balanced_index() for _ in range(self.batch_size)]
        else:
            n = len(self._index)
            indices = [int(self._rng.integers(0, n)) for _ in range(self.batch_size)]
        return self._collate(indices)

    def _take_sequential(self) -> list[int]:
        n = int(self._order.shape[0])
        if n <= 0:
            return []
        start = self._cursor
        end = min(start + self.batch_size, n)
        if start >= n:
            self._cursor = 0
            start, end = 0, min(self.batch_size, n)
        sl = self._order[start:end]
        self._cursor = end
        return [int(i) for i in sl]

    def _sample_class_balanced_index(self) -> int:
        labels = list(OFFICIAL_TRAIN_LABELS)
        for _ in range(self.max_resample_attempts):
            label = str(labels[int(self._rng.integers(0, len(labels)))])
            pool = self._label_pools.get(label) or []
            if not pool:
                continue
            idx = int(pool[int(self._rng.integers(0, len(pool)))])
            _rgb, mask = self._load_tile(self._index[idx])
            class_id = LABEL_TO_CLASS_ID[label]
            if int(np.count_nonzero(mask == class_id)) >= self.min_selected_class_pixels:
                return idx
        return int(self._rng.integers(0, len(self._index)))

    def _load_tile(self, ref: _TileRef) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.asarray(Image.open(ref.image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(ref.mask_path), dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if rgb.shape[:2] != (self.canonical_size, self.canonical_size):
            raise ValueError(
                f"Canonical patch shape mismatch for {ref.image_path}: "
                f"rgb={rgb.shape[:2]}, expected {self.canonical_size}²"
            )
        if tuple(mask.shape[:2]) != (self.canonical_size, self.canonical_size):
            raise ValueError(
                f"Canonical mask shape mismatch for {ref.mask_path}: "
                f"mask={mask.shape[:2]}, expected {self.canonical_size}²"
            )
        return extract_tls_pfmb_tile(
            rgb, mask, ref.tile, self.read_size, self.target_size
        )

    def _load_index(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        ref = self._index[int(idx)]
        rgb, mask = self._load_tile(ref)
        return rgb, mask, ref.record.slide_id

    def _collate(self, indices: Sequence[int]) -> dict[str, Any]:
        if self._pool is not None and len(indices) > 1:
            loaded = list(self._pool.map(self._load_index, indices))
        else:
            loaded = [self._load_index(i) for i in indices]
        images = []
        masks = []
        slide_ids = []
        for rgb, mask, slide_id in loaded:
            if self.transform is None:
                tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
            else:
                tensor = patch_to_tensor(rgb, self.transform)
            images.append(tensor)
            masks.append(mask)
            slide_ids.append(slide_id)
        return {
            "image": torch.stack(images, dim=0),
            "semantic_mask": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.long),
            "slide_id": slide_ids,
        }


# Probe scripts historically imported this name. Same PFMB sampler; no center crop.
FixedGridClassWeightedSampler = FixedGridIndexSampler
