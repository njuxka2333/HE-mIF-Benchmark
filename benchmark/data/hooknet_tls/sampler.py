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

from .planning import (
    IGNORE_LABEL,
    PatchPlanRecord,
    load_selected_tiles_csv,
)
from data.image_transforms import patch_to_tensor
from data.tiling import TileBounds, center_pad_2d, center_pad_hwc, pfmb_grid
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
    tile: TileBounds,
    read_size: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop one PFMB tile, center-pad to ``read_size``, resize to model input."""
    y0, y1, x0, x1 = tile
    image_crop = np.asarray(rgb[y0:y1, x0:x1])
    mask_crop = np.asarray(mask[y0:y1, x0:x1])
    image_crop = center_pad_hwc(image_crop, read_size, read_size, fill_value=0)
    mask_crop = center_pad_2d(mask_crop, read_size, read_size, fill_value=IGNORE_LABEL)
    if image_crop.shape[0] != output_size or image_crop.shape[1] != output_size:
        image_crop = resize_image_np(image_crop, (output_size, output_size))
        mask_crop = resize_mask(mask_crop, (output_size, output_size))
    return image_crop, mask_crop.astype(np.int64, copy=False)


@dataclass(frozen=True)
class _TileRef:
    record: PatchPlanRecord
    tile: TileBounds
    image_path: Path
    mask_path: Path


class FixedGridTileSampler:
    """Finite-index PFMB tiles over HookNet-TLS canonical PNGs."""

    def __init__(
        self,
        experiment: Any,
        sampling: Any,
        *,
        image_transform: Any,
        split: str = "train",
        seed_offset: int = 0,
        patch_workers: int | None = None,
        class_balanced: bool = False,
    ) -> None:
        if getattr(sampling, "fixed_grid_root", None) is None:
            raise ValueError("FixedGridTileSampler requires fixed_grid_sampling.fixed_grid_root")

        self.image_transform = image_transform
        self.split = str(split)
        self.class_balanced = bool(class_balanced)
        self.min_selected_class_pixels = max(
            1, int(getattr(sampling, "fixed_grid_min_selected_class_pixels", 1) or 1)
        )
        self.max_resample_attempts = max(
            1, int(getattr(sampling, "fixed_grid_max_resample_attempts", 32) or 32)
        )
        self.batch_size = max(1, int(getattr(sampling, "sampler_batch_size", 1) or 1))

        fixed_grid_root = Path(sampling.fixed_grid_root).resolve()
        if not fixed_grid_root.is_dir():
            raise ValueError(f"fixed_grid_root is not a directory: {fixed_grid_root}")
        self.fixed_grid_root = fixed_grid_root
        summary_path = fixed_grid_root / "manifest_summary.json"
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
        self.canonical_grid_size = int(summary["output_size"])
        target_mpp = (
            experiment.target_mpp
            if experiment.target_mpp is not None
            else self.reference_mpp
        )
        self.target_mpp = float(target_mpp)
        self.target_patch_size = int(experiment.target_patch_size)
        self.tile_read_size = tls_source_read_size(
            self.target_patch_size, self.target_mpp, reference_mpp=self.reference_mpp
        )
        if self.tile_read_size > self.canonical_grid_size:
            raise ValueError(
                f"Target geometry {self.target_patch_size}px @ {self.target_mpp} mpp requires "
                f"{self.tile_read_size}px from canonical {self.canonical_grid_size}px @ "
                f"{self.reference_mpp} mpp; cannot derive this view."
            )
        self.tile_bounds: list[TileBounds] = pfmb_grid(
            self.canonical_grid_size, self.canonical_grid_size, self.tile_read_size
        )
        if not self.tile_bounds:
            raise ValueError("PFMB tiling produced no TLS tiles")

        patch_catalog_path = fixed_grid_root / "patches.csv"
        if not patch_catalog_path.is_file():
            raise FileNotFoundError(
                f"TLS patch catalog missing: {patch_catalog_path}. "
                "Expected patches.csv (same crop columns as Cell-niches)."
            )
        patch_records = [
            record
            for record in load_selected_tiles_csv(patch_catalog_path)
            if record.split == self.split
        ]
        self._tile_refs: list[_TileRef] = []
        self._label_pools: dict[str, list[int]] = {name: [] for name in OFFICIAL_TRAIN_LABELS}
        for patch_record in patch_records:
            image_relative_path = patch_record.image_relpath
            mask_relative_path = patch_record.mask_relpath or image_relative_path
            if not image_relative_path:
                continue
            image_path = (
                fixed_grid_root / image_relative_path
                if "/" in image_relative_path.replace("\\", "/")
                else fixed_grid_root / self.split / "images" / Path(image_relative_path).name
            )
            mask_path = (
                fixed_grid_root / mask_relative_path
                if "/" in mask_relative_path.replace("\\", "/")
                else fixed_grid_root / self.split / "masks" / Path(mask_relative_path).name
            )
            first_tile_index = len(self._tile_refs)
            class_counts = {
                "tls": patch_record.count_tls,
                "gc": patch_record.count_gc,
                "rest": patch_record.count_rest,
            }
            for tile_bounds in self.tile_bounds:
                self._tile_refs.append(
                    _TileRef(
                        record=patch_record,
                        tile=tile_bounds,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
            for label, count in class_counts.items():
                if count > 0:
                    # Parent-level pool: tile membership is checked after PFMB extract.
                    self._label_pools[label].extend(
                        range(first_tile_index, first_tile_index + len(self.tile_bounds))
                    )

        if not self._tile_refs:
            raise FileNotFoundError(
                f"No readable fixed-grid patches under {fixed_grid_root / self.split}"
            )

        seed = int(getattr(sampling, "seed", 42) or 42) + int(seed_offset)
        self._rng = np.random.default_rng(seed)
        self._sequential = False
        self._order = np.arange(len(self._tile_refs), dtype=np.int64)
        self._cursor = 0
        workers = (
            patch_workers
            if patch_workers is not None
            else int(getattr(experiment, "iterator_cpus", 1) or 1)
        )
        self.patch_workers = max(1, int(workers))
        self._pool = (
            ThreadPoolExecutor(max_workers=self.patch_workers, thread_name_prefix="tls_fg_patch")
            if self.patch_workers > 1
            else None
        )

    def __len__(self) -> int:
        return len(self._tile_refs)

    def begin_full_pass(self) -> None:
        """Cover every PFMB tile once (full_grid eval)."""
        self._sequential = True
        self.reset_epoch()

    def reset_epoch(self, epoch: int | None = None) -> None:
        del epoch
        n = len(self._tile_refs)
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
            n = len(self._tile_refs)
            indices = [int(self._rng.integers(0, n)) for _ in range(self.batch_size)]
        return self._make_batch(indices)

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
            _rgb, mask = self._load_tile_pixels(self._tile_refs[idx])
            class_id = LABEL_TO_CLASS_ID[label]
            if int(np.count_nonzero(mask == class_id)) >= self.min_selected_class_pixels:
                return idx
        return int(self._rng.integers(0, len(self._tile_refs)))

    def _load_tile_pixels(self, ref: _TileRef) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.asarray(Image.open(ref.image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(ref.mask_path), dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if rgb.shape[:2] != (self.canonical_grid_size, self.canonical_grid_size):
            raise ValueError(
                f"Canonical patch shape mismatch for {ref.image_path}: "
                f"rgb={rgb.shape[:2]}, expected {self.canonical_grid_size}²"
            )
        if tuple(mask.shape[:2]) != (self.canonical_grid_size, self.canonical_grid_size):
            raise ValueError(
                f"Canonical mask shape mismatch for {ref.mask_path}: "
                f"mask={mask.shape[:2]}, expected {self.canonical_grid_size}²"
            )
        return extract_tls_pfmb_tile(
            rgb, mask, ref.tile, self.tile_read_size, self.target_patch_size
        )

    def _load_tile_record(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        ref = self._tile_refs[int(idx)]
        rgb, mask = self._load_tile_pixels(ref)
        return rgb, mask, ref.record.slide_id

    def _make_batch(self, indices: Sequence[int]) -> dict[str, Any]:
        if self._pool is not None and len(indices) > 1:
            loaded = list(self._pool.map(self._load_tile_record, indices))
        else:
            loaded = [self._load_tile_record(i) for i in indices]
        images = []
        masks = []
        slide_ids = []
        for rgb, mask, slide_id in loaded:
            if self.image_transform is None:
                tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
            else:
                tensor = patch_to_tensor(rgb, self.image_transform)
            images.append(tensor)
            masks.append(mask)
            slide_ids.append(slide_id)
        return {
            "image": torch.stack(images, dim=0),
            "semantic_mask": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.long),
            "slide_id": slide_ids,
        }
