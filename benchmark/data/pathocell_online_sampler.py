"""PathoCell / Lizard online HDF sampler (PFMB tiles, optional class-balanced draws)."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.hooknet_tls_wsd import patch_to_tensor
from data.pathocell_hdf import (
    PathoCellTile,
    extract_pathocell_tile_triple_at_np,
    is_pfmb_tiling,
    load_pathocell_volume,
    pathocell_hdf_metadata,
    pathocell_tile_geometry,
    plan_pathocell_tiles,
    resolve_pathocell_tiling,
    PATHOCELL_NATIVE_MPP,
)


class _PatchIndexDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return int(idx)


class DistributedWeightedSampler(Sampler[int]):
    """One global multinomial draw, then shard ``rank::num_replicas`` (DDP)."""

    def __init__(
        self,
        dataset: Dataset,
        weights: np.ndarray,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape[0] != len(dataset):
            raise ValueError("weights must match dataset length")
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        weights = torch.as_tensor(self.weights, dtype=torch.double)
        global_draws = torch.multinomial(
            weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(global_draws[self.rank :: self.num_replicas])


@dataclass(frozen=True)
class _PatchRef:
    hdf_path: Path
    tile: PathoCellTile


class PathoCellOnlineSampler:
    """Random (optionally class-balanced) PFMB tiles from PathoCell/Lizard HDFs."""

    def __init__(
        self,
        exp: Any,
        samples: Sequence[Any],
        transform: Any = None,
        batch_size: int = 16,
        seed: int = 42,
        volume_cache_size: int = 8,
        patch_workers: int | None = None,
        uniform_class_sampling: bool = False,
        num_classes: int | None = None,
        exclude_classes: Sequence[int] | None = None,
        sample_excluded_classes: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: Any = None,
    ) -> None:
        del device
        self.exp = exp
        self.transform = transform
        self.batch_size = max(1, int(batch_size))
        self.volume_cache_size = max(1, int(volume_cache_size))
        self.uniform_class_sampling = bool(uniform_class_sampling)
        self.sample_excluded_classes = bool(sample_excluded_classes)
        self.exclude_classes = {int(c) for c in (exclude_classes or [0])}
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.tiling = resolve_pathocell_tiling(exp)
        self.renumber = is_pfmb_tiling(self.tiling)
        native_mpp = getattr(getattr(exp, "dataset_config", None), "native_mpp", None)
        if native_mpp is None:
            native_mpp = PATHOCELL_NATIVE_MPP
        target_mpp = exp.target_mpp if getattr(exp, "use_mpp", True) else None
        self.read_size, self.output_size = pathocell_tile_geometry(
            int(exp.target_patch_size),
            target_mpp,
            native_mpp=float(native_mpp),
        )
        self.patch_refs: list[_PatchRef] = []
        present_per_tile: list[set[int]] = []
        class_tile_count: dict[int, int] = {}
        for sample in samples:
            path = getattr(sample, "image_path", None)
            if path is None:
                continue
            hdf = Path(path)
            if not hdf.is_file():
                continue
            height, width, has_semantic, _has_instance = pathocell_hdf_metadata(hdf)
            if not has_semantic:
                continue
            _, tiles = plan_pathocell_tiles(height, width, self.read_size, protocol=self.tiling)
            if self.uniform_class_sampling:
                image_hwc, semantic_hw, instance_hw = load_pathocell_volume(hdf)
                del image_hwc, instance_hw
                if semantic_hw is None:
                    continue
            else:
                semantic_hw = None
            for tile in tiles:
                if semantic_hw is not None:
                    y0, y1, x0, x1 = tile
                    crop = semantic_hw[y0:y1, x0:x1]
                    classes = {int(v) for v in np.unique(crop) if int(v) >= 0}
                    if not self.sample_excluded_classes:
                        classes -= self.exclude_classes
                    if self.num_classes is not None:
                        classes = {c for c in classes if c < self.num_classes}
                    present_per_tile.append(classes)
                    for cls in classes:
                        class_tile_count[cls] = class_tile_count.get(cls, 0) + 1
                self.patch_refs.append(_PatchRef(hdf_path=hdf, tile=tile))
        if not self.patch_refs:
            raise FileNotFoundError("PathoCellOnlineSampler: no readable HDF tiles")

        n = len(self.patch_refs)
        if self.uniform_class_sampling:
            weights = np.zeros(n, dtype=np.float64)
            for i, classes in enumerate(present_per_tile):
                w = 0.0
                for cls in classes:
                    freq = max(1, class_tile_count.get(cls, 1))
                    w += 1.0 / float(freq)
                weights[i] = w if w > 0 else 1e-6
            self._patch_weights = weights
        else:
            self._patch_weights = None

        self._rng = np.random.default_rng(int(seed))
        self._sequential = False
        self._order = np.arange(n, dtype=np.int64)
        self._cursor = 0
        workers = patch_workers if patch_workers is not None else int(getattr(exp, "iterator_cpus", 1) or 1)
        self.patch_workers = max(1, int(workers))
        self._pool = (
            ThreadPoolExecutor(max_workers=self.patch_workers, thread_name_prefix="pc_patch")
            if self.patch_workers > 1
            else None
        )
        self._volume_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = (
            OrderedDict()
        )
        self._cache_lock = threading.Lock()
        self._weight_sampler: DistributedWeightedSampler | None = None
        self._weight_iter = None
        if self.uniform_class_sampling and self.world_size > 1 and self._patch_weights is not None:
            self._weight_sampler = DistributedWeightedSampler(
                _PatchIndexDataset(n),
                self._patch_weights,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=int(seed),
            )

    def __len__(self) -> int:
        return len(self.patch_refs)

    def begin_full_pass(self) -> None:
        self._sequential = True
        self._cursor = 0
        self._order = np.arange(len(self.patch_refs), dtype=np.int64)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._volume_cache.clear()

    def next_batch(self) -> dict[str, Any]:
        if self._sequential:
            indices = self._take_sequential()
        else:
            indices = [self._sample_index() for _ in range(self.batch_size)]
        return self._collate(indices)

    def _take_sequential(self) -> list[int]:
        n = int(self._order.shape[0])
        if n <= 0:
            return []
        start = self._cursor
        if start >= n:
            self._cursor = 0
            start = 0
        end = min(start + self.batch_size, n)
        sl = self._order[start:end]
        self._cursor = end
        return [int(i) for i in sl]

    def _sample_index(self) -> int:
        n = len(self.patch_refs)
        if self._weight_sampler is not None:
            if self._weight_iter is None:
                self._weight_sampler.set_epoch(int(self._rng.integers(0, 10_000_000)))
                self._weight_iter = iter(self._weight_sampler)
            try:
                return int(next(self._weight_iter))
            except StopIteration:
                self._weight_sampler.set_epoch(self._weight_sampler.epoch + 1)
                self._weight_iter = iter(self._weight_sampler)
                return int(next(self._weight_iter))
        if self.uniform_class_sampling and self._patch_weights is not None:
            p = self._patch_weights / self._patch_weights.sum()
            return int(self._rng.choice(n, p=p))
        return int(self._rng.integers(0, n))

    def _cached_volume(self, hdf_path: Path):
        key = str(hdf_path)
        with self._cache_lock:
            if key in self._volume_cache:
                self._volume_cache.move_to_end(key)
                return self._volume_cache[key]
        volume = load_pathocell_volume(hdf_path)
        with self._cache_lock:
            self._volume_cache[key] = volume
            self._volume_cache.move_to_end(key)
            while len(self._volume_cache) > self.volume_cache_size:
                self._volume_cache.popitem(last=False)
        return volume

    def _load_index(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.patch_refs[int(idx)]
        image_hwc, semantic_hw, instance_hw = self._cached_volume(ref.hdf_path)
        image, semantic, instance = extract_pathocell_tile_triple_at_np(
            image_hwc,
            semantic_hw,
            instance_hw,
            ref.tile,
            self.read_size,
            self.output_size,
            protocol=self.tiling,
            renumber_instances=self.renumber,
        )
        if semantic is None:
            semantic = np.zeros((self.output_size, self.output_size), dtype=np.int64)
        if instance is None:
            instance = np.zeros_like(semantic)
        return image, semantic.astype(np.int64, copy=False), instance.astype(np.int64, copy=False)

    def _collate(self, indices: Sequence[int]) -> dict[str, Any]:
        if self._pool is not None and len(indices) > 1:
            loaded = list(self._pool.map(self._load_index, indices))
        else:
            loaded = [self._load_index(i) for i in indices]
        images = []
        masks = []
        instances = []
        for rgb, semantic, instance in loaded:
            if self.transform is None:
                tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
            else:
                tensor = patch_to_tensor(rgb, self.transform)
            images.append(tensor)
            masks.append(semantic)
            instances.append(instance)
        return {
            "image": torch.stack(images, dim=0),
            "semantic_mask": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.long),
            "instance_mask": torch.as_tensor(np.stack(instances, axis=0), dtype=torch.long),
        }
