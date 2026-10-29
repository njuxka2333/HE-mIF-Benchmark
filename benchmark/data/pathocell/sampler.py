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

from data.image_transforms import patch_to_tensor
from .hdf import (
    load_pathocell_volume,
    pathocell_hdf_metadata,
    PATHOCELL_NATIVE_MPP,
)
from data.tiling import (
    PFMB_TILE_PROTOCOL,
    TileBounds,
    extract_tile_arrays,
    is_pfmb_tiling,
    plan_tiles,
    tile_geometry,
)


class _TileIndexDataset(Dataset):
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
class _TileRef:
    hdf_path: Path
    tile: TileBounds


class HDFTileBatchSampler:
    """Yield random or class-balanced tiles from annotated HDF volumes."""

    def __init__(
        self,
        experiment_config: Any,
        samples: Sequence[Any],
        image_transform: Any = None,
        batch_size: int = 16,
        seed: int = 42,
        max_cached_volumes: int = 8,
        patch_workers: int | None = None,
        uniform_class_sampling: bool = False,
        num_classes: int | None = None,
        exclude_classes: Sequence[int] | None = None,
        sample_excluded_classes: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.image_transform = image_transform
        self.batch_size = max(1, int(batch_size))
        self.max_cached_volumes = max(1, int(max_cached_volumes))
        self.uniform_class_sampling = bool(uniform_class_sampling)
        self.sample_excluded_classes = bool(sample_excluded_classes)
        self.exclude_classes = {int(c) for c in (exclude_classes or [0])}
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.tile_protocol = str(
            getattr(experiment_config, "pathocell_tiling", PFMB_TILE_PROTOCOL)
            or PFMB_TILE_PROTOCOL
        ).strip().lower()
        self.renumber_instances = is_pfmb_tiling(self.tile_protocol)
        native_mpp = getattr(
            getattr(experiment_config, "dataset_config", None), "native_mpp", None
        )
        if native_mpp is None:
            native_mpp = PATHOCELL_NATIVE_MPP
        target_mpp = (
            experiment_config.target_mpp
            if getattr(experiment_config, "use_mpp", True)
            else None
        )
        self.tile_read_size, self.output_patch_size = tile_geometry(
            int(experiment_config.target_patch_size),
            target_mpp,
            native_mpp=float(native_mpp),
        )
        self.tile_refs: list[_TileRef] = []
        classes_by_tile: list[set[int]] = []
        tile_frequency_by_class: dict[int, int] = {}
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
            _, tiles = plan_tiles(height, width, self.tile_read_size, protocol=self.tile_protocol)
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
                    classes = {int(value) for value in np.unique(crop) if int(value) >= 0}
                    if not self.sample_excluded_classes:
                        classes -= self.exclude_classes
                    if self.num_classes is not None:
                        classes = {c for c in classes if c < self.num_classes}
                    classes_by_tile.append(classes)
                    for cls in classes:
                        tile_frequency_by_class[cls] = tile_frequency_by_class.get(cls, 0) + 1
                self.tile_refs.append(_TileRef(hdf_path=hdf, tile=tile))
        if not self.tile_refs:
            raise FileNotFoundError("HDFTileBatchSampler: no readable annotated HDF tiles")

        n_tiles = len(self.tile_refs)
        if self.uniform_class_sampling:
            tile_weights = np.zeros(n_tiles, dtype=np.float64)
            for index, classes in enumerate(classes_by_tile):
                w = 0.0
                for cls in classes:
                    freq = max(1, tile_frequency_by_class.get(cls, 1))
                    w += 1.0 / float(freq)
                tile_weights[index] = w if w > 0 else 1e-6
            self._tile_weights = tile_weights
        else:
            self._tile_weights = None

        self._random = np.random.default_rng(int(seed))
        self._sequential_pass = False
        self._tile_order = np.arange(n_tiles, dtype=np.int64)
        self._tile_cursor = 0
        workers = (
            patch_workers
            if patch_workers is not None
            else int(getattr(experiment_config, "iterator_cpus", 1) or 1)
        )
        self.patch_workers = max(1, int(workers))
        self._pool = (
            ThreadPoolExecutor(max_workers=self.patch_workers, thread_name_prefix="pc_patch")
            if self.patch_workers > 1
            else None
        )
        self._cached_volumes: OrderedDict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = (
            OrderedDict()
        )
        self._volume_cache_lock = threading.Lock()
        self._distributed_weight_sampler: DistributedWeightedSampler | None = None
        self._weight_iterator = None
        if self.uniform_class_sampling and self.world_size > 1 and self._tile_weights is not None:
            self._distributed_weight_sampler = DistributedWeightedSampler(
                _TileIndexDataset(n_tiles),
                self._tile_weights,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=int(seed),
            )

    def __len__(self) -> int:
        return len(self.tile_refs)

    def begin_full_pass(self) -> None:
        self._sequential_pass = True
        self._tile_cursor = 0
        self._tile_order = np.arange(len(self.tile_refs), dtype=np.int64)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._cached_volumes.clear()

    def next_batch(self) -> dict[str, Any]:
        if self._sequential_pass:
            indices = self._take_sequential()
        else:
            indices = [self._sample_tile_index() for _ in range(self.batch_size)]
        return self._make_batch(indices)

    def _take_sequential(self) -> list[int]:
        n = int(self._tile_order.shape[0])
        if n <= 0:
            return []
        start = self._tile_cursor
        if start >= n:
            self._tile_cursor = 0
            start = 0
        end = min(start + self.batch_size, n)
        sl = self._tile_order[start:end]
        self._tile_cursor = end
        return [int(i) for i in sl]

    def _sample_tile_index(self) -> int:
        n_tiles = len(self.tile_refs)
        if self._distributed_weight_sampler is not None:
            if self._weight_iterator is None:
                self._distributed_weight_sampler.set_epoch(
                    int(self._random.integers(0, 10_000_000))
                )
                self._weight_iterator = iter(self._distributed_weight_sampler)
            try:
                return int(next(self._weight_iterator))
            except StopIteration:
                self._distributed_weight_sampler.set_epoch(
                    self._distributed_weight_sampler.epoch + 1
                )
                self._weight_iterator = iter(self._distributed_weight_sampler)
                return int(next(self._weight_iterator))
        if self.uniform_class_sampling and self._tile_weights is not None:
            probabilities = self._tile_weights / self._tile_weights.sum()
            return int(self._random.choice(n_tiles, p=probabilities))
        return int(self._random.integers(0, n_tiles))

    def _cached_volume(self, hdf_path: Path):
        key = str(hdf_path)
        with self._volume_cache_lock:
            if key in self._cached_volumes:
                self._cached_volumes.move_to_end(key)
                return self._cached_volumes[key]
        volume = load_pathocell_volume(hdf_path)
        with self._volume_cache_lock:
            self._cached_volumes[key] = volume
            self._cached_volumes.move_to_end(key)
            while len(self._cached_volumes) > self.max_cached_volumes:
                self._cached_volumes.popitem(last=False)
        return volume

    def _load_tile(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.tile_refs[int(idx)]
        image_hwc, semantic_hw, instance_hw = self._cached_volume(ref.hdf_path)
        image, semantic, instance = extract_tile_arrays(
            image_hwc,
            semantic_hw,
            instance_hw,
            ref.tile,
            self.tile_read_size,
            self.output_patch_size,
            protocol=self.tile_protocol,
            renumber_instances=self.renumber_instances,
        )
        if semantic is None:
            semantic = np.zeros((self.output_patch_size, self.output_patch_size), dtype=np.int64)
        if instance is None:
            instance = np.zeros_like(semantic)
        return image, semantic.astype(np.int64, copy=False), instance.astype(np.int64, copy=False)

    def _make_batch(self, indices: Sequence[int]) -> dict[str, Any]:
        if self._pool is not None and len(indices) > 1:
            loaded = list(self._pool.map(self._load_tile, indices))
        else:
            loaded = [self._load_tile(i) for i in indices]
        images = []
        masks = []
        instances = []
        for rgb, semantic, instance in loaded:
            if self.image_transform is None:
                tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
            else:
                tensor = patch_to_tensor(rgb, self.image_transform)
            images.append(tensor)
            masks.append(semantic)
            instances.append(instance)
        return {
            "image": torch.stack(images, dim=0),
            "semantic_mask": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.long),
            "instance_mask": torch.as_tensor(np.stack(instances, axis=0), dtype=torch.long),
        }
