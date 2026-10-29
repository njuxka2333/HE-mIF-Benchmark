import numpy as np
import torch

from data.loader import get_dataset_config, load_split_samples
from data.pathocell.sampler import (
    DistributedWeightedSampler,
    HDFTileBatchSampler,
    _TileIndexDataset,
)


def test_distributed_weighted_sampler_shards_one_global_draw_sequence():
    dataset = _TileIndexDataset(8)
    weights = np.arange(1, 9, dtype=np.float64)
    rank0 = DistributedWeightedSampler(dataset, weights, num_replicas=2, rank=0, seed=42)
    rank1 = DistributedWeightedSampler(dataset, weights, num_replicas=2, rank=1, seed=42)
    rank0.set_epoch(3)
    rank1.set_epoch(3)

    generator = torch.Generator().manual_seed(45)
    global_draws = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double),
        rank0.total_size,
        replacement=True,
        generator=generator,
    ).tolist()
    assert list(rank0) == global_draws[0::2]
    assert list(rank1) == global_draws[1::2]


def test_hdf_tile_sampler_batch_shapes():
    cfg = get_dataset_config("pathocell")
    samples = load_split_samples(cfg, split_filter="train", max_samples=1)
    if not samples:
        return
    sampler = HDFTileBatchSampler(
        experiment_config=type(
            "Exp",
            (),
            {
                "target_patch_size": 224,
                "target_mpp": 0.5,
                "use_mpp": True,
            },
        )(),
        samples=samples,
        image_transform=None,
        batch_size=2,
        seed=0,
    )
    batch = sampler.next_batch()
    assert batch["image"].shape[0] == 2
    assert batch["semantic_mask"].shape[0] == 2
    assert batch["image"].ndim == 4
    if "instance_mask" in batch:
        assert batch["instance_mask"].shape == batch["semantic_mask"].shape
    sampler.close()


def test_pathocell_volume_contains_instance_mask():
    cfg = get_dataset_config("pathocell")
    samples = load_split_samples(cfg, split_filter="train", max_samples=1)
    if not samples or samples[0].image_path is None:
        return
    from data.pathocell.hdf import load_pathocell_volume

    _, semantic, instance = load_pathocell_volume(samples[0].image_path)
    assert semantic is not None
    assert instance is not None
