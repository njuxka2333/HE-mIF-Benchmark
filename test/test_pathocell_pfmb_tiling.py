"""Tests for PFMB-aligned PathoCell tiling (linspace + center-pad + instance renumber)."""

from __future__ import annotations

import numpy as np

from data.tiling import (
    PFMB_TILE_PROTOCOL,
    extract_tile_arrays,
    pfmb_grid,
    plan_tiles,
    renumber_instance_ids,
)


def test_pfmb_linspace_tile_count_matches_schuerch_fov():
    # Typical PathoCell FOV size at 20x.
    tiles = pfmb_grid(1440, 1920, 224)
    assert len(tiles) == 63


def test_pfmb_tiling_center_pads_edge_tiles_to_read_size():
    volume = np.zeros((500, 500, 3), dtype=np.uint8)
    semantic = np.zeros((500, 500), dtype=np.int64)
    instance = np.zeros((500, 500), dtype=np.int64)
    _, tiles = plan_tiles(500, 500, 224, protocol=PFMB_TILE_PROTOCOL)
    assert len(tiles) == 9
    for tile in tiles:
        image, sem, inst = extract_tile_arrays(
            volume,
            semantic,
            instance,
            tile,
            224,
            224,
            protocol=PFMB_TILE_PROTOCOL,
        )
        assert image.shape == (224, 224, 3)
        assert sem.shape == (224, 224)
        assert inst.shape == (224, 224)


def test_pfmb_instance_renumber_per_tile():
    instance = np.array(
        [
            [0, 5, 0],
            [5, 0, 9],
            [0, 0, 9],
        ],
        dtype=np.int64,
    )
    renumbered = renumber_instance_ids(instance)
    assert set(np.unique(renumbered).tolist()) == {0, 1, 2}


def test_pfmb_tiling_finds_all_quadrant_instances():
    from metrics import get_instance_level_labels

    height = width = 448
    tile = 224
    volume = np.zeros((height, width, 3), dtype=np.uint8)
    semantic_hw = np.zeros((height, width), dtype=np.int64)
    instance_hw = np.zeros((height, width), dtype=np.int64)
    instance_hw[32, 32] = 1
    instance_hw[32, 352] = 2
    instance_hw[352, 32] = 3
    instance_hw[352, 352] = 4
    for inst_id, (y, x) in enumerate([(32, 32), (32, 352), (352, 32), (352, 352)], start=1):
        semantic_hw[y, x] = inst_id

    label_dict = {0: "background", 1: "c1", 2: "c2", 3: "c3", 4: "c4"}
    _, tiles = plan_tiles(height, width, tile, protocol=PFMB_TILE_PROTOCOL)

    total = 0
    for tile_bounds in tiles:
        _, semantic, instance = extract_tile_arrays(
            volume,
            semantic_hw,
            instance_hw,
            tile_bounds,
            tile,
            tile,
            protocol=PFMB_TILE_PROTOCOL,
            renumber_instances=True,
        )
        oracle = np.stack([(semantic == cls).astype(np.float32) for cls in label_dict])
        y_true, _ = get_instance_level_labels(oracle, semantic, instance, label_dict)
        total += len(y_true)
    assert total == 4
