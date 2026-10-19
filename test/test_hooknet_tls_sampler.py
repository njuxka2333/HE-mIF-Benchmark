"""TLS PFMB tiling over the canonical 1024 px cell (no center crop)."""

from __future__ import annotations

import numpy as np

from data.hooknet_tls import IGNORE_LABEL
from data.hooknet_tls_sampler import (
    extract_tls_pfmb_tile,
    tls_source_read_size,
)
from data.pathocell_hdf import pfmb_linspace_tiles


def test_tls_read_size_matches_native_geometry():
    assert tls_source_read_size(1024, 0.25, reference_mpp=0.25) == 1024
    assert tls_source_read_size(256, 0.5, reference_mpp=0.25) == 512
    assert tls_source_read_size(256, 0.25, reference_mpp=0.25) == 256
    assert tls_source_read_size(224, 0.5, reference_mpp=0.25) == 448


def test_pfmb_tiles_cover_canonical_cell_without_overlap():
    for read_size in (1024, 512, 256, 448):
        tiles = pfmb_linspace_tiles(1024, 1024, read_size)
        covered = np.zeros((1024, 1024), dtype=np.int32)
        for y0, y1, x0, x1 in tiles:
            covered[y0:y1, x0:x1] += 1
        assert covered.min() == 1
        assert covered.max() == 1
        assert len(tiles) == int(np.ceil(1024 / read_size)) ** 2


def test_corner_label_survives_pfmb_tiles_not_center_crop():
    rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.int64)
    mask[10:40, 10:40] = 2  # GC in the corner; a 512 center crop would drop it
    tiles = pfmb_linspace_tiles(1024, 1024, 512)
    found = False
    for tile in tiles:
        _image, out_mask = extract_tls_pfmb_tile(rgb, mask, tile, 512, 256)
        if np.count_nonzero(out_mask == 2) > 0:
            found = True
    assert found
    center = mask[256:768, 256:768]
    assert np.count_nonzero(center == 2) == 0


def test_short_edge_tile_is_padded_with_ignore():
    rgb = np.ones((1024, 1024, 3), dtype=np.uint8)
    mask = np.ones((1024, 1024), dtype=np.int64)
    tiles = pfmb_linspace_tiles(1024, 1024, 448)
    short = [t for t in tiles if (t[1] - t[0]) < 448 or (t[3] - t[2]) < 448]
    assert short
    image, out_mask = extract_tls_pfmb_tile(rgb, mask, short[0], 448, 224)
    assert image.shape == (224, 224, 3)
    assert out_mask.shape == (224, 224)
    assert IGNORE_LABEL in set(np.unique(out_mask).tolist())


def test_index_sampler_reads_one_real_canonical_tile():
    from pathlib import Path
    from types import SimpleNamespace

    from data.hooknet_tls_sampler import FixedGridIndexSampler

    root = Path("/data3/houjunlin/fxx/datasets/HookNet_TLS_fixed_grid_ps1024")
    if not (root / "patches.csv").is_file():
        return
    exp = SimpleNamespace(target_patch_size=256, target_mpp=0.5, iterator_cpus=1)
    official = SimpleNamespace(
        fixed_grid_root=root,
        sampler_batch_size=2,
        seed=0,
        fixed_grid_min_selected_class_pixels=1,
        fixed_grid_max_resample_attempts=4,
    )
    sampler = FixedGridIndexSampler(
        exp, samples=None, official=official, transform=None, split="valid", patch_workers=1
    )
    try:
        assert sampler.read_size == 512
        assert len(sampler.tiles) == 4
        assert len(sampler) == 4 * 3716 or len(sampler) > 4
        sampler.begin_full_pass()
        batch = sampler.next_batch()
        assert batch["image"].shape == (2, 3, 256, 256)
        assert batch["semantic_mask"].shape == (2, 256, 256)
    finally:
        sampler.close()
