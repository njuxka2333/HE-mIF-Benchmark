"""Unit tests for PathoCell volume IO and shared tile geometry."""

from __future__ import annotations

from conftest import load_dataset_samples
from data.pathocell.hdf import PATHOCELL_NATIVE_MPP, read_pathocell_hdf
from data.tiling import plan_tiles, tile_geometry, tile_top_left


def test_pathocell_pfmb_tiles_start_at_top_left_when_volume_is_small():
    planned_read, _ = tile_geometry(1024, 0.25, native_mpp=PATHOCELL_NATIVE_MPP)
    read_size, tiles = plan_tiles(500, 500, planned_read)
    assert read_size == planned_read
    assert tile_top_left(tiles[0]) == (0, 0)


def test_pathocell_tile_geometry_at_native_mpp():
    read_size, output_size = tile_geometry(
        128, 0.5, native_mpp=PATHOCELL_NATIVE_MPP
    )
    assert read_size == round(128 * 0.5 / PATHOCELL_NATIVE_MPP)
    assert output_size == 128


def test_pathocell_volume_grid_tiles():
    _, samples = load_dataset_samples("pathocell", max_samples=1)
    sample = samples[0]
    image, mask, _instance = read_pathocell_hdf(sample.image_path)
    assert image.mode == "RGB"
    assert mask is not None

    planned_read, _ = tile_geometry(128, 0.5, native_mpp=PATHOCELL_NATIVE_MPP)
    read_size, expected_tiles = plan_tiles(mask.shape[0], mask.shape[1], planned_read)
    assert read_size == planned_read
    assert len(expected_tiles) > 1
    assert tile_top_left(expected_tiles[0]) == (0, 0)
