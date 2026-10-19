"""Unit tests for PathoCell HDF tiling helpers."""

from __future__ import annotations

from conftest import load_dataset_samples
from data.pathocell_hdf import (
    PATHOCELL_NATIVE_MPP,
    PATHOCELL_TILING_MINIMAL,
    pathocell_tile_geometry,
    plan_pathocell_tiles,
    read_pathocell_hdf,
    tile_top_left,
)


def test_pathocell_tiles_pad_when_volume_too_small():
    planned_read, _ = pathocell_tile_geometry(1024, 0.25)
    read_size, tiles = plan_pathocell_tiles(
        500, 500, planned_read, protocol=PATHOCELL_TILING_MINIMAL
    )
    assert read_size == planned_read
    assert tile_top_left(tiles[0]) == ((500 - planned_read) // 2, (500 - planned_read) // 2)


def test_pathocell_tile_geometry_at_native_mpp():
    read_size, output_size = pathocell_tile_geometry(128, 0.5)
    assert read_size == round(128 * 0.5 / PATHOCELL_NATIVE_MPP)
    assert output_size == 128


def test_pathocell_hdf_helpers_grid_tiles_volume():
    _, samples = load_dataset_samples("pathocell", max_samples=1)
    sample = samples[0]
    image, mask, _instance = read_pathocell_hdf(sample.image_path)
    assert image.mode == "RGB"
    assert mask is not None

    planned_read, _ = pathocell_tile_geometry(128, 0.5)
    read_size, expected_tiles = plan_pathocell_tiles(mask.shape[0], mask.shape[1], planned_read)
    assert read_size == planned_read
    assert len(expected_tiles) > 1
    assert tile_top_left(expected_tiles[0]) == (0, 0)
