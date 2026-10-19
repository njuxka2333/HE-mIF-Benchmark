"""Geometry helpers for paper-native HE feeds (numpy only, no checkpoints)."""

from __future__ import annotations

import numpy as np

from models.mif_feed import prepare_he_feed, stitch_predictions


def test_whole_window_is_one_tile() -> None:
    he = np.zeros((64, 64, 3), dtype=np.uint8)
    tiles, coords, tile_px, phys_px = prepare_he_feed(
        "hemit", he, patch_um=16.0, model_mpp=0.25, tile_px=64, feed="whole"
    )
    assert len(tiles) == 1
    assert coords == [(0, 0)]
    assert tile_px == phys_px == 64


def test_histoplexer_single_tile_passthrough() -> None:
    pred = np.ones((11, 32, 32), np.float32)
    out = stitch_predictions("histoplexer", [pred], [(0, 0)], tile_px=128, phys_px=128)
    assert out.shape == (11, 32, 32)


def test_miphei_mosaic_is_2x2_at_half_mpp() -> None:
    he = np.zeros((32, 32, 3), dtype=np.uint8)
    tiles, coords, tile_px, phys_px = prepare_he_feed(
        "mipheivit", he, patch_um=256.0, model_mpp=0.5, tile_px=256, feed="mosaic"
    )
    assert phys_px == 512
    assert tile_px == 256
    assert len(tiles) == 4
    assert coords == [(0, 0), (0, 256), (256, 0), (256, 256)]


def test_miphei_stitch_crops_padding() -> None:
    tiles = [np.full((2, 4, 4), fill, dtype=np.float32) for fill in (1.0, 2.0, 3.0, 4.0)]
    coords = [(0, 0), (0, 4), (4, 0), (4, 4)]
    out = stitch_predictions("mipheivit", tiles, coords, tile_px=4, phys_px=6)
    assert out.shape == (2, 6, 6)
    assert float(out[0, 0, 0]) == 1.0
    assert float(out[0, 0, 5]) == 2.0
    assert float(out[0, 5, 0]) == 3.0
