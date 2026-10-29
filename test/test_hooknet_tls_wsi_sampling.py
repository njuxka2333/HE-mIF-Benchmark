"""Tests for HookNet-TLS wholeslidedata helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.hooknet_tls.annotations import HOOKNET_TLS_LABELS, xml_path_for_slide
from data.hooknet_tls.wsi_sampling import iter_tls_sliding_patch_coords, tls_raster_foreground_pixels
from data.loader import get_dataset_config
from conftest import TEST_DATA_CONFIG


def test_hooknet_tls_label_map():
    assert HOOKNET_TLS_LABELS["tls"] == 1
    assert HOOKNET_TLS_LABELS["gc"] == 2


def test_hooknet_tls_paths():
    cfg = get_dataset_config("hooknet_tls", str(TEST_DATA_CONFIG))
    cfg.annotations_subdir = "annotations/data_v2/annotations_svs_spacings"
    slide = "TCGA-2F-A9KO-01Z-00-DX1"
    # test fixture may not ship XML; function should still resolve path object
    xml = xml_path_for_slide(cfg, slide)
    assert xml is None or isinstance(xml, Path)


class _FakeWSI:
    spacings = [0.252]

    def __init__(self, width: int, height: int, *, ratio: int = 2):
        self.shapes = [(width, height)]
        self._ratio = ratio

    def get_downsampling_from_spacing(self, spacing: float) -> int:
        del spacing
        return self._ratio


def test_tls_raster_foreground_filter():
    # 10x10 mask with a 2x2 TLS blob; WSI 100x100 maps 1 mask px per 10 WSI px
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    wsi = _FakeWSI(100, 100, ratio=2)
    patch_size = 10
    spacing = 0.5

    assert tls_raster_foreground_pixels(mask, wsi, 40, 40, patch_size, spacing) > 0
    assert tls_raster_foreground_pixels(mask, wsi, 0, 0, patch_size, spacing) == 0

    coords = list(
        iter_tls_sliding_patch_coords(
            wsi,
            mask,
            patch_size=patch_size,
            spacing=spacing,
            min_foreground_pixels=1,
        )
    )
    assert coords
    assert all(
        tls_raster_foreground_pixels(mask, wsi, c["x"], c["y"], patch_size, spacing) >= 1 for c in coords
    )
