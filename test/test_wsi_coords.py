"""CLAM contour-grid coordinates are deterministic for a known box."""

from __future__ import annotations

import numpy as np

from preprocessing.wsi.coords import contour_grid_coords


def test_contour_grid_keeps_four_pt_cells_inside_tissue():
    contour = np.array(
        [[[0, 0]], [[2048, 0]], [[2048, 2048]], [[0, 2048]]],
        dtype=np.int32,
    )
    coords = contour_grid_coords(
        contour,
        holes=[],
        img_w=4096,
        img_h=4096,
        patch_size=512,
        step_size=512,
        patch_downsample=(1.0, 1.0),
        use_padding=True,
        contour_fn="four_pt",
    )
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert (0, 0) in set(map(tuple, coords.tolist()))
    assert (1536, 1536) in set(map(tuple, coords.tolist()))
    assert coords[:, 0].max() < 2048
    assert coords[:, 1].max() < 2048
