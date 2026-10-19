"""End-to-end patch extraction via feature_extraction/extract_features_fp.py (production code path)."""

from __future__ import annotations

import h5py
import numpy as np


def test_wsi_target_mpp_read_size_uses_model_patch_size(tmp_path):
    from PIL import Image

    from data.dataset_h5 import Whole_Slide_Bag_FP

    h5_path = tmp_path / "coords.h5"
    with h5py.File(h5_path, "w") as handle:
        coords = handle.create_dataset("coords", data=np.array([[0, 0]], dtype=np.int32))
        coords.attrs["patch_level"] = 0
        coords.attrs["patch_size"] = 256

    class RecordingWsi:
        properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

        def __init__(self):
            self.last_size = None

        def read_region(self, coord, level, size):
            del coord, level
            self.last_size = tuple(size)
            return Image.new("RGB", tuple(size), color=(200, 200, 200))

    wsi = RecordingWsi()
    dataset = Whole_Slide_Bag_FP(
        file_path=str(h5_path),
        wsi=wsi,
        target_patch_size=1024,
        target_mpp=0.25,
        fallback_mpp=0.25,
        fast_read=True,
    )
    image, _, _ = dataset[0]

    assert wsi.last_size == (1024, 1024)
    assert image.shape[-2:] == (1024, 1024)
