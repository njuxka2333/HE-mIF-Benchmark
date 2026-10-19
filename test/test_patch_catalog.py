"""TLS patches.csv uses the same crop columns as Cell-niches."""

from __future__ import annotations

from pathlib import Path

from data.hooknet_tls import PatchPlanRecord, load_selected_tiles_csv, write_selected_tiles_csv
from data.patch_catalog import TLS_PATCH_FIELDS


def _record() -> PatchPlanRecord:
    return PatchPlanRecord(
        slide_id="TCGA-18-3406-01Z-00-DX1",
        patient_id="TCGA-18-3406",
        split="train",
        grid_ix=52,
        grid_iy=13,
        x_level0=52825,
        y_level0=13206,
        patch_size=1024,
        stride=1024,
        wsi_path="",
        valid_pixel_count=62447,
        count_tls=62447,
        count_gc=0,
        count_rest=0,
        valid_fraction=0.059554100036621094,
        image_relpath="train/images/TCGA-18-3406-01Z-00-DX1_ix52_iy13.png",
        mask_relpath="train/masks/TCGA-18-3406-01Z-00-DX1_ix52_iy13.png",
        source_width_l0=1016,
        source_height_l0=1016,
        source_mpp=0.2519999445422657,
    )


def test_tls_patches_csv_roundtrip(tmp_path: Path):
    path = tmp_path / "patches.csv"
    record = _record()
    write_selected_tiles_csv(path, [record])
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:9] == TLS_PATCH_FIELDS[:9]
    assert "x_he" in header and "x_level0" not in header
    loaded = load_selected_tiles_csv(path)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.slide_id == record.slide_id
    assert got.split == "train"
    assert got.grid_ix == 52 and got.grid_iy == 13
    assert got.x_level0 == 52825 and got.y_level0 == 13206
    assert got.source_width_l0 == 1016 and got.source_height_l0 == 1016
    assert abs(got.source_mpp - record.source_mpp) < 1e-12
    assert got.image_relpath.endswith("train/images/TCGA-18-3406-01Z-00-DX1_ix52_iy13.png")
    assert got.mask_relpath.endswith("train/masks/TCGA-18-3406-01Z-00-DX1_ix52_iy13.png")
