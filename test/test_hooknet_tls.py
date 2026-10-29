"""Unit tests for HookNet-TLS fixed-grid patch planning."""

from __future__ import annotations

from data.hooknet_tls.planning import (
    DEFAULT_PATCH_SIZE_L0,
    PatchPlanRecord,
    apply_manifest_selection,
    check_coordinate_uniqueness,
    check_non_overlap,
    is_positive_patch,
    is_rest_only_patch,
    manifest_determinism_hash,
    polygon_bounds_to_grid_indices,
    rest_keep_count,
)


def test_polygon_bounds_to_grid_indices_basic():
    p = 1024
    assert (0, 0) in polygon_bounds_to_grid_indices(0, 0, 10, 10, p)
    assert (1, 1) in polygon_bounds_to_grid_indices(1024, 1024, 2047, 2047, p)
    assert (2, 2) in polygon_bounds_to_grid_indices(2048, 2048, 4096, 4096, p)


def test_polygon_bounds_single_cell_on_boundary():
    p = 100
    cells = polygon_bounds_to_grid_indices(100, 0, 199.999, 50, p)
    assert cells == {(1, 0)}


def test_non_overlap_synthetic_records():
    p = 64
    records = [
        PatchPlanRecord(
            slide_id="s1",
            patient_id="p1",
            split="train",
            grid_ix=0,
            grid_iy=0,
            x_level0=0,
            y_level0=0,
            patch_size=p,
            stride=p,
            wsi_path="/tmp/x.svs",
            valid_pixel_count=10,
            count_tls=10,
            count_gc=0,
            count_rest=0,
            valid_fraction=10 / (p * p),
        ),
        PatchPlanRecord(
            slide_id="s1",
            patient_id="p1",
            split="train",
            grid_ix=1,
            grid_iy=0,
            x_level0=p,
            y_level0=0,
            patch_size=p,
            stride=p,
            wsi_path="/tmp/x.svs",
            valid_pixel_count=5,
            count_tls=5,
            count_gc=0,
            count_rest=0,
            valid_fraction=5 / (p * p),
        ),
    ]
    assert check_coordinate_uniqueness(records)["passed"]
    assert check_non_overlap(records, p)["passed"]


def test_determinism_hash_stable():
    p = 32
    records = [
        PatchPlanRecord(
            slide_id="s1",
            patient_id="p1",
            split="train",
            grid_ix=1,
            grid_iy=2,
            x_level0=32,
            y_level0=64,
            patch_size=p,
            stride=p,
            wsi_path="/tmp/x.svs",
            valid_pixel_count=3,
            count_tls=1,
            count_gc=1,
            count_rest=1,
            valid_fraction=3 / (p * p),
        )
    ]
    h1 = manifest_determinism_hash(records)
    h2 = manifest_determinism_hash(list(reversed(records)))
    assert h1 == h2
    assert len(h1) == 64


def test_grid_indices_cover_bbox_extent():
    p = DEFAULT_PATCH_SIZE_L0
    cells = polygon_bounds_to_grid_indices(0, 0, p * 2 + 1, p * 2 + 1, p)
    assert (0, 0) in cells and (2, 2) in cells
    assert (3, 3) not in cells


def _record(
    *,
    tls: int = 0,
    gc: int = 0,
    rest: int = 0,
    slide_id: str = "s1",
    grid_ix: int = 0,
    grid_iy: int = 0,
) -> PatchPlanRecord:
    p = 64
    valid = tls + gc + rest
    return PatchPlanRecord(
        slide_id=slide_id,
        patient_id="p1",
        split="train",
        grid_ix=grid_ix,
        grid_iy=grid_iy,
        x_level0=grid_ix * p,
        y_level0=grid_iy * p,
        patch_size=p,
        stride=p,
        wsi_path="/tmp/x.svs",
        valid_pixel_count=valid,
        count_tls=tls,
        count_gc=gc,
        count_rest=rest,
        valid_fraction=valid / (p * p),
    )


def test_positive_and_rest_only_classification():
    assert is_positive_patch(_record(tls=1))
    assert is_positive_patch(_record(gc=1))
    assert is_positive_patch(_record(tls=1, rest=100))
    assert is_rest_only_patch(_record(rest=10))
    assert not is_rest_only_patch(_record(tls=1, rest=10))


def test_manifest_selection_keeps_all_positive_and_subsamples_rest():
    records = [
        _record(tls=1, grid_ix=0),
        _record(gc=2, grid_ix=1),
        _record(rest=50, grid_ix=2),
        _record(rest=50, grid_ix=3),
        _record(rest=50, grid_ix=4),
        _record(rest=50, grid_ix=5),
        _record(rest=50, grid_ix=6),
        _record(rest=50, grid_ix=7),
        _record(rest=50, grid_ix=8),
        _record(rest=50, grid_ix=9),
    ]
    selected, stats = apply_manifest_selection(records, rest_keep_fraction=0.10, random_seed=42)
    assert stats["positive_patch_count"] == 2
    assert stats["rest_candidate_count"] == 8
    assert stats["selected_rest_count"] == 1
    assert stats["final_patch_count"] == 3
    assert len(selected) == 3
    assert all(is_positive_patch(r) for r in selected[:2])
    assert is_rest_only_patch(selected[2])


def test_manifest_selection_deterministic_with_seed():
    records = [_record(rest=10, grid_ix=i) for i in range(20)] + [_record(tls=1, grid_ix=100)]
    a, stats_a = apply_manifest_selection(records, rest_keep_fraction=0.25, random_seed=42)
    b, stats_b = apply_manifest_selection(records, rest_keep_fraction=0.25, random_seed=42)
    assert stats_a == stats_b
    assert [r.grid_ix for r in a] == [r.grid_ix for r in b]


def test_rest_keep_count_rounding():
    assert rest_keep_count(100, 0.10) == 10
    assert rest_keep_count(0, 0.10) == 0
    assert rest_keep_count(5, 1.0) == 5
