"""Geometry invariants for canonical physical HookNet-TLS tiles."""

from data.hooknet_tls.physical_grid import (
    physical_bounds_to_grid_indices,
    physical_fov_um,
    physical_grid_edges,
)


def test_physical_grid_edges_are_adjacent_after_rounding():
    fov = physical_fov_um(1024, 0.25)
    edges = [physical_grid_edges(i, source_mpp=0.19448, fov_um=fov) for i in range(8)]
    assert all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1))
    assert all(right > left for left, right in edges)


def test_physical_grid_has_target_fov():
    assert physical_fov_um(1024, 0.25) == 256.0
    x0, x1 = physical_grid_edges(1, source_mpp=0.25, fov_um=256.0)
    assert (x0, x1) == (1024, 2048)


def test_polygon_bounds_select_physical_cells():
    cells = physical_bounds_to_grid_indices(
        0, 0, 512, 512, source_mpp=0.25, fov_um=256.0
    )
    assert cells == {(0, 0)}
    cells = physical_bounds_to_grid_indices(
        1024, 1024, 1025, 1025, source_mpp=0.25, fov_um=256.0
    )
    assert cells == {(1, 1)}
