"""``python -m training.train --config`` routes all downstream YAMLs."""

from __future__ import annotations

from pathlib import Path

from training.train import _config_route

DOWNSTREAM = Path(__file__).resolve().parents[1] / "configs" / "downstream"


def test_train_router_uses_decoder_mode_not_task_name():
    assert (
        _config_route(DOWNSTREAM / "prognosis/tcga_coad/encoder/rosie.yaml") == "offline"
    )
    assert (
        _config_route(DOWNSTREAM / "cell_semantic_segmentation/pathocell/encoder/hemit.yaml") == "online"
    )
    assert (
        _config_route(DOWNSTREAM / "cell_semantic_segmentation/pathocell/output/hemit.yaml") == "virtual_staining_unet"
    )
    assert _config_route(DOWNSTREAM / "tls_region_segmentation/hooknet_tls/encoder/rosie.yaml") == "online"
    assert _config_route(DOWNSTREAM / "tls_region_segmentation/hooknet_tls/output/rosie.yaml") == "virtual_staining_feature"
    assert (
        _config_route(DOWNSTREAM / "virtual_staining/cell_niches/output/rosie.yaml")
        == "virtual_staining_score"
    )
    assert (
        _config_route(DOWNSTREAM / "virtual_staining/cell_niches/output/mipheivit.yaml")
        == "virtual_staining_score"
    )


def test_every_downstream_yaml_has_a_known_route():
    known = {
        "offline",
        "online",
        "virtual_staining_unet",
        "virtual_staining_feature",
        "virtual_staining_score",
    }
    paths = sorted(DOWNSTREAM.rglob("*.yaml"))
    assert paths
    missing = [str(path) for path in paths if _config_route(path) not in known]
    assert not missing, missing
