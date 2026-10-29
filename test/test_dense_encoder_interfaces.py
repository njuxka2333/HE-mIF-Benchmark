from pathlib import Path

import pytest
import torch
import yaml

from heads import (
    FeatureUNETR,
    SegmentationFeaturePipeline,
    build_segmentation_head,
)
from heads.unetr import PfmbUNETR, classify_encoder_pyramid
from heads.interfaces import (
    get_default_interface,
    validate_dense_config,
    validate_dense_runtime_features,
)
from heads.vit_pyramid import ViTFeaturePyramidBuilder


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/downstream"
PRODUCTION_CONFIGS = {
    "gigatime": (
        "cell_semantic_segmentation/pathocell/encoder/gigatime.yaml",
        "tls_region_segmentation/hooknet_tls/encoder/gigatime.yaml",
    ),
    "hemit": (
        "cell_semantic_segmentation/pathocell/encoder/hemit.yaml",
        "tls_region_segmentation/hooknet_tls/encoder/hemit.yaml",
    ),
    "histoplexer": (
        "cell_semantic_segmentation/pathocell/encoder/histoplexer.yaml",
        "tls_region_segmentation/hooknet_tls/encoder/histoplexer.yaml",
    ),
    "uni2": (
        "cell_semantic_segmentation/pathocell/encoder/uni2.yaml",
        "tls_region_segmentation/hooknet_tls/encoder/uni2.yaml",
    ),
    "mipheivit": (
        "cell_semantic_segmentation/pathocell/encoder/mipheivit.yaml",
        "tls_region_segmentation/hooknet_tls/encoder/mipheivit.yaml",
    ),
}


def _raw_config(filename: str) -> dict:
    with (CONFIG_DIR / filename).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("model", PRODUCTION_CONFIGS)
def test_dense_tasks_share_canonical_interface_and_preprocessing(model):
    expected = get_default_interface(model)
    cell_filename, tls_filename = PRODUCTION_CONFIGS[model]
    cell = _raw_config(cell_filename)
    tls = _raw_config(tls_filename)

    assert cell["experiment"].get("dense_interface", "default") == "default"
    assert cell["experiment"]["target_features"] == expected["target_features"]
    assert cell["head"]["decoder_mode"] in {"pfmb", "cellseg"}
    assert tls["head"]["decoder_mode"] == "feature"
    assert tls["run"]["fixed_grid_sampling"]["fixed_grid_root"]


@pytest.mark.parametrize(
    ("model", "grids"),
    [
        ("gigatime", [32, 16, 8, 4, 2]),
        ("hemit", [32, 16, 8, 8]),
        ("histoplexer", [64, 32, 16, 8]),
        ("uni2", [16, 16, 16, 16]),
        ("mipheivit", [18, 18, 18, 18]),
    ],
)
def test_runtime_validator_enforces_canonical_geometry(model, grids):
    interface = get_default_interface(model)
    features = {
        name: torch.zeros(1, channels, grid, grid)
        for name, channels, grid in zip(
            interface["target_features"], interface["raw_channels"], grids
        )
    }
    assert list(validate_dense_runtime_features(model, features)) == interface[
        "target_features"
    ]


def test_native_models_bypass_builder_and_vits_share_builder_class():
    for model in ("gigatime", "hemit", "histoplexer"):
        interface = get_default_interface(model)
        assert interface.get("feature_processor") is None

    built = []
    for model in ("uni2", "mipheivit"):
        interface = get_default_interface(model)
        head = build_segmentation_head(
            "unetr",
            feature_layers=interface["target_features"],
            in_channels=interface["raw_channels"],
            num_classes=3,
            head_config={
                "decoder_mode": "pfmb",
                "feature_processor": {"name": interface["feature_processor"]},
            },
        )
        assert isinstance(head, SegmentationFeaturePipeline)
        assert isinstance(head.feature_processor, ViTFeaturePyramidBuilder)
        assert isinstance(head.decoder, PfmbUNETR)
        built.append(type(head.feature_processor))
    assert built[0] is built[1] is ViTFeaturePyramidBuilder


def test_hemit_h1_is_retained_without_pool_or_resize():
    raw_cell = _raw_config(PRODUCTION_CONFIGS["hemit"][0])
    raw_tls = _raw_config(PRODUCTION_CONFIGS["hemit"][1])
    assert raw_cell["experiment"]["target_features"] == [
        "fused_z1",
        "fused_z2",
        "fused_z3",
        "z4",
    ]
    assert raw_tls["experiment"]["target_features"] == ["z1", "z2", "z3", "z4"]
    for raw in (raw_cell, raw_tls):
        assert "feature_processor" not in raw["head"]
        assert not raw["training"].get("phase1c_pool_z4", False)
    assert classify_encoder_pyramid([(32, 32), (16, 16), (8, 8), (8, 8)]) == (
        "deepest_plateau"
    )


def test_pfmb_accepts_gigatime_five_level_native_interface():
    interface = get_default_interface("gigatime")
    head = build_segmentation_head(
        "unetr",
        feature_layers=interface["target_features"],
        in_channels=interface["raw_channels"],
        num_classes=3,
        head_config={"decoder_mode": "pfmb"},
    )
    assert isinstance(head, PfmbUNETR)
    assert classify_encoder_pyramid(
        [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
    ) == "native5"


def test_tls_histoplexer_keeps_official_z2345_pyramid():
    raw = _raw_config("tls_region_segmentation/hooknet_tls/encoder/histoplexer.yaml")
    assert raw["head"]["decoder_mode"] == "feature"
    assert raw["experiment"]["target_features"] == ["z2", "z3", "z4", "z5"]


def test_decoder_choice_does_not_change_encoder_interface():
    interface = get_default_interface("histoplexer")
    feature = build_segmentation_head(
        "unetr",
        feature_layers=interface["target_features"],
        in_channels=interface["raw_channels"],
        num_classes=3,
        head_config={"decoder_mode": "feature"},
    )
    pfmb = build_segmentation_head(
        "unetr",
        feature_layers=interface["target_features"],
        in_channels=interface["raw_channels"],
        num_classes=3,
        head_config={"decoder_mode": "pfmb"},
    )
    assert isinstance(feature, FeatureUNETR)
    assert isinstance(pfmb, PfmbUNETR)
    assert feature.feature_layers == pfmb.feature_layers == interface["target_features"]
