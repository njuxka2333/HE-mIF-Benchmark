from pathlib import Path

import torch
import yaml

from heads import (
    FeatureUNETR,
    SegmentationFeaturePipeline,
    build_segmentation_head,
)
from training.config import load_eval_config
from training.offline_train import _validate_task_config
from training.online_segmentation_train import _assert_pathocell_segmentation_head
from heads.unetr import Deconv2DBlock, PfmbUNETR
from heads.vit_pyramid import CellViTDeconvBlock, ViTFeaturePyramidBuilder


LAYERS = ["z1", "z2", "z3", "z4"]


def _small_builder(*, grid_channels: int = 8) -> ViTFeaturePyramidBuilder:
    return ViTFeaturePyramidBuilder(
        feature_layers=LAYERS,
        in_channels=[grid_channels] * 4,
        embed_dim=grid_channels,
        skip_dim_11=grid_channels,
        skip_dim_12=grid_channels,
        bottleneck_dim=grid_channels,
        mid_dim=grid_channels,
        shallow_dim=grid_channels,
    )


def _features(grid: int, channels: int = 8) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(1, channels, grid, grid)
        for name in LAYERS
    }


def test_builder_reuses_pfmb_cellvit_deconvolution_block():
    assert Deconv2DBlock is CellViTDeconvBlock


def test_builder_constructs_cellvit_8_4_2_1_pyramid():
    builder = _small_builder().eval()
    with torch.no_grad():
        outputs = builder(_features(grid=3))

    assert list(outputs) == LAYERS
    assert [tuple(outputs[name].shape) for name in LAYERS] == [
        (1, 8, 24, 24),
        (1, 8, 12, 12),
        (1, 8, 6, 6),
        (1, 8, 3, 3),
    ]
    assert builder.last_trace["input_shapes"]["z4"] == [1, 8, 3, 3]
    assert builder.last_trace["output_shapes"]["z1"] == [1, 8, 24, 24]


def test_builder_preserves_miphei_native_18_grid_as_deepest_level():
    builder = _small_builder().eval()
    with torch.no_grad():
        outputs = builder(_features(grid=18))

    assert [tuple(outputs[name].shape[-2:]) for name in LAYERS] == [
        (144, 144),
        (72, 72),
        (36, 36),
        (18, 18),
    ]


def test_builder_rejects_nonuniform_input_grids():
    features = _features(grid=4)
    features["z4"] = torch.randn(1, 8, 2, 2)
    builder = _small_builder()
    try:
        builder(features)
    except ValueError as exc:
        assert "requires four equal-grid ViT maps" in str(exc)
    else:
        raise AssertionError("nonuniform maps must not enter CellViT-style processing")


def test_feature_unetr_adapters_receive_processed_pyramid():
    processor_config = {
        "name": "cellvit_pyramid",
        "embed_dim": 8,
        "skip_dim_11": 8,
        "skip_dim_12": 8,
        "bottleneck_dim": 8,
        "mid_dim": 8,
        "shallow_dim": 8,
    }
    model = build_segmentation_head(
        "unetr",
        feature_layers=LAYERS,
        in_channels=[8] * 4,
        num_classes=3,
        head_config={
            "decoder_mode": "feature",
            "adapter_channels": [8] * 4,
            "feature_processor": processor_config,
        },
    ).eval()
    assert isinstance(model, SegmentationFeaturePipeline)
    assert isinstance(model.decoder, FeatureUNETR)
    adapter_inputs: dict[str, tuple[int, ...]] = {}
    handles = [
        model.decoder.adapters[name].register_forward_pre_hook(
            lambda _module, args, layer=name: adapter_inputs.__setitem__(
                layer, tuple(args[0].shape)
            )
        )
        for name in LAYERS
    ]
    try:
        with torch.no_grad():
            logits = model(
                _features(grid=4),
                image=torch.randn(1, 3, 32, 32),
                output_size=(32, 32),
            )
    finally:
        for handle in handles:
            handle.remove()

    assert [adapter_inputs[name][-2:] for name in LAYERS] == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
    ]
    assert logits.shape == (1, 3, 32, 32)


def test_build_segmentation_head_accepts_feature_processor_config():
    model = build_segmentation_head(
        "unetr",
        feature_layers=LAYERS,
        in_channels=[8] * 4,
        num_classes=3,
        head_config={
            "decoder_mode": "feature",
            "adapter_channels": [8] * 4,
            "feature_processor": {
                "name": "cellvit_pyramid",
                "embed_dim": 8,
                "skip_dim_11": 8,
                "skip_dim_12": 8,
                "bottleneck_dim": 8,
                "mid_dim": 8,
                "shallow_dim": 8,
            },
        },
    )
    assert isinstance(model, SegmentationFeaturePipeline)
    assert isinstance(model.feature_processor, ViTFeaturePyramidBuilder)
    assert isinstance(model.decoder, FeatureUNETR)
    assert not hasattr(model.decoder, "feature_processor")


def test_vit_configs_enable_shared_processor_with_task_specific_decoders():
    root = Path(__file__).resolve().parents[1] / "configs" / "downstream"
    expected = {
        "tls_region_segmentation/hooknet_tls/encoder/uni2.yaml": ("feature", "bilinear_pyramid"),
        "cell_semantic_segmentation/pathocell/encoder/uni2.yaml": ("pfmb", None),
    }
    for filename, (decoder_mode, processor_name) in expected.items():
        config = yaml.safe_load((root / filename).read_text(encoding="utf-8"))
        assert config["head"]["decoder_mode"] == decoder_mode
        processor = config["head"].get("feature_processor") or {}
        assert processor.get("name") == processor_name


def test_miphei_cell_pfmb_passes_online_and_offline_task_guards():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "downstream"
        / "cell_semantic_segmentation"
        / "pathocell"
        / "encoder"
        / "uni2.yaml"
    )
    config = load_eval_config(config_path)
    _assert_pathocell_segmentation_head(config)
    _validate_task_config(config)


def test_shared_processor_can_feed_pfmb_without_rebuilding_vit_skips():
    model = build_segmentation_head(
        "unetr",
        feature_layers=LAYERS,
        in_channels=[8] * 4,
        num_classes=3,
        head_config={
            "decoder_mode": "pfmb",
            "feature_processor": {
                "name": "cellvit_pyramid",
                "embed_dim": 8,
            },
        },
    ).eval()

    assert isinstance(model, SegmentationFeaturePipeline)
    assert isinstance(model.feature_processor, ViTFeaturePyramidBuilder)
    assert isinstance(model.decoder, PfmbUNETR)
    assert model.decoder.preprocessed_vit
    assert isinstance(model.decoder.decoder1, torch.nn.Identity)
    assert isinstance(model.decoder.decoder2, torch.nn.Identity)
    assert isinstance(model.decoder.decoder3, torch.nn.Identity)

    with torch.no_grad():
        logits = model(
            _features(grid=2),
            image=torch.randn(1, 3, 32, 32),
            output_size=(32, 32),
        )
    assert logits.shape == (1, 3, 32, 32)


def test_external_processor_preserves_pfmb_uniform_path_outputs():
    torch.manual_seed(7)
    legacy = PfmbUNETR(
        feature_layers=LAYERS,
        in_channels=[8] * 4,
        num_classes=3,
        embed_dim=8,
        drop_rate=0.0,
    ).eval()
    refactored = build_segmentation_head(
        "unetr",
        feature_layers=LAYERS,
        in_channels=[8] * 4,
        num_classes=3,
        head_config={
            "decoder_mode": "pfmb",
            "embed_dim": 8,
            "feature_processor": {
                "name": "cellvit_pyramid",
                "embed_dim": 8,
            },
        },
    ).eval()
    assert isinstance(refactored, SegmentationFeaturePipeline)
    assert isinstance(refactored.decoder, PfmbUNETR)

    refactored.load_state_dict(legacy.state_dict(), strict=True)

    features = _features(grid=2)
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        legacy_logits = legacy(features, image=image, output_size=(32, 32))
        refactored_logits = refactored(features, image=image, output_size=(32, 32))
    torch.testing.assert_close(refactored_logits, legacy_logits)
