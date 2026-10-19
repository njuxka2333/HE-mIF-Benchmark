import torch

from heads import build_segmentation_head, forward_segmentation_head
from heads.unetr import PfmbUNETR, classify_encoder_pyramid, forms_native_encoder_pyramid


def test_unetr_forward_with_z0():
    features = {
        "z1": torch.randn(2, 1536, 16, 16),
        "z2": torch.randn(2, 1536, 16, 16),
        "z3": torch.randn(2, 1536, 16, 16),
        "z4": torch.randn(2, 1536, 16, 16),
    }
    image = torch.randn(2, 3, 224, 224)
    model = PfmbUNETR(
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[1536, 1536, 1536, 1536],
        num_classes=4,
        use_z0=True,
        drop_rate=0.0,
    )
    logits = model(features, image=image, output_size=(224, 224))
    assert logits.shape == (2, 4, 224, 224)


def test_unetr_projects_mixed_channel_skips():
    features = {
        "stage0": torch.randn(2, 96, 32, 32),
        "stage1": torch.randn(2, 192, 16, 16),
        "stage2": torch.randn(2, 384, 8, 8),
        "stage3": torch.randn(2, 768, 4, 4),
    }
    image = torch.randn(2, 3, 128, 128)
    model = PfmbUNETR(
        feature_layers=["stage0", "stage1", "stage2", "stage3"],
        in_channels=[96, 192, 384, 768],
        num_classes=4,
        embed_dim=768,
        use_z0=True,
        drop_rate=0.0,
    )
    logits = model(features, image=image, output_size=(128, 128))
    assert logits.shape == (2, 4, 128, 128)


def test_forms_native_encoder_pyramid():
    assert forms_native_encoder_pyramid([(32, 32), (16, 16), (8, 8), (4, 4)])
    assert not forms_native_encoder_pyramid([(16, 16), (16, 16), (16, 16), (16, 16)])
    assert not forms_native_encoder_pyramid([(32, 32), (16, 16), (8, 8), (8, 8)])
    assert not forms_native_encoder_pyramid([(32, 32), (16, 16), (8, 8)])


def test_classify_encoder_pyramid_rejects_irregular_skips():
    assert classify_encoder_pyramid([(16, 16)] * 4) == "uniform"
    assert classify_encoder_pyramid([(32, 32), (16, 16), (8, 8), (4, 4)]) == "native"
    assert classify_encoder_pyramid([(32, 32), (16, 16), (8, 8), (8, 8)]) == (
        "deepest_plateau"
    )
    try:
        classify_encoder_pyramid([(32, 32), (16, 16), (12, 12), (6, 6)])
    except ValueError as exc:
        assert "neither four equal ViT maps" in str(exc)
    else:
        raise AssertionError("irregular feature grids must not silently use the uniform route")


def test_unetr_native_pyramid_uses_native_path():
    features = {
        "stage0": torch.randn(2, 96, 32, 32),
        "stage1": torch.randn(2, 192, 16, 16),
        "stage2": torch.randn(2, 384, 8, 8),
        "stage3": torch.randn(2, 768, 4, 4),
    }
    image = torch.randn(2, 3, 128, 128)
    model = PfmbUNETR(
        feature_layers=["stage0", "stage1", "stage2", "stage3"],
        in_channels=[96, 192, 384, 768],
        num_classes=4,
        embed_dim=768,
        use_z0=True,
        drop_rate=0.0,
    )

    def _fail_uniform(*args, **kwargs):
        raise AssertionError("uniform-grid path should not run for native pyramid skips")

    model._forward_upsample_uniform = _fail_uniform  # type: ignore[method-assign]
    logits = model(features, image=image, output_size=(128, 128))
    assert logits.shape == (2, 4, 128, 128)


def test_unetr_uniform_grid_uses_uniform_path():
    features = {
        "z1": torch.randn(2, 1536, 16, 16),
        "z2": torch.randn(2, 1536, 16, 16),
        "z3": torch.randn(2, 1536, 16, 16),
        "z4": torch.randn(2, 1536, 16, 16),
    }
    image = torch.randn(2, 3, 224, 224)
    model = PfmbUNETR(
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[1536, 1536, 1536, 1536],
        num_classes=4,
        use_z0=True,
        drop_rate=0.0,
    )

    def _fail_native(*args, **kwargs):
        raise AssertionError("native-pyramid path should not run for ViT uniform grids")

    model._forward_upsample_native_pyramid = _fail_native  # type: ignore[method-assign]
    logits = model(features, image=image, output_size=(224, 224))
    assert logits.shape == (2, 4, 224, 224)


def test_build_segmentation_head_unified_decoder_mode():
    legacy = build_segmentation_head(
        "feature_unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[64, 64, 64, 64],
        num_classes=3,
        head_config={"dropout": 0.1},
    )
    feature = build_segmentation_head(
        "unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[64, 64, 64, 64],
        num_classes=3,
        head_config={"decoder_mode": "feature", "dropout": 0.1},
    )
    pfmb = build_segmentation_head(
        "unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[64, 64, 64, 64],
        num_classes=3,
        head_config={"decoder_mode": "pfmb", "drop_rate": 0.1},
    )
    assert legacy.__class__.__name__ == "FeatureUNETR"
    assert feature.__class__.__name__ == "FeatureUNETR"
    assert pfmb.__class__.__name__ == "PfmbUNETR"


def test_forward_segmentation_head_routes_by_model_type():
    features = {
        "z1": torch.randn(1, 32, 8, 8),
        "z2": torch.randn(1, 32, 8, 8),
        "z3": torch.randn(1, 32, 8, 8),
        "z4": torch.randn(1, 32, 8, 8),
    }
    image = torch.randn(1, 3, 64, 64)
    masks = torch.zeros(1, 64, 64, dtype=torch.long)
    model = build_segmentation_head(
        "unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[32, 32, 32, 32],
        num_classes=2,
        head_config={"decoder_mode": "pfmb", "drop_rate": 0.0},
    )
    logits = forward_segmentation_head(model, features, masks, images=image)
    assert logits.shape == (1, 2, 64, 64)


def test_forward_segmentation_head_routes_wrapped_pfmb_unetr_z0():
    class ModuleWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    features = {
        "z1": torch.randn(1, 32, 8, 8),
        "z2": torch.randn(1, 32, 8, 8),
        "z3": torch.randn(1, 32, 8, 8),
        "z4": torch.randn(1, 32, 8, 8),
    }
    image = torch.randn(1, 3, 64, 64)
    masks = torch.zeros(1, 64, 64, dtype=torch.long)
    model = build_segmentation_head(
        "unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[32, 32, 32, 32],
        num_classes=2,
        head_config={"decoder_mode": "pfmb", "drop_rate": 0.0},
    )
    logits = forward_segmentation_head(ModuleWrapper(model), features, masks, images=image)
    assert logits.shape == (1, 2, 64, 64)


def test_forward_segmentation_head_routes_feature_unetr_z0():
    features = {
        "z1": torch.randn(1, 32, 8, 8),
        "z2": torch.randn(1, 32, 8, 8),
        "z3": torch.randn(1, 32, 8, 8),
        "z4": torch.randn(1, 32, 8, 8),
    }
    image = torch.randn(1, 3, 64, 64)
    masks = torch.zeros(1, 64, 64, dtype=torch.long)
    model = build_segmentation_head(
        "feature_unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[32, 32, 32, 32],
        num_classes=2,
        head_config={"use_z0": True, "dropout": 0.0},
    )
    logits = forward_segmentation_head(model, features, masks, images=image)
    assert logits.shape == (1, 2, 64, 64)
