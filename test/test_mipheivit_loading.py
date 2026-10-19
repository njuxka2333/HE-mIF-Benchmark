"""MIPHEI-ViT encoder / checkpoint alignment tests."""

from __future__ import annotations

import pytest

from models.registry import get_model_config, load_benchmark_registry


def test_mipheivit_benchmark_config_uses_hoptimus0():
    load_benchmark_registry.cache_clear()
    cfg = get_model_config("mipheivit")
    assert cfg.extra.get("encoder_name") == "hoptimus0"
    assert cfg.extra.get("num_classes") == 16
    assert cfg.output.channels == 16
    assert cfg.extra.get("foundation_hub", {}).get("repo_id") == "bioptimus/H-optimus-0"


@pytest.mark.requires_weights
def test_mipheivit_safetensors_aligns_with_hoptimus_encoder(device):
    """Release model.safetensors should load cleanly on a 40-block H-Optimus backbone."""
    from conftest import ensure_model_weights, local_checkpoint_path
    from models.mipheivit_inference import get_model

    ensure_model_weights("mipheivit")
    ckpt = local_checkpoint_path("mipheivit")
    assert ckpt is not None

    try:
        model = get_model(
            device,
            str(ckpt),
            img_size=256,
            encoder_name="hoptimus0",
            num_classes=16,
            use_lora=True,
        )
    except ImportError as exc:
        pytest.skip(f"Missing dependency: {exc}")
    except OSError as exc:
        pytest.skip(f"Cannot load weights: {exc}")
    except Exception as exc:
        name = type(exc).__name__
        if name in {"GatedRepoError", "HTTPStatusError", "ProxyError"} or "401" in str(exc):
            pytest.skip(f"Cannot download H-Optimus-0 weights: {exc}")
        raise

    depth = len(model.net.encoder.vit.blocks)
    assert depth == 40, f"expected H-Optimus 40-block ViT, got depth={depth}"
