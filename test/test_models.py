"""Model loading and forward pass for all four encoders."""

from __future__ import annotations

import pytest
import torch

from conftest import MODELS, ensure_model_weights, local_checkpoint_path


@pytest.mark.requires_weights
@pytest.mark.parametrize("model", MODELS)
def test_model_forward(model, device):
    from utils.experiment_config import ExtractionTask
    from models.registry import load_benchmark_registry
    from feature_extraction.feature_extractor import build_feature_extractor
    from feature_extraction.layers import MODEL_LAYER_NAMES
    from models.loader import load_model_from_config

    ensure_model_weights(model)
    load_benchmark_registry.cache_clear()

    try:
        base = load_model_from_config(model, device, gpu_num=1)
    except ImportError as exc:
        pytest.skip(f"Missing dependency for {model}: {exc}")
    except OSError as exc:
        pytest.skip(f"Cannot load weights for {model}: {exc}")
    except Exception as exc:
        name = type(exc).__name__
        if name in {"GatedRepoError", "HTTPStatusError", "ProxyError"} or "401" in str(exc):
            pytest.skip(f"Cannot download weights for {model}: {exc}")
        raise

    layer = MODEL_LAYER_NAMES[model][-1]
    task = ExtractionTask(
        head="linear_probe",
        model=model,
        target_features=[layer],
        use_gap=True,
    )
    extractor = build_feature_extractor(base, task).to(device).eval()

    sizes = {
        "rosie": (128, 128),
        "hemit": (1024, 1024),
        "gigatime": (256, 256),
        "mipheivit": (256, 256),
    }
    h, w = sizes[model]
    x = torch.randn(1, 3, h, w, device=device)
    with torch.no_grad():
        out = extractor(x)
    assert out.shape[0] == 1
    assert out.shape[-1] > 0


@pytest.mark.requires_weights
@pytest.mark.parametrize("model", MODELS)
def test_checkpoint_resolves(model):
    ensure_model_weights(model)
    ckpt = local_checkpoint_path(model)
    assert ckpt is not None and ckpt.is_file(), model
