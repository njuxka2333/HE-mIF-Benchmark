"""Decode frozen HE→mIF generator output (shared by VS scoring and UNet training)."""

from __future__ import annotations

import torch

from models.mif_output import (
    FrozenMifGenerator,
    decode_generated_mif,
    resolve_mif_channels,
    resolve_mif_layer,
)


def test_resolve_mif_layer_defaults() -> None:
    assert resolve_mif_layer("rosie") == "mif0"
    assert resolve_mif_layer("hemit") == "mif"
    assert resolve_mif_layer("gigatime", "custom") == "custom"
    assert resolve_mif_channels("hemit") == 3
    assert resolve_mif_channels("rosie") == 50


def test_decode_dict_layer_is_passthrough() -> None:
    stack = torch.randn(2, 3, 8, 8)
    out = decode_generated_mif({"mif": stack, "z4": torch.zeros(1)}, model_name="hemit", layer="mif")
    assert out.data_ptr() == stack.data_ptr()


def test_decode_gigatime_tensor_applies_sigmoid() -> None:
    logits = torch.tensor([[[[10.0, -10.0]]]])
    out = decode_generated_mif(logits, model_name="gigatime", layer="mif", input_hw=(1, 2))
    assert float(out[0, 0, 0, 0]) > 0.99
    assert float(out[0, 0, 0, 1]) < 0.01


def test_decode_histoplexer_list_uses_last_and_crops() -> None:
    coarse = torch.zeros(1, 2, 4, 4)
    fine = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).expand(1, 2, 4, 4).clone()
    out = decode_generated_mif(
        [coarse, fine], model_name="histoplexer", layer="mif", input_hw=(8, 8)
    )
    assert tuple(out.shape) == (1, 2, 2, 2)


def test_frozen_generator_forces_eval() -> None:
    class _Stub(torch.nn.Module):
        def forward(self, x):
            return x

    inner = _Stub()
    inner.train()
    gen = FrozenMifGenerator(inner, model_name="hemit", layer="mif")
    gen.train(True)
    assert not gen.training
    assert not inner.training
    rgb = torch.randn(1, 3, 4, 4)
    out = gen(rgb)
    assert tuple(out.shape) == (1, 3, 4, 4)
