"""Frozen HE→mIF generator output.

Virtual staining and output-layer segmentation share this: load the foundation
model, run it, return the generated marker stack. Scoring vs training happens
after this tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

# Marker stack widths (histoplexer = generated_markers).
MODEL_MIF_CHANNELS: Mapping[str, int] = {
    "rosie": 50,
    "gigatime": 23,
    "mipheivit": 16,
    "histoplexer": 11,
    "hemit": 3,
}

MODEL_MIF_LAYER: Mapping[str, str] = {
    "rosie": "mif0",
    "gigatime": "mif",
    "mipheivit": "mif",
    "histoplexer": "mif",
    "hemit": "mif",
}


def resolve_mif_layer(model_name: str, override: str | None = None) -> str:
    if override:
        return str(override)
    key = str(model_name).strip().lower()
    if key not in MODEL_MIF_LAYER:
        raise KeyError(
            f"unknown model {model_name!r} for mIF layer; "
            f"known={sorted(MODEL_MIF_LAYER)} or pass mif_layer="
        )
    return MODEL_MIF_LAYER[key]


def resolve_mif_channels(model_name: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    key = str(model_name).strip().lower()
    if key not in MODEL_MIF_CHANNELS:
        raise KeyError(
            f"unknown model {model_name!r} for mIF channels; "
            f"known={sorted(MODEL_MIF_CHANNELS)} or pass in_channels="
        )
    return int(MODEL_MIF_CHANNELS[key])


def decode_generated_mif(
    raw: Any,
    *,
    model_name: str,
    layer: str,
    input_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Turn a foundation-model forward into ``[B, C, H, W]`` markers.

    * Feature-extraction wrappers return a dict keyed by ``layer`` (already
      decoded). Those tensors are returned as-is.
    * Inference wrappers (virtual staining) return a tensor or a list; GigaTIME
      logits are sigmoided and HistoPlexer is center-cropped to ``input//4``.
    """
    name = str(model_name).strip().lower()
    if isinstance(raw, Mapping):
        if layer not in raw:
            raise KeyError(
                f"generator {name!r} missing layer {layer!r}; keys={list(raw)}"
            )
        tensor = raw[layer]
    elif isinstance(raw, (list, tuple)):
        tensor = raw[-1] if name == "histoplexer" else raw[0]
    else:
        tensor = raw
    if not torch.is_tensor(tensor):
        raise TypeError(f"generated mIF must be a tensor, got {type(tensor)!r}")
    tensor = tensor.float()
    if isinstance(raw, Mapping):
        return tensor
    if name == "gigatime":
        tensor = torch.sigmoid(tensor)
    if name == "histoplexer":
        in_px = int(input_hw[-1] if input_hw is not None else tensor.shape[-1] * 4)
        out_px = max(1, in_px // 4)
        _, _, height, width = tensor.shape
        y0 = max(0, (height - out_px) // 2)
        x0 = max(0, (width - out_px) // 2)
        tensor = tensor[:, :, y0 : y0 + out_px, x0 : x0 + out_px]
    return tensor


class FrozenMifGenerator(nn.Module):
    """Wrap a HE→mIF foundation model so it stays in eval and has no grad."""

    def __init__(self, wrapper: nn.Module, *, model_name: str, layer: str) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.model_name = str(model_name).strip().lower()
        self.layer = str(layer)
        for param in self.wrapper.parameters():
            param.requires_grad_(False)
        self.wrapper.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenMifGenerator:
        del mode
        self.wrapper.eval()
        return super().train(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.wrapper(rgb)
        return decode_generated_mif(
            out,
            model_name=self.model_name,
            layer=self.layer,
            input_hw=(int(rgb.shape[-2]), int(rgb.shape[-1])),
        )


def load_frozen_mif_generator(
    model_name: str,
    device: torch.device,
    *,
    layer: str | None = None,
    gpu_num: int = 1,
    use_benchmark_config: bool = False,
    config_path: str | Path | None = None,
) -> tuple[FrozenMifGenerator, Any]:
    """Load the same inference wrapper virtual staining uses, frozen."""
    from models import get_custom_transformer, get_model

    kwargs: dict[str, Any] = {
        "device": device,
        "gpu_num": int(gpu_num),
        "use_benchmark_config": bool(use_benchmark_config),
    }
    if config_path is not None:
        kwargs["config_path"] = str(config_path)
    wrapper = get_model(model_name, **kwargs)
    wrapper.eval()
    transform = get_custom_transformer(
        model_name,
        use_benchmark_config=bool(use_benchmark_config),
        config_path=None if config_path is None else str(config_path),
    )
    gen = FrozenMifGenerator(
        wrapper=wrapper,
        model_name=model_name,
        layer=resolve_mif_layer(model_name, layer),
    )
    gen.to(device)
    return gen, transform


def _load_mipheivit_vstain_generator(device: torch.device) -> tuple[FrozenMifGenerator, Any]:
    """Paper-native MIPHEI-ViT wrapper (H-optimus-0 + LoRA), same as the Table-4 scorer."""
    from data.marker_registry import MODEL_GEOMETRY
    from models.registry import get_model_config
    from utils.checkpoints import resolve_model_checkpoint
    from models.mipheivit_inference import get_model as get_miphei_model
    from models.mipheivit_inference import get_trans

    tile_px = int(MODEL_GEOMETRY["mipheivit"]["input_size"])
    cfg = get_model_config("mipheivit")
    ckpt = resolve_model_checkpoint(cfg)
    wrapper = get_miphei_model(
        device,
        str(ckpt),
        img_size=tile_px,
        encoder_name="hoptimus0",
        num_classes=16,
        use_lora=True,
    )
    generator = FrozenMifGenerator(
        wrapper, model_name="mipheivit", layer=resolve_mif_layer("mipheivit")
    )
    generator.to(device)
    generator.eval()
    return generator, get_trans(tile_px)


def load_vstain_generator(
    model_name: str,
    device: torch.device,
    *,
    gpu_num: int = 1,
) -> tuple[FrozenMifGenerator, Any]:
    """Load the generator virtual-staining scoring uses.

    MIPHEI-ViT keeps its H-optimus-0 + LoRA inference wrapper. Everyone else
    uses ``load_frozen_mif_generator`` (same object output-layer training uses).
    """
    name = str(model_name).strip().lower()
    if name == "histoplexer":
        from models.mif_feed import purge_histoplexer_src_shadow

        purge_histoplexer_src_shadow()
    if name == "mipheivit":
        return _load_mipheivit_vstain_generator(device)
    return load_frozen_mif_generator(name, device, gpu_num=gpu_num)


__all__ = [
    "MODEL_MIF_CHANNELS",
    "MODEL_MIF_LAYER",
    "FrozenMifGenerator",
    "decode_generated_mif",
    "load_frozen_mif_generator",
    "load_vstain_generator",
    "resolve_mif_channels",
    "resolve_mif_layer",
]
