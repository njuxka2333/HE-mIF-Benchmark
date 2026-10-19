"""Shared encoder loading for TLS / PathoCell example-tile rendering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from feature_extraction.extractor import build_feature_extractor
from heads import FeatureUNETR, SegmentationFeaturePipeline
from models import get_custom_transformer, get_model
from models.registry import apply_model_env, get_model_config
from utils.paths import checkpoint_root, reference_model_path

CKPT_SEARCH = checkpoint_root()


def inner_unetr(module: torch.nn.Module) -> torch.nn.Module:
    routed = getattr(module, "module", module)
    if isinstance(routed, SegmentationFeaturePipeline):
        return routed.decoder
    return routed


def state_has_z0(state: dict) -> bool:
    return any(part == "decoder0" for key in state for part in str(key).split("."))


def bind_no_z0(decoder: torch.nn.Module) -> None:
    """HEMIT (and any head trained with use_z0=false) has no decoder0/z0_merge."""
    unetr = inner_unetr(decoder)
    if not isinstance(unetr, FeatureUNETR):
        return

    def _forward(features, output_size=None, image=None):  # noqa: ARG001
        adapted = [unetr.adapters[name](features[name]) for name in unetr.feature_layers]
        x = adapted[-1]
        for skip, block in zip(reversed(adapted[:-1]), unetr.decoder_blocks):
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = block(torch.cat([skip, x], dim=1))
        if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
            x = F.interpolate(
                x, size=tuple(output_size), mode="bilinear", align_corners=False
            )
        return unetr.classifier(x)

    unetr.forward = _forward


def _imagenet_trans(size: int | None):
    ops = []
    if size:
        ops.append(transforms.Resize((int(size), int(size))))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(ops)


def _pix2pix_trans(size: int | None):
    ops = []
    if size:
        ops.append(transforms.Resize((int(size), int(size))))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


class _MipheiLayerWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_layers(self, x: torch.Tensor):
        vit = self.net.encoder.vit
        feats = vit.forward_intermediates(
            x,
            indices=(-4, -3, -2, -1),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return [feat for feat in feats if feat is not None]


class _HemitLayerWrapper(torch.nn.Module):
    def __init__(self, net_g: torch.nn.Module):
        super().__init__()
        self.net_g = net_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_g(x)


class _HistoplexerLayerWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def forward_layers(self, x: torch.Tensor) -> dict:
        feats = self.net(x, encode_only=True)
        out = {}
        for i, feat in enumerate(feats):
            out[f"encoder_{i}"] = feat
            out[f"z{i}"] = feat
        if feats:
            out["bottleneck"] = feats[-1]
        return out


def _load_miphei(device: torch.device, img_size: int):
    root = reference_model_path("mipheivit")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from safetensors.torch import load_file
    from src.generators.mipheivit import get_mipheivit

    net = get_mipheivit("hoptimus0", int(img_size), 16, use_lora=True)
    ckpt = CKPT_SEARCH / "mipheivit.safetensors"
    sd = load_file(str(ckpt))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(
        f"    mipheivit ckpt missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return _MipheiLayerWrapper(net).to(device).eval(), _imagenet_trans(img_size)


def _load_hemit(device: torch.device, img_size: int):
    import importlib.util

    path = reference_model_path("hemit") / "models" / "networks.py"
    spec = importlib.util.spec_from_file_location("_hemit_pix2pix_networks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HEMIT networks from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.ResnetGeneratorSwinT(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=6,
        img_size=1024,
        patch_size=32,
        embed_dim=96,
        window_size=32,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
    )
    sd = torch.load(CKPT_SEARCH / "hemit_v1.pth", map_location="cpu", weights_only=False)
    current = net.state_dict()
    filtered = {
        key: value
        for key, value in sd.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = net.load_state_dict(filtered, strict=False)
    skipped = [key for key in sd if key not in filtered]
    print(
        f"    hemit ckpt matched={len(filtered)} skipped={len(skipped)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return _HemitLayerWrapper(net).to(device).eval(), _pix2pix_trans(img_size)


def _load_histoplexer(device: torch.device, img_size: int):
    from models.mif_feed import purge_histoplexer_src_shadow

    purge_histoplexer_src_shadow()
    root = reference_model_path("histoplexer")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models.generator import unet_translator

    net = unet_translator(
        input_nc=3,
        output_nc=11,
        use_high_res=True,
        use_multiscale=True,
        ngf=32,
        depth=6,
        encoder_padding=1,
        decoder_padding=1,
        extra_feature_size=0,
        device=device,
    )
    blob = torch.load(CKPT_SEARCH / "histoplexer.pt", map_location="cpu", weights_only=False)
    sd = blob.get("trans_ema_state_dict") or blob["trans_state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(
        f"    histoplexer ckpt missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    ops = [transforms.Resize((int(img_size), int(img_size)))] if img_size else []
    ops.append(transforms.ToTensor())
    return _HistoplexerLayerWrapper(net).to(device).eval(), transforms.Compose(ops)


def build_encoder(exp, device: torch.device):
    """Native wrappers; MIPHEI/HEMIT/HistoPlexer load from vendor checkouts."""
    cfg_path = str(exp.benchmark_config_path) if exp.benchmark_config_path else None
    try:
        apply_model_env(get_model_config(exp.model, cfg_path))
    except KeyError:
        pass
    name = str(exp.model).lower()
    size = int(exp.target_patch_size)
    if name == "mipheivit":
        os.environ["MIPHEI_IMG_SIZE"] = str(size)
        base, transform = _load_miphei(device, size)
    elif name in {"hemit", "hemmit"}:
        os.environ["HEMIT_TARGET_SIZE"] = str(size)
        base, transform = _load_hemit(device, size)
    elif name == "histoplexer":
        os.environ["HISTOPLEXER_TARGET_SIZE"] = str(size)
        base, transform = _load_histoplexer(device, size)
    else:
        gpu_num = torch.cuda.device_count() if device.type == "cuda" else 1
        base = get_model(
            name,
            device,
            gpu_num,
            use_benchmark_config=False,
            config_path=cfg_path,
        )
        transform = get_custom_transformer(
            name,
            use_benchmark_config=False,
            config_path=cfg_path,
        )
    encoder = build_feature_extractor(base, exp.extraction_task()).to(device).eval()
    return encoder, transform


def load_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint payload at {path}")
    if "model_state_dict" in state:
        return state
    return {"model_state_dict": state}
