"""MIPHEI-ViT inference wrapper (H-optimus-0 encoder + optional generator checkpoint).

The archived ``_orig_pyc/mipheivit_inference.pyc`` is gone; this matches the
API ``feature_extraction.layers`` and ``training.online_segmentation_train``
still call: ``wrapper.net.encoder.vit``, ``forward`` = generated mIF, and
``_load_checkpoint`` so frozen H-optimus-0 can skip the generator weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms

_TRUE = {"1", "true", "yes", "on"}


def _vendor_root() -> Path:
    from utils.paths import reference_model_path

    root = reference_model_path("mipheivit")
    if (root / "src" / "generators" / "mipheivit.py").is_file():
        return root
    raise ImportError(
        "MIPHEI-ViT checkout not found; clone it next to this repo as in the "
        f"README (expected {root / 'src' / 'generators' / 'mipheivit.py'})"
    )


def _ensure_vendor_on_path() -> Path:
    root = _vendor_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in _TRUE


def _load_checkpoint(net: nn.Module, ckpt_path) -> None:
    """Load MIPHEI generator + LoRA weights. Frozen H-optimus patches this out."""
    if ckpt_path is None or str(ckpt_path).strip() in {"", "None"}:
        print("MIPHEI-ViT: no generator checkpoint path; encoder-only weights", flush=True)
        return
    path = Path(str(ckpt_path))
    if not path.is_file():
        raise FileNotFoundError(f"MIPHEI generator checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(
        f"MIPHEI-ViT: loaded {path.name} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


class MIPHEIWrapper(nn.Module):
    """Expose ``.net`` so layer extraction can reach ``encoder.vit``."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        vit = self.net.encoder.vit
        feats = vit.forward_intermediates(
            x,
            indices=(-4, -3, -2, -1),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return [feat for feat in feats if feat is not None]


def get_model(
    device: torch.device,
    ckpt_path: Optional[str] = None,
    gpu_num: int = 1,
    img_size: int = 256,
    encoder_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    use_lora: Optional[bool] = None,
) -> nn.Module:
    del gpu_num
    _ensure_vendor_on_path()
    from src.generators.mipheivit import get_mipheivit

    size = int(os.environ.get("MIPHEI_IMG_SIZE", img_size) or img_size)
    name = encoder_name or os.environ.get("MIPHEI_ENCODER_NAME") or "hoptimus0"
    classes = num_classes
    if classes is None:
        classes = int(os.environ.get("MIPHEI_NUM_CLASSES", "16"))
    if use_lora is None:
        use_lora = _env_flag("MIPHEI_USE_LORA", True)

    net = get_mipheivit(str(name), int(size), int(classes), use_lora=bool(use_lora))
    if _env_flag("MIPHEI_SKIP_GENERATOR_CKPT", False):
        print(
            "H-optimus-0 encoder route: skipping MIPHEI generator/LoRA checkpoint",
            flush=True,
        )
    else:
        _load_checkpoint(net, ckpt_path)
    model = MIPHEIWrapper(net)
    return model.to(device).eval()


def get_trans(img_size: Optional[int] = None):
    size = img_size
    if size is None:
        raw = os.environ.get("MIPHEI_IMG_SIZE", "256")
        size = int(raw) if str(raw).strip() else 256
    size = int(size) if size else None
    ops = []
    if size:
        ops.append(transforms.Resize((size, size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(ops)
