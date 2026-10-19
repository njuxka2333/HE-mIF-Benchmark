"""UNI2-h benchmark wrapper following PFMB's timm integration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked
from torchvision import transforms


from utils.paths import UNI2_CHECKPOINT

UNI2_HF_ID = "MahmoodLab/UNI2-h"
UNI2_IMG_SIZE = 224
UNI2_PATCH_SIZE = 14
UNI2_REG_TOKENS = 8
UNI2_EMBED_DIM = 1536
UNI2_FILENAME = "pytorch_model.bin"
UNI2_LOCAL_CKPT = UNI2_CHECKPOINT


def _uni2_kwargs() -> dict:
    return {
        "img_size": UNI2_IMG_SIZE,
        "patch_size": UNI2_PATCH_SIZE,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": UNI2_EMBED_DIM,
        "mlp_ratio": 5.33334,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": UNI2_REG_TOKENS,
        "dynamic_img_size": True,
    }


class UNI2Wrapper(nn.Module):
    """Expose UNI2 spatial layers and final patch map for feature extraction."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_patches(x)

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net.forward_features(x)[:, UNI2_REG_TOKENS + 1 :, :]
        grid_h = x.shape[-2] // UNI2_PATCH_SIZE
        grid_w = x.shape[-1] // UNI2_PATCH_SIZE
        return tokens.reshape(x.shape[0], grid_h, grid_w, -1).permute(0, 3, 1, 2)

    def forward_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Global token before patch tokens (index UNI2_REG_TOKENS after register tokens)."""
        tokens = self._vit_model().forward_features(x)
        return tokens[:, UNI2_REG_TOKENS, :]

    def forward_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.net(x)
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("UNI2 features_only model must return a list/tuple of feature maps")
        return [out for out in outputs if out is not None]

    def _vit_model(self) -> nn.Module:
        vit = getattr(self.net, "model", None)
        if vit is not None:
            return vit
        return self.net

    def forward_layers_by_indices(self, x: torch.Tensor, indices: list[int]) -> list[torch.Tensor]:
        """ViT block maps for architecture-interface screening."""
        vit = self._vit_model()
        if not hasattr(vit, "forward_intermediates"):
            raise TypeError("UNI2 wrapper cannot expose indexed ViT intermediates")
        feats = vit.forward_intermediates(
            x,
            indices=list(indices),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return [feat for feat in feats if feat is not None]


def get_model(device: torch.device, gpu_num: int = 1) -> nn.Module:
    del gpu_num
    import timm

    try:
        if UNI2_LOCAL_CKPT.exists():
            ckpt_path = UNI2_LOCAL_CKPT
        else:
            from huggingface_hub import hf_hub_download

            from utils.hf_auth import hf_token

            ckpt_path = hf_hub_download(
                UNI2_HF_ID,
                filename=UNI2_FILENAME,
                token=hf_token(),
            )
        net = timm.create_model(
            "vit_giant_patch14_224",
            pretrained=False,
            features_only=True,
            out_indices=(-4, -3, -2, -1),
            **_uni2_kwargs(),
        )
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if state and not next(iter(state)).startswith("model."):
            state = {f"model.{key}": value for key, value in state.items()}
        load_result = net.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"missing UNI2 weights: {load_result.missing_keys}")
        return UNI2Wrapper(net).to(device).eval()
    except Exception as exc:
        print(f"[uni2] local cached load failed, falling back to timm hf_hub: {exc}", flush=True)

    net = timm.create_model(
        f"hf_hub:{UNI2_HF_ID}",
        pretrained=True,
        features_only=True,
        out_indices=(-4, -3, -2, -1),
        **_uni2_kwargs(),
    )
    return UNI2Wrapper(net).to(device).eval()


def get_trans(img_size: Optional[int] = UNI2_IMG_SIZE):
    ops = []
    if img_size is not None and img_size > 0:
        ops.append(transforms.Resize((img_size, img_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(ops)
