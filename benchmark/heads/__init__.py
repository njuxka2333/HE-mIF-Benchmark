
"""Trainable heads for downstream tasks on frozen extracted features."""
from __future__ import annotations
from typing import Any, Mapping, OrderedDict as OrderedDictType
import torch
import torch.nn as nn
import torch.nn.functional as F
from heads.unetr import PfmbUNETR
from heads.bilinear import BilinearViTFeaturePyramidBuilder
from heads.vit_pyramid import ViTFeaturePyramidBuilder
UNETR_HEAD_NAME = 'unetr'
DECODER_MODES = frozenset({'feature', 'pfmb'})
CELLVIT_PYRAMID_PROCESSOR_NAMES = frozenset({'cellvit', 'cellvit_pyramid'})
BILINEAR_PYRAMID_PROCESSOR_NAMES = frozenset({'bilinear_pyramid', 'vit_bilinear', 'bilinear'})
def build_feature_processor(processor_name: str, *, feature_layers: list[str], in_channels: list[int], processor_config: Mapping[str, Any] | None=None, dropout: float=0.0) -> nn.Module:
    """Instantiate a decoder-independent ViT/CNN feature preprocessor."""
    name = str(processor_name).strip().lower()
    config = dict(processor_config or {})
    config.pop('name', None)
    if name in CELLVIT_PYRAMID_PROCESSOR_NAMES:
        config.setdefault('drop_rate', dropout)
        return ViTFeaturePyramidBuilder(feature_layers=feature_layers, in_channels=in_channels, **config)
    else:
        if name in BILINEAR_PYRAMID_PROCESSOR_NAMES:
            return BilinearViTFeaturePyramidBuilder(feature_layers=feature_layers, in_channels=in_channels, **config)
        else:
            supported = sorted(CELLVIT_PYRAMID_PROCESSOR_NAMES | BILINEAR_PYRAMID_PROCESSOR_NAMES)
            raise ValueError(f'feature_processor.name must be one of {supported}, got {processor_name!r}')
def normalize_head_name(head_name: str) -> str:
    """Map legacy head aliases to the canonical UNETR registry name."""
    if head_name in {UNETR_HEAD_NAME, 'feature_unetr'}:
        return UNETR_HEAD_NAME
    else:
        return head_name
def _resolve_base_decoder_mode(head_name: str, head_config: Mapping[str, Any] | None) -> str:
    """\n    Resolve UNETR decoder recipe from config.\n\n    ``decoder_mode: feature`` — classic UNETR skip fusion (TLS / region seg).\n    ``decoder_mode: pfmb`` — CellViT/PFMB recipe (PathoCell cell seg).\n\n    Legacy ``evaluation.head`` values still infer the mode when ``decoder_mode``\n    is omitted: ``feature_unetr`` -> ``feature``, ``unetr`` -> ``pfmb``.\n    """
    config = dict(head_config or {})
    if 'decoder_mode' in config:
        mode = str(config['decoder_mode']).strip().lower()
        if mode not in DECODER_MODES:
            raise ValueError(f'head.decoder_mode must be one of {sorted(DECODER_MODES)}, got {mode!r}')
        else:
            return mode
    else:
        legacy = normalize_head_name(head_name)
        if legacy != UNETR_HEAD_NAME:
            raise ValueError(f'decoder_mode is required for head={head_name!r}; use head: {UNETR_HEAD_NAME} with decoder_mode in the head section.')
        else:
            if head_name == 'feature_unetr':
                return 'feature'
            else:
                return 'pfmb'
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.0):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
class FeatureUNETR(nn.Module):
    """\n    UNETR-style decoder for pre-extracted multi-scale feature maps.\n\n    Unlike the reference PFMB ``UnetR``, this module does not own or run an encoder.\n    It consumes an ordered dict of frozen spatial features and learns only adapters\n    plus the decoder.\n\n    An additional RGB branch (z0) is always fused after the multi-scale skip\n    path. Encoder skips remain on ``feature_layers`` (typically z1-z4); z0 is\n    not an encoder feature key.\n    """
    def __init__(self, feature_layers: list[str], in_channels: list[int], num_classes: int, adapter_channels: list[int] | None=None, dropout: float=0.0, z0_channels: int=64):
        super().__init__()
        if len(feature_layers) != len(in_channels):
            raise ValueError('feature_layers and in_channels must have the same length')
        else:
            if len(feature_layers) < 2:
                raise ValueError('FeatureUNETR needs at least two feature levels')
            else:
                self.feature_layers = list(feature_layers)
                self.use_z0 = True
                self.z0_channels = int(z0_channels)
                adapter_in_channels = [int(ch) for ch in in_channels]
                self.adapter_in_channels = adapter_in_channels
                if adapter_channels is None:
                    defaults = [64, 128, 256, 512, 512]
                    adapter_channels = defaults[:len(feature_layers)]
                    if len(adapter_channels) < len(feature_layers):
                        adapter_channels.extend([defaults[(-1)]] * (len(feature_layers) - len(defaults)))
                if len(adapter_channels) != len(feature_layers):
                    raise ValueError('adapter_channels must match feature_layers length')
                else:
                    self.adapters = nn.ModuleDict({name: ConvBlock(in_ch, out_ch, dropout=dropout) for name, in_ch, out_ch in zip(feature_layers, self.adapter_in_channels, adapter_channels)})
                    self.decoder_blocks = nn.ModuleList()
                    current_channels = adapter_channels[(-1)]
                    for skip_channels in reversed(adapter_channels[:(-1)]):
                        self.decoder_blocks.append(ConvBlock(current_channels + skip_channels, skip_channels, dropout=dropout))
                        current_channels = skip_channels
                    self.decoder0 = nn.Sequential(ConvBlock(3, 32, dropout=dropout), ConvBlock(32, self.z0_channels, dropout=dropout))
                    self.z0_merge = ConvBlock(current_channels + self.z0_channels, current_channels, dropout=dropout)
                    self.classifier = nn.Conv2d(current_channels, num_classes, kernel_size=1)
    def forward(self, features: OrderedDictType[str, torch.Tensor] | dict[str, torch.Tensor], output_size: tuple[int, int] | None=None, image: torch.Tensor | None=None) -> torch.Tensor:
        adapted = [self.adapters[name](features[name]) for name in self.feature_layers]
        x = adapted[(-1)]
        for skip, block in zip(reversed(adapted[:(-1)]), self.decoder_blocks):
            x = F.interpolate(x, size=skip.shape[(-2):], mode='bilinear', align_corners=False)
            x = block(torch.cat([skip, x], dim=1))
        if image is None:
            raise ValueError('FeatureUNETR requires an RGB image tensor for decoder-side z0')
        else:
            z0 = self.decoder0(image)
            target_hw = output_size or z0.shape[(-2):]
            if x.shape[(-2):] != target_hw:
                x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
            if z0.shape[(-2):] != target_hw:
                z0 = F.interpolate(z0, size=target_hw, mode='bilinear', align_corners=False)
            x = self.z0_merge(torch.cat([z0, x], dim=1))
            return self.classifier(x)
class SegmentationFeaturePipeline(nn.Module):
    """Compose decoder-independent feature preprocessing with a task decoder."""
    def __init__(self, feature_processor: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.feature_processor = feature_processor
        self.decoder = decoder
        self.feature_layers = list(feature_processor.feature_layers)
        self.use_z0 = bool(getattr(decoder, 'use_z0', False))
    def forward(self, features: Mapping[str, torch.Tensor], output_size: tuple[int, int] | None=None, image: torch.Tensor | None=None) -> torch.Tensor:
        pyramid = self.feature_processor(features)
        return self.decoder(pyramid, output_size=output_size, image=image)
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool=True, assign: bool=False):
        """Accept checkpoints from both decoder-coupled predecessor layouts."""
        keys = list(state_dict)
        if any((key.startswith('decoder.') for key in keys)):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        else:
            migrated = {}
            if isinstance(self.decoder, PfmbUNETR) and any((key.startswith('decoder1.') for key in keys)):
                skip_prefixes = {'decoder1.': f'feature_processor.skip_processors.{self.feature_layers[0]}.', 'decoder2.': f'feature_processor.skip_processors.{self.feature_layers[1]}.', 'decoder3.': f'feature_processor.skip_processors.{self.feature_layers[2]}.'}
                for key, value in state_dict.items():
                    target = next((replacement + key[len(prefix):] for prefix, replacement in skip_prefixes.items() if key.startswith(prefix)), None)
                    if target is not None:
                        migrated[target] = value
                    else:
                        if key.startswith(('skip_projections.', 'skip_adapter')):
                            continue
                        else:
                            migrated[f'decoder.{key}'] = value
            else:
                for key, value in state_dict.items():
                    if key.startswith('feature_processor.'):
                        migrated[key] = value
                    else:
                        migrated[f'decoder.{key}'] = value
            return super().load_state_dict(migrated, strict=strict, assign=assign)
class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_c(self.attention_a(x) * self.attention_b(x))
        return (scores, x)
class AttentionMILBackbone(nn.Module):
    """Shared gated-attention MIL backbone with an MCAT-AMIL-style rho projection."""
    def __init__(self, input_dim: int, hidden_dim: int=512, embed_dim: int=256, dropout: float=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(hidden_dim, embed_dim, dropout)
        self.rho = nn.Sequential(nn.Linear(hidden_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.last_attention = None
        self.last_attention_logits = None
    def forward(self, x_path: torch.Tensor) -> torch.Tensor:
        h = self.fc(x_path)
        attention, h = self.attention(h)
        attention = torch.transpose(attention, 1, 0)
        self.last_attention_logits = attention
        weights = F.softmax(attention, dim=1)
        self.last_attention = weights
        h = torch.mm(weights, h)
        return self.rho(h).squeeze(0)
class SurvivalAttentionMIL(nn.Module):
    """MCAT-style WSI-only attention MIL survival head."""
    def __init__(self, input_dim: int, hidden_dim: int=512, embed_dim: int=256, dropout: float=0.25, n_classes: int=4):
        super().__init__()
        self.backbone = AttentionMILBackbone(input_dim, hidden_dim, embed_dim, dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)
    def forward(self, x_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x_path)
        logits = self.classifier(h).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        prediction = torch.topk(logits, 1, dim=1)[1]
        return (hazards, survival, prediction)
class ClassificationAttentionMIL(nn.Module):
    """Classification wrapper around the shared patient-level MIL backbone."""
    def __init__(self, input_dim: int, hidden_dim: int=512, embed_dim: int=256, dropout: float=0.25, n_classes: int=2):
        super().__init__()
        self.backbone = AttentionMILBackbone(input_dim, hidden_dim, embed_dim, dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)
    def forward(self, x_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.classifier(self.backbone(x_path)).unsqueeze(0)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.topk(logits, 1, dim=1)[1]
        return (logits, probabilities, prediction)
class PoolingMILClassifier(nn.Module):
    """Diagnostic patient classifier using fixed mean or max instance pooling."""
    def __init__(self, input_dim: int, n_classes: int=2, *, pooling: str, hidden_dim: int | None=None, dropout: float=0.0):
        super().__init__()
        if pooling not in {'mean', 'max'}:
            raise ValueError(f'pooling must be \'mean\' or \'max\', got {pooling!r}')
        else:
            self.pooling = pooling
            if hidden_dim is None:
                self.classifier = nn.Linear(input_dim, n_classes)
            else:
                self.classifier = nn.Sequential(nn.Linear(input_dim, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), n_classes))
    def forward(self, x_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_path.ndim != 2:
            raise ValueError(f'PoolingMILClassifier expects [N,D], got {tuple(x_path.shape)}')
        else:
            pooled = x_path.mean(dim=0) if self.pooling == 'mean' else x_path.max(dim=0).values
            logits = self.classifier(pooled).unsqueeze(0)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.topk(logits, 1, dim=1)[1]
            return (logits, probabilities, prediction)
class MeanPoolLinearMIL(PoolingMILClassifier):
    def __init__(self, input_dim: int, n_classes: int=2):
        super().__init__(input_dim, n_classes, pooling='mean')
class MeanPoolMLPMIL(PoolingMILClassifier):
    pass
    def __init__(self, input_dim: int, n_classes: int=2, hidden_dim: int=64, dropout: float=0.25):
        super().__init__(input_dim, n_classes, pooling='mean', hidden_dim=hidden_dim, dropout=dropout)
class MaxPoolLinearMIL(PoolingMILClassifier):
    def __init__(self, input_dim: int, n_classes: int=2):
        super().__init__(input_dim, n_classes, pooling='max')
class LinearProbeHead(nn.Module):
    """UNI-style linear classifier over frozen patch embeddings."""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(int(input_dim), int(n_classes), bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f'LinearProbeHead expects [N, D] features, got {tuple(features.shape)}')
        else:
            return self.classifier(features)
def _build_base_segmentation_head(head_name: str, *, feature_layers: list[str], in_channels: list[int], num_classes: int, head_config: Mapping[str, Any] | None=None) -> nn.Module:
    head_config = dict(head_config or {})
    canonical = normalize_head_name(head_name)
    if canonical == UNETR_HEAD_NAME:
        mode = resolve_decoder_mode(head_name, head_config)
        head_config = dict(head_config)
        head_config.pop('feature_layers', None)
        head_config.pop('decoder_mode', None)
        head_config.pop('pyramid_routing', None)
        head_config.pop('use_z0', None)
        processor_config = head_config.pop('feature_processor', None)
        dropout = float(head_config.pop('dropout', head_config.pop('drop_rate', 0.0)))
        feature_processor = None
        decoder_in_channels = [int(channel) for channel in in_channels]
        if processor_config:
            processor_config = dict(processor_config)
            processor_name = str(processor_config.pop('name', 'cellvit_pyramid')).strip().lower()
            feature_processor = build_feature_processor(processor_name, feature_layers=feature_layers, in_channels=decoder_in_channels, processor_config=processor_config, dropout=dropout)
            decoder_in_channels = list(feature_processor.output_channels)
        if mode == 'feature':
            decoder = FeatureUNETR(feature_layers=feature_layers, in_channels=decoder_in_channels, num_classes=num_classes, adapter_channels=head_config.pop('adapter_channels', None), dropout=dropout, z0_channels=int(head_config.pop('z0_channels', 64)))
        else:
            pfmb_embed_dim = head_config.pop('embed_dim', None)
            if pfmb_embed_dim is None and feature_processor is not None:
                    pfmb_embed_dim = feature_processor.embed_dim
            decoder = PfmbUNETR(feature_layers=feature_layers, in_channels=decoder_in_channels, num_classes=num_classes, embed_dim=pfmb_embed_dim, drop_rate=dropout, preprocessed_vit=feature_processor is not None)
        if feature_processor is not None:
            return SegmentationFeaturePipeline(feature_processor, decoder)
        else:
            return decoder
    else:
        head_cls = HEAD_REGISTRY[head_name]
        return head_cls(feature_layers=feature_layers, in_channels=in_channels, num_classes=num_classes, **head_config)
def _forward_base_segmentation_head(model: nn.Module, features: Mapping[str, torch.Tensor], masks: torch.Tensor, *, images: torch.Tensor | None=None) -> torch.Tensor:
    output_size = tuple(masks.shape[(-2):])
    routed_model = getattr(model, 'module', model)
    if isinstance(routed_model, SegmentationFeaturePipeline):
        return model(features, output_size=output_size, image=images)
    else:
        if isinstance(routed_model, PfmbUNETR):
            return model(features, image=images, output_size=output_size)
        else:
            if isinstance(routed_model, FeatureUNETR):
                return model(features, output_size=output_size, image=images)
            else:
                return model(features, output_size=output_size)
HEAD_REGISTRY = {UNETR_HEAD_NAME: FeatureUNETR, 'feature_unetr': FeatureUNETR, 'survival_attention_mil': SurvivalAttentionMIL, 'classification_attention_mil': ClassificationAttentionMIL, 'mean_pool_linear_mil': MeanPoolLinearMIL, 'mean_pool_mlp_mil': MeanPoolMLPMIL, 'max_pool_linear_mil': MaxPoolLinearMIL, 'linear_probe': LinearProbeHead}

_ORIGINAL_RESOLVE_DECODER_MODE = _resolve_base_decoder_mode
_ORIGINAL_BUILD_SEGMENTATION_HEAD = _build_base_segmentation_head
_ORIGINAL_FORWARD_SEGMENTATION_HEAD = _forward_base_segmentation_head
DECODER_MODES = frozenset(set(DECODER_MODES) | {"cellseg", "mif_unet", "mif_feature"})


def resolve_decoder_mode(head_name: str, config: dict | None = None) -> str:
    config = dict(config or {})
    mode = config.get("decoder_mode")
    if mode is not None:
        key = str(mode).lower()
        if key in {"cellseg", "mif_unet", "mif_feature"}:
            return key
    return _ORIGINAL_RESOLVE_DECODER_MODE(head_name, config)


def build_segmentation_head(
    head_name: str,
    *,
    feature_layers,
    in_channels,
    num_classes: int,
    head_config=None,
    **kwargs,
):
    head_config = dict(head_config or {})
    mode = resolve_decoder_mode(head_name, head_config)
    if mode == "cellseg":
        from heads.cellseg import CellsegUNETR

        layers = feature_layers or head_config.get("feature_layers")
        if layers is None:
            raise ValueError("cellseg decoder requires feature_layers")
        decoder_channels = head_config.get("decoder_channels", [128, 256, 512, 512])
        drop_rate = float(head_config.get("dropout", head_config.get("drop_rate", 0.0)))
        return CellsegUNETR(
            feature_layers=list(layers),
            in_channels=[int(ch) for ch in in_channels],
            num_classes=int(num_classes),
            decoder_channels=[int(ch) for ch in decoder_channels],
            drop_rate=drop_rate,
        )
    if mode == "mif_unet":
        from heads.mif_unet import build_mif_unet

        cfg = dict(head_config)
        cfg["num_classes"] = int(num_classes)
        if "in_channels" not in cfg:
            cfg["in_channels"] = _resolve_scalar_in_channels(in_channels, head_config)
        if feature_layers is not None and "feature_layers" not in cfg:
            cfg["feature_layers"] = list(feature_layers)
        return build_mif_unet(cfg)
    if mode == "mif_feature":
        from heads.mif_feature import build_mif_feature_unetr

        cfg = dict(head_config)
        cfg["num_classes"] = int(num_classes)
        if "in_channels" not in cfg:
            cfg["in_channels"] = _resolve_scalar_in_channels(in_channels, head_config)
        if feature_layers is not None and "feature_layers" not in cfg:
            cfg["feature_layers"] = list(feature_layers)
        return build_mif_feature_unetr(cfg)
    return _ORIGINAL_BUILD_SEGMENTATION_HEAD(
        head_name,
        feature_layers=feature_layers,
        in_channels=in_channels,
        num_classes=num_classes,
        head_config=head_config,
        **kwargs,
    )


def _resolve_scalar_in_channels(in_channels, head_config: dict) -> int:
    if isinstance(in_channels, int):
        return int(in_channels)
    if isinstance(in_channels, (list, tuple)) and len(in_channels) == 1:
        return int(in_channels[0])
    if "in_channels" in head_config:
        return int(head_config["in_channels"])
    raise ValueError(
        "mif_unet/mif_feature require scalar in_channels in head_config "
        f"(got in_channels={in_channels!r})"
    )


def forward_segmentation_head(model, features, masks, images=None):
    from heads.cellseg import CellsegUNETR
    from heads.mif_feature import MifFeatureUNETR
    from heads.mif_unet import MifUNet

    routed = getattr(model, "module", model)
    output_size = tuple(masks.shape[-2:])
    if isinstance(routed, CellsegUNETR):
        return model(features, image=images, output_size=output_size)
    if isinstance(routed, MifUNet):
        if images is not None and images.shape[1] == 3 and routed.generator is not None:
            return model(rgb=images, output_size=output_size)
        return model(image=images, output_size=output_size)
    if isinstance(routed, MifFeatureUNETR):
        return model(rgb=images, output_size=output_size)
    return _ORIGINAL_FORWARD_SEGMENTATION_HEAD(model, features, masks, images=images)


class ShazamSurvivalABMIL(nn.Module):
    """Shazam single-teacher ABMIL survival head (bias-free attention)."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        embed_dim: int | None = None,
        dropout: float = 0.25,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        embed = int(hidden_dim if embed_dim is None else embed_dim)
        hidden = int(hidden_dim)
        self.fc1 = nn.Linear(int(input_dim), embed)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(embed, hidden)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(float(dropout))
        self.attention = nn.Sequential(nn.Linear(hidden, 1, bias=False))
        self.classifier = nn.Linear(embed, int(n_classes))
        self.last_attention = None

    def forward(self, x_path):
        x = self.dropout1(self.relu(self.fc1(x_path)))
        a_h = self.dropout2(self.tanh(self.fc2(x)))
        a = self.attention(a_h)
        w = torch.softmax(a, dim=0)
        self.last_attention = w.detach()
        z = (w * x).sum(dim=0, keepdim=True)
        logits = self.classifier(z)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        prediction = torch.topk(logits, 1, dim=1)[1]
        return hazards, survival, prediction


HEAD_REGISTRY = dict(HEAD_REGISTRY)
HEAD_REGISTRY["shazam_survival_abmil"] = ShazamSurvivalABMIL
