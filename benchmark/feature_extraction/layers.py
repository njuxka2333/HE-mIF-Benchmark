
"""Per-model layer forward maps and feature selection."""
from __future__ import annotations
import re
from typing import Dict, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
VIT_EVEN_LAYER_NAMES = ('even1', 'even2', 'even3', 'even4')
VIT_EVEN_ALIASES = tuple((f'vit_{name}' for name in VIT_EVEN_LAYER_NAMES))
def gap_map(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)
def _norm_name(name: str) -> str:
    return re.sub('[\\s\\-]+', '_', name.strip().lower())
def _rosie_cnn_stage_layers(wrapper: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    from models.rosie_inference import CNN_INPUT_SIZE
    backbone = wrapper.backbone if hasattr(wrapper, 'backbone') else wrapper
    features = backbone.features
    is_raw = float(x.detach().amin().cpu()) >= (-0.1) and float(x.detach().amax().cpu()) <= 1.5
    h, w = x.shape[(-2):]
    if h != CNN_INPUT_SIZE or w != CNN_INPUT_SIZE:
        x = F.interpolate(x, size=(CNN_INPUT_SIZE, CNN_INPUT_SIZE), mode='bilinear', align_corners=False)
    if is_raw:
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
    index_to_name = {0: 'stem', 1: 'stage0', 3: 'stage1', 5: 'stage2', 7: 'stage3'}
    aliases = {'layer1': 'stage0', 'layer3': 'stage1', 'layer5': 'stage2', 'layer7': 'stage3', 'convnext_stage3': 'stage3'}
    out = {}
    for i, layer in enumerate(features):
        x = layer(x)
        if i in index_to_name:
            out[index_to_name[i]] = x
    for alias, target in aliases.items():
        if target in out:
            out[alias] = out[target]
    return out
def _rosie_splat_pyramid_layers(wrapper: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Native ROSIE 50-channel Gaussian-splat output as a 4-level spatial pyramid."""
    if not hasattr(wrapper, 'num_channels'):
        raise TypeError('ROSIE splat layers require ROSIESpatialWrapper')
    else:
        splat = wrapper(x)
        if splat.dim() != 4:
            raise TypeError(f'ROSIE splat expected [B, C, H, W], got {tuple(splat.shape)}')
        else:
            out = {}
            feat = splat
            for i, name in enumerate(('mif0', 'mif1', 'mif2', 'mif3')):
                out[name] = feat
                out[f'splat{i}'] = feat
                if i < 3:
                    feat = F.avg_pool2d(feat, kernel_size=2, stride=2)
            return out
_ROSIE_STAGE_LAYERS = frozenset({'stage1', 'stage2', 'stage0', 'convnext_stage3', 'layer1', 'layer7', 'layer5', 'stage3', 'stem', 'layer3'})
_ROSIE_SPLAT_LAYERS = frozenset({'mif', 'mif2', 'mif3', 'splat0', 'splat2', 'mif1', 'mif0', 'output', 'final', 'splat1', 'splat3'})
OUTPUT_LAYER_ALIASES = frozenset({'output', 'mif', 'final'})
def _attach_output_aliases(out: Dict[str, torch.Tensor], tensor: torch.Tensor, *, primary: str='mif') -> None:
    """Publish prediction-head tensor under ``mif`` / ``output`` / ``final`` (+ primary)."""
    out[primary] = tensor
    for alias in OUTPUT_LAYER_ALIASES:
        out[alias] = tensor
def _rosie_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    requested = {_norm_name(name) for name in requested_layers or []}
    need_stages = not requested or bool(requested & _ROSIE_STAGE_LAYERS)
    need_splat = bool(requested & _ROSIE_SPLAT_LAYERS)
    if requested and (not need_stages) and (not need_splat):
        raise KeyError(f'Unknown ROSIE layer(s) {sorted(requested)}. Use ConvNeXt stages (stage0-stage3) or splat pyramid (mif0-mif3 / mif).')
    else:
        out = {}
        if need_stages:
            out.update(_rosie_cnn_stage_layers(wrapper, x))
        if need_splat:
            out.update(_rosie_splat_pyramid_layers(wrapper, x))
            if 'mif0' in out:
                _attach_output_aliases(out, out['mif0'], primary='mif')
        return out
def _hemit_single_layers(net: nn.Module, x: torch.Tensor, *, need_fused: bool) -> Dict[str, torch.Tensor]:
    z0 = net.initial_layers(x)
    feat = z0
    downsampled = []
    for down_layer in net.downsampling_layers:
        feat = down_layer(feat)
        downsampled.append(feat)
    z4 = net.resnet_blocks(feat)
    z1 = downsampled[0] if downsampled else z0
    z2 = downsampled[1] if len(downsampled) > 1 else z1
    z3 = downsampled[2] if len(downsampled) > 2 else z2
    out = {'z0': z0, 'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'resnet_blocks': z4, 'bottleneck': z4}
    if not need_fused:
        return out
    else:
        h, w = x.shape[(-2):]
        if hasattr(net.swinT.patch_embed, 'img_size'):
            net.swinT.patch_embed.img_size = (h, w)
        if hasattr(net.swinT, 'img_size'):
            net.swinT.img_size = (h, w)
        tokens = net.swinT.patch_embed(x)
        tokens = net.swinT.pos_drop(tokens)
        swin_features = []
        for stage in net.swinT.layers:
            if stage.downsample is not None:
                tokens = stage.downsample(tokens)
            if len(swin_features) < len(net.cross_atts):
                expected_channels = net.cross_atts[len(swin_features)].swinT_transform.in_channels
                token_channels = tokens.shape[(-1)] if tokens.ndim in (3, 4) else (-1)
                if token_channels == expected_channels:
                    if tokens.ndim == 3:
                        side = int(round(tokens.shape[1] ** 0.5))
                        if side * side != tokens.shape[1]:
                            raise ValueError(f'HEMIT flattened Swin tokens do not form a square grid: {tuple(tokens.shape)}')
                        else:
                            swin = tokens.view(tokens.shape[0], side, side, expected_channels).permute(0, 3, 1, 2)
                    else:
                        swin = tokens.permute(0, 3, 1, 2)
                    swin_features.append(swin.contiguous())
            for block in stage.blocks:
                tokens = block(tokens)
        if len(swin_features) < 3:
            raise ValueError(f'HEMIT expected three Swin skip maps, got {len(swin_features)}')
        else:
            for index, (cnn, swin) in enumerate(zip(downsampled[:3], swin_features[:3], strict=False), start=1):
                out[f'fused_z{index}'] = net.cross_atts[index - 1](cnn, swin)
            return out
_HEMIT_ENCODER_LAYERS = frozenset({'fused_z2', 'z1', 'z3', 'resnet_blocks', 'bottleneck', 'z2', 'fused_z3', 'z0', 'z4', 'fused_z1'})
def _hemit_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    """Expose CNN / Swin-fused skips and optional virtual-staining output (``mif``)."""
    net = wrapper.net_g if hasattr(wrapper, 'net_g') else wrapper
    requested = {_norm_name(name) for name in requested_layers or []}
    need_output = bool(requested & OUTPUT_LAYER_ALIASES) if requested else False
    need_encoder = not requested or bool(requested & _HEMIT_ENCODER_LAYERS) or any((name.startswith('fused_z') for name in requested))
    out = {}
    if need_encoder:
        need_fused = any((name.startswith('fused_z') for name in requested))
        if not need_fused:
            out.update(_hemit_single_layers(net, x, need_fused=False))
        else:
            per_image = [_hemit_single_layers(net, x[index:index + 1], need_fused=True) for index in range(x.shape[0])]
            out.update({name: torch.cat([features[name] for features in per_image], dim=0) for name in per_image[0]})
    if need_output:
        pred = wrapper(x) if hasattr(wrapper, 'forward') else net(x)
        _attach_output_aliases(out, pred, primary='mif')
    if requested and (not out):
        raise KeyError(f'Unknown HEMIT layer(s) {sorted(requested)}. Use encoder skips (z0-z4 / fused_z*) or mif/output/final.')
    else:
        return out
_GIGATIME_ENCODER_LAYERS = frozenset({'x4_0', 'x3_0', 'x4_2', 'x4_3', 'x2_0', 'x1_0', 'x0_0', 'x4_1'})
def _gigatime_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    requested = {_norm_name(name) for name in requested_layers or []}
    need_output = bool(requested & OUTPUT_LAYER_ALIASES) if requested else False
    need_encoder = not requested or bool(requested & _GIGATIME_ENCODER_LAYERS)
    out = {}
    if need_encoder:
        n = wrapper.net if hasattr(wrapper, 'net') else wrapper
        x0_0 = n.conv0_0(x)
        x1_0 = n.conv1_0(n.pool(x0_0))
        x2_0 = n.conv2_0(n.pool(x1_0))
        x3_0 = n.conv3_0(n.pool(x2_0))
        x4_0 = n.conv4_0(n.pool(x3_0))
        out.update({'x0_0': x0_0, 'x1_0': x1_0, 'x2_0': x2_0, 'x3_0': x3_0, 'x4_0': x4_0, 'x4_1': x4_0, 'x4_2': x4_0, 'x4_3': x4_0})
    if need_output:
        pred = wrapper(x)
        _attach_output_aliases(out, pred, primary='mif')
    if requested and (not out):
        raise KeyError(f'Unknown GigaTIME layer(s) {sorted(requested)}. Use encoder maps (x0_0-x4_0) or mif/output/final.')
    else:
        return out
def _evenly_spaced_vit_indices(depth: int, n: int=4) -> List[int]:
    """Return approximately 25/50/75/100% ViT block indices."""
    if depth < n:
        raise ValueError(f'ViT depth must be >= {n}, got {depth}')
    else:
        indices = [round((i + 1) * depth / n) - 1 for i in range(n)]
        return [max(0, min(depth - 1, int(idx))) for idx in indices]
def _mipheivit_last_four_vit_blocks(encoder: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Last four ViT block maps (PFMB / UNI2 ``features_only`` style)."""
    vit = encoder.vit
    depth = len(vit.blocks)
    if depth < 4:
        raise ValueError(f'MIPHEI ViT expected depth >= 4, got {depth}')
    else:
        indices = list(range(depth - 4, depth))
        feats = vit.forward_intermediates(x, indices=indices, norm=False, output_fmt='NCHW', intermediates_only=True)
        return list(feats)
def _forward_vit_indices_from_wrapper(wrapper: nn.Module, x: torch.Tensor, indices: Sequence[int], model_name: str) -> List[torch.Tensor]:
    if hasattr(wrapper, 'forward_layers_by_indices'):
        layers = wrapper.forward_layers_by_indices(x, list(indices))
        if not isinstance(layers, (list, tuple)):
            raise TypeError(f'{model_name} indexed layer extraction expects a list/tuple')
        else:
            return [layer for layer in layers if layer is not None]
    else:
        net = wrapper.net if hasattr(wrapper, 'net') else wrapper
        encoder = getattr(net, 'encoder', None)
        vit = getattr(encoder, 'vit', None)
        if vit is None:
            vit = getattr(net, 'model', None) or net
        if not hasattr(vit, 'forward_intermediates'):
            raise TypeError(f'{model_name} does not expose forward_intermediates for indexed ViT layers')
        else:
            feats = vit.forward_intermediates(x, indices=list(indices), norm=False, output_fmt='NCHW', intermediates_only=True)
            return [feat for feat in feats if feat is not None]
def _add_evenly_spaced_vit_layers(out: Dict[str, torch.Tensor], wrapper: nn.Module, x: torch.Tensor, *, model_name: str) -> None:
    net = wrapper.net if hasattr(wrapper, 'net') else wrapper
    encoder = getattr(net, 'encoder', None)
    vit = getattr(encoder, 'vit', None)
    if vit is None:
        vit = getattr(net, 'model', None) or net
    blocks = getattr(vit, 'blocks', None)
    if blocks is None:
        raise TypeError(f'{model_name} does not expose ViT blocks for even-spaced layers')
    else:
        indices = _evenly_spaced_vit_indices(len(blocks))
        layers = _forward_vit_indices_from_wrapper(wrapper, x, indices, model_name)
        if len(layers) < 4:
            raise ValueError(f'{model_name} expected 4 even-spaced feature maps, got {len(layers)}')
        else:
            for name, feat in zip(VIT_EVEN_LAYER_NAMES, layers[(-4):], strict=False):
                out[name] = feat
                out[f'vit_{name}'] = feat
_MIPHEI_ENCODER_LAYERS = frozenset({'z1', 'z2', 'z3', 'z4', 'encoder', 'layer1', 'layer2', 'layer3', 'layer4', 'z4_finalnorm', *VIT_EVEN_LAYER_NAMES, *VIT_EVEN_ALIASES})
def _mipheivit_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    # ***<module>._mipheivit_layers: Failure: Different control flow
    net = wrapper.net if hasattr(wrapper, 'net') else wrapper
    requested = {_norm_name(name) for name in requested_layers or []}
    need_output = bool(requested & OUTPUT_LAYER_ALIASES) if requested else False
    need_encoder = not requested or bool(requested & _MIPHEI_ENCODER_LAYERS)
    out = {}
    want_pyramid = bool(requested & ({'z1', 'z3', 'z4_finalnorm', 'z2', 'layer1', 'layer2', 'layer3'} | set(VIT_EVEN_LAYER_NAMES) | set(VIT_EVEN_ALIASES)))
    want_z4 = not requested or bool(requested & {'z4', 'layer4'}) or want_pyramid
    want_encoder_map = 'encoder' in requested
    if need_output and need_encoder and want_z4 and (not want_pyramid) and (not want_encoder_map):
        from models.patch_compat import restore_spatial, spatial_hw
        h, w = spatial_hw(x)
        if hasattr(wrapper, '_prepare_encoder_grid'):
            wrapper._prepare_encoder_grid(h, w)
        vit = net.encoder.vit
        captured = {}
        def _capture_last_block(_module, _inputs, output):
            captured['tokens'] = output
        handle = vit.blocks[(-1)].register_forward_hook(_capture_last_block)
        x_enc = x
        orig_hw = (h, w)
        resized = False
        try:
            features = net.encoder(x_enc)
            pred = net.decoder(features, x_enc)
        except (RuntimeError, ValueError, AssertionError):
            if not hasattr(wrapper, '_upsample_to_build_size'):
                raise
            else:
                x_enc, orig_hw, resized = wrapper._upsample_to_build_size(x)
                if not resized:
                    raise
                else:
                    wrapper._prepare_encoder_grid(wrapper.build_size, wrapper.build_size)
                    features = net.encoder(x_enc)
                    pred = net.decoder(features, x_enc)
        finally:
            handle.remove()
    else:
        if need_encoder:
            if hasattr(wrapper, '_prepare_encoder_grid'):
                wrapper._prepare_encoder_grid(x.shape[(-2)], x.shape[(-1)])
            if hasattr(wrapper, 'forward_layers'):
                layers = wrapper.forward_layers(x)
            else:
                layers = _mipheivit_last_four_vit_blocks(net.encoder, x)
            if not isinstance(layers, (list, tuple)):
                raise TypeError('MIPHEI layer extraction expects a list/tuple of feature maps')
            else:
                layers = [layer for layer in layers if layer is not None]
                if len(layers) < 4:
                    raise ValueError(f'MIPHEI expected at least 4 feature maps, got {len(layers)}')
                else:
                    z1, z2, z3, z4 = layers[(-4):]
                    out.update({'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'layer1': z1, 'layer2': z2, 'layer3': z3, 'layer4': z4})
                    if want_encoder_map or not requested:
                        out['encoder'] = net.encoder(x)
                    if 'z4_finalnorm' in requested:
                        vit = net.encoder.vit
                        final_norm = getattr(vit, 'norm', None)
                        if final_norm is None:
                            raise TypeError('MIPHEI encoder does not expose the final token LayerNorm as vit.norm')
                        else:
                            out['z4_finalnorm'] = final_norm(z4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    if requested & (set(VIT_EVEN_LAYER_NAMES) | set(VIT_EVEN_ALIASES)):
                        _add_evenly_spaced_vit_layers(out, wrapper, x, model_name='MIPHEI')
        if need_output:
            pred = wrapper(x)
            _attach_output_aliases(out, pred, primary='mif')
        if requested and (not out):
            raise KeyError(f'Unknown MIPHEI layer(s) {sorted(requested)}. Use encoder maps (z1-z4 / even*) or mif/output/final.')
        else:
            return out
    tokens = captured.get('tokens')
    if tokens is None:
        raise RuntimeError('MIPHEI failed to capture last ViT block for z4')
    else:
        if tokens.ndim != 3:
            raise TypeError(f'Expected ViT block tokens [B,N,C], got {tuple(tokens.shape)}')
        else:
            prefix = int(getattr(net.encoder, 'num_prefix_tokens', 0) or 0)
            tok = tokens[:, prefix:, :] if prefix else tokens
            side = int(round(tok.shape[1] ** 0.5))
            if side * side != tok.shape[1]:
                raise ValueError(f'MIPHEI z4 tokens do not form a square grid: N={tok.shape[1]} prefix={prefix}')
            else:
                z4 = tok.permute(0, 2, 1).contiguous().view(tok.shape[0], tok.shape[2], side, side)
                if resized:
                    z4 = restore_spatial(z4, orig_hw, resized=True)
                    pred = restore_spatial(pred, orig_hw, resized=True)
                out['z4'] = z4
                out['layer4'] = z4
                _attach_output_aliases(out, pred, primary='mif')
                return out
def _uni2_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    if hasattr(wrapper, 'forward_layers'):
        layers = wrapper.forward_layers(x)
    else:
        layers = wrapper(x)
    if not isinstance(layers, (list, tuple)):
        raise TypeError('UNI2 layer extraction expects a list/tuple of feature maps')
    else:
        layers = [layer for layer in layers if layer is not None]
        if len(layers) < 4:
            raise ValueError(f'UNI2 expected at least 4 feature maps, got {len(layers)}')
        else:
            z1, z2, z3, z4 = layers[(-4):]
            out = {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'layer1': z1, 'layer2': z2, 'layer3': z3, 'layer4': z4, 'encoder': z4}
            requested = {_norm_name(name) for name in requested_layers or []}
            if requested & (set(VIT_EVEN_LAYER_NAMES) | set(VIT_EVEN_ALIASES)):
                _add_evenly_spaced_vit_layers(out, wrapper, x, model_name='UNI2')
            return out
_HISTOPLEXER_ENCODER_LAYERS = frozenset({'encoder_6', 'z1', 'z3', 'center_up', 'encoder_3', 'bottleneck', 'encoder_5', 'z2', 'encoder_1', 'z0', 'encoder_0', 'encoder_2', 'z4', 'encoder_4', 'center_down', 'z5', 'z6'})
def _histoplexer_layers(wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    requested = {_norm_name(name) for name in requested_layers or []}
    need_output = bool(requested & OUTPUT_LAYER_ALIASES) if requested else False
    need_encoder = not requested or bool(requested & _HISTOPLEXER_ENCODER_LAYERS)
    out = {}
    if need_encoder:
        if hasattr(wrapper, 'forward_layers'):
            out.update(wrapper.forward_layers(x))
        else:
            raise TypeError('HistoPlexer layer extraction expects a wrapper.forward_layers(x) method')
    if need_output:
        pred = wrapper(x)
        if isinstance(pred, (list, tuple)):
            if not pred:
                raise ValueError('HistoPlexer forward returned an empty output list')
            else:
                pred = pred[(-1)]
        _attach_output_aliases(out, pred, primary='mif')
    if requested and (not out):
        raise KeyError(f'Unknown HistoPlexer layer(s) {sorted(requested)}. Use encoder/bottleneck maps or mif/output/final.')
    else:
        return out
_LAYER_FN = {'rosie': _rosie_layers, 'hemit': _hemit_layers, 'gigatime': _gigatime_layers, 'mipheivit': _mipheivit_layers, 'uni2': _uni2_layers, 'histoplexer': _histoplexer_layers}
MODEL_LAYER_NAMES: Dict[str, List[str]] = {'rosie': ['stem', 'stage0', 'stage1', 'stage2', 'stage3', 'mif0', 'mif1', 'mif2', 'mif3', 'mif', 'output', 'final', 'splat0', 'splat1', 'splat2', 'splat3'], 'hemit': ['z0', 'z1', 'z2', 'z3', 'z4', 'fused_z1', 'fused_z2', 'fused_z3', 'resnet_blocks', 'bottleneck', 'mif', 'output', 'final'], 'gigatime': ['x0_0', 'x1_0', 'x2_0', 'x3_0', 'x4_0', 'mif', 'output', 'final'], 'mipheivit': ['z1', 'z2', 'z3', 'z4', 'encoder', 'layer1', 'layer2', 'layer3', 'layer4', 'even1', 'even2', 'even3', 'even4', 'mif', 'output', 'final'], 'uni2': ['z1', 'z2', 'z3', 'z4', 'encoder', 'even1', 'even2', 'even3', 'even4'], 'histoplexer': ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'encoder_0', 'encoder_1', 'encoder_2', 'encoder_3', 'encoder_4', 'encoder_5', 'encoder_6', 'center_down', 'bottleneck', 'center_up', 'mif', 'output', 'final']}
def compute_model_layers(model: str, wrapper: nn.Module, x: torch.Tensor, requested_layers: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    aliases = {'hemmit': 'hemit', 'miphei': 'mipheivit', 'uni_2': 'uni2'}
    key = aliases.get(model, model)
    if key == 'rosie':
        return _rosie_layers(wrapper, x, requested_layers=requested_layers)
    else:
        if key == 'mipheivit':
            return _mipheivit_layers(wrapper, x, requested_layers=requested_layers)
        else:
            if key == 'uni2':
                return _uni2_layers(wrapper, x, requested_layers=requested_layers)
            else:
                if key == 'hemit':
                    return _hemit_layers(wrapper, x, requested_layers=requested_layers)
                else:
                    fn = _LAYER_FN.get(key)
                    if fn is None:
                        raise ValueError(f'No layer map for model \'{model}\'')
                    else:
                        if key in {'histoplexer', 'gigatime'}:
                            return fn(wrapper, x, requested_layers=requested_layers)
                        else:
                            return fn(wrapper, x)
def select_layers(all_layers: Dict[str, torch.Tensor], names: Sequence[str]) -> Dict[str, torch.Tensor]:
    if not names:
        raise ValueError('target_features must list at least one layer name')
    else:
        selected = {}
        missing = []
        for name in names:
            key = _norm_name(name)
            if key in all_layers:
                selected[key] = all_layers[key]
            else:
                missing.append(name)
        if missing:
            available = ', '.join(sorted(all_layers))
            raise KeyError(f'Unknown layer(s) {missing}. Available: {available}')
        else:
            return selected
def format_feature_output(layers: Dict[str, torch.Tensor], use_gap: bool):
    if use_gap:
        if len(layers) == 1:
            name = next(iter(layers))
            return {name: gap_map(layers[name])}
        else:
            return {k: gap_map(v) for k, v in layers.items()}
    else:
        if len(layers) == 1:
            return next(iter(layers.values()))
        else:
            return layers
