"""Per-layer feature-directory helpers for dual encoder / virtual-stain extraction."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import h5py
import numpy as np
from utils.file_utils import PATCH_BAG_SCHEMA_VERSION, PATCH_BAG_STAGE_COMPLETE, is_complete_patch_bag_h5, patch_bag_part_path
OutputPathsFn = Callable[[Any, str, str], Tuple[Any, Any]]
def default_layer_feat_dirs(feat_dir: Path, layers: Sequence[str]) -> Dict[str, Path]:
    """Map layers → feat roots.\n\n    First layer keeps ``feat_dir`` (compatible with existing encoder bags).\n    Remaining layers use ``{feat_dir}__{layer}``.\n    """
    if not layers:
        raise ValueError('layers must be non-empty')
    else:
        mapping = {}
        for index, layer in enumerate(layers):
            name = str(layer).strip()
            if not name:
                continue
            else:
                if index == 0:
                    mapping[name] = Path(feat_dir)
                else:
                    mapping[name] = Path(f'{feat_dir}__{name}')
        if not mapping:
            raise ValueError('layers must contain at least one non-empty name')
        else:
            return mapping
def resolve_layer_feat_dirs(feat_dir: Path, layers: Sequence[str], *, split_layer_feat_dirs: bool, layer_feat_dirs: Optional[Mapping[str, Any]]=None) -> Optional[Dict[str, Path]]:
    """Return per-layer base feat dirs, or ``None`` when splitting is disabled."""
    layer_list = [str(x).strip() for x in layers if str(x).strip()]
    if not split_layer_feat_dirs or len(layer_list) <= 1:
        return None
    else:
        if layer_feat_dirs:
            mapping = {}
            for layer in layer_list:
                if layer not in layer_feat_dirs:
                    raise KeyError(f'run.layer_feat_dirs missing entry for target_features layer {layer!r}')
                else:
                    mapping[layer] = Path(layer_feat_dirs[layer])
            return mapping
        else:
            return default_layer_feat_dirs(feat_dir, layer_list)
def _split_relative(exp) -> Path:
    """Relative split subdir under the original feat_dir base (``train`` / ``.``)."""
    base = getattr(exp, 'feat_dir_base', None)
    if base is None:
        return Path('.')
    else:
        try:
            rel = Path(exp.feat_dir).resolve().relative_to(Path(base).resolve())
        except ValueError:
            return Path('.')
        return rel
def active_layer_roots(exp) -> Optional[Dict[str, Path]]:
    """Per-layer feat roots for the *current* split (``feat_dir`` may already be ``…/train``)."""
    bases = getattr(exp, 'layer_feat_dirs', None)
    if not bases:
        return
    else:
        rel = _split_relative(exp)
        if str(rel) in {'', '.'}:
            return {layer: Path(root) for layer, root in bases.items()}
        else:
            return {layer: Path(root) / rel for layer, root in bases.items()}
def layer_h5_paths(exp, sample_id: str, model_name: str, *, output_paths_fn: OutputPathsFn) -> Optional[Dict[str, Path]]:
    """Concrete H5 destinations for each configured layer, or ``None`` if not splitting."""
    roots = active_layer_roots(exp)
    if not roots:
        return
    else:
        saved = exp.feat_dir
        out = {}
        try:
            for layer, root in roots.items():
                exp.feat_dir = root
                _pt, h5_path = output_paths_fn(exp, sample_id, model_name)
                out[layer] = Path(h5_path)
        finally:
            exp.feat_dir = saved
    return out
def export_patch_bag_layers(src_h5: str | Path, dest_by_layer: Mapping[str, str | Path], *, keep_src: bool=False) -> Dict[str, Path]:
    """Split a multi-layer patch bag into one single-layer bag per destination."""
    src_h5 = Path(src_h5)
    if not dest_by_layer:
        raise ValueError('dest_by_layer must be non-empty')
    else:
        written = {}
        with h5py.File(src_h5, 'r') as src:
            if 'features' not in src or not isinstance(src['features'], h5py.Group):
                raise ValueError(f'{src_h5} has no features/ group')
            else:
                available = set(src['features'].keys())
                missing = [layer for layer in dest_by_layer if layer not in available]
                if missing:
                    raise KeyError(f'{src_h5} missing layer(s) {missing}; available={sorted(available)}')
                else:
                    coords = np.asarray(src['coords'])
                    masks = np.asarray(src['masks']) if 'masks' in src else None
                    patch_uids = np.asarray(src['patch_uids']) if 'patch_uids' in src else None
                    attrs = {key: src.attrs[key] for key in src.attrs.keys()}
                    for layer, dest in dest_by_layer.items():
                        dest_path = Path(dest)
                        part = patch_bag_part_path(dest_path)
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        if part.is_file():
                            part.unlink()
                        feat = np.asarray(src['features'][layer])
                        with h5py.File(part, 'w') as dst:
                            dst.create_dataset('coords', data=coords.astype(np.int32, copy=False), chunks=(1, 2), dtype=np.int32)
                            if masks is not None:
                                dst.create_dataset('masks', data=masks, chunks=(1,) + masks.shape[1:], dtype=masks.dtype)
                            if patch_uids is not None:
                                dst.create_dataset('patch_uids', data=patch_uids.astype(np.int64, copy=False), chunks=(1,), dtype=np.int64)
                            feature_group = dst.create_group('features')
                            feature_group.create_dataset(layer, data=feat, chunks=(1,) + feat.shape[1:], dtype=feat.dtype)
                            for key, value in attrs.items():
                                if key in {'bag_stage', 'schema_version', 'feature_layers'}:
                                    continue
                                else:
                                    dst.attrs[key] = value
                            dst.attrs['schema_version'] = attrs.get('schema_version', PATCH_BAG_SCHEMA_VERSION)
                            dst.attrs['bag_stage'] = PATCH_BAG_STAGE_COMPLETE
                            dst.attrs['feature_layers'] = json.dumps([layer])
                            if 'feature_dtype' in attrs:
                                dst.attrs['feature_dtype'] = attrs['feature_dtype']
                            if masks is not None and 'mask_dtype' in attrs:
                                dst.attrs['mask_dtype'] = attrs['mask_dtype']
                        os.replace(part, dest_path)
                        written[layer] = dest_path
        if not keep_src and src_h5.is_file():
                src_h5.unlink()
        return written
def export_split_h5_layers(src_h5: str | Path, dest_by_layer: Mapping[str, str | Path], *, keep_src: bool=False) -> Dict[str, Path]:
    """Split a multi-layer single-cell H5 into one single-layer H5 per destination."""
    src_h5 = Path(src_h5)
    if not dest_by_layer:
        raise ValueError('dest_by_layer must be non-empty')
    else:
        written = {}
        meta_keys = ('labels', 'coords', 'sample_ids', 'case_ids', 'label_names', 'image_paths')
        with h5py.File(src_h5, 'r') as src:
            if 'features' not in src or not isinstance(src['features'], h5py.Group):
                raise ValueError(f'{src_h5} has no features/ group')
            else:
                available = set(src['features'].keys())
                missing = [layer for layer in dest_by_layer if layer not in available]
                if missing:
                    raise KeyError(f'{src_h5} missing layer(s) {missing}; available={sorted(available)}')
                else:
                    attrs = {key: src.attrs[key] for key in src.attrs.keys()}
                    for layer, dest in dest_by_layer.items():
                        dest_path = Path(dest)
                        part = dest_path.with_name(dest_path.name + '.part')
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        if part.is_file():
                            part.unlink()
                        if dest_path.is_file():
                            dest_path.unlink()
                        feat = np.asarray(src['features'][layer])
                        with h5py.File(part, 'w') as dst:
                            for key, value in attrs.items():
                                if key == 'feature_layers':
                                    continue
                                else:
                                    dst.attrs[key] = value
                            dst.attrs['feature_layers'] = json.dumps([layer])
                            group = dst.create_group('features')
                            chunk = (min(1024, feat.shape[0]),) + feat.shape[1:]
                            group.create_dataset(layer, data=feat, chunks=chunk)
                            for key in meta_keys:
                                if key not in src:
                                    continue
                                else:
                                    data = np.asarray(src[key])
                                    if data.dtype.kind in {'O', 'S', 'U'}:
                                        dst.create_dataset(key, data=data)
                                    else:
                                        dst.create_dataset(key, data=data, chunks=(min(1024, data.shape[0]),) + data.shape[1:])
                        os.replace(part, dest_path)
                        written[layer] = dest_path
        if not keep_src and src_h5.is_file():
                src_h5.unlink()
        return written
def single_cell_layer_h5_paths(exp, split_filter: Optional[str], *, split_h5_path_fn: Callable[..., Any]) -> Optional[Dict[str, Path]]:
    """Per-layer single-cell H5 destinations for the current split, or ``None``."""
    roots = active_layer_roots(exp)
    if not roots:
        return
    else:
        saved = exp.feat_dir
        out = {}
        try:
            for layer, root in roots.items():
                exp.feat_dir = root
                out[layer] = Path(split_h5_path_fn(exp, split_filter))
        finally:
            exp.feat_dir = saved
    return out
