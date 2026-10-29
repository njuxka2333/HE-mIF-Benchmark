"""Run the configured encoder/output extraction loop."""
from __future__ import annotations
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data import DataLoader
from data.loader import SplitSample, build_wsi_path_map, find_wsi_path, load_split_samples
from utils.experiment_config import load_experiment_config
from feature_extraction.wsi_io import (
    find_wsi_paths,
    open_wsi,
    save_hdf5_subprocess,
    save_patch_bag_subprocess,
    save_tensor_subprocess,
)
from feature_extraction import build_feature_extractor
from feature_extraction.layer_storage import (
    active_layer_roots,
    export_patch_bag_layers,
    export_split_h5_layers,
    layer_h5_paths,
    single_cell_layer_h5_paths,
)
from models import get_custom_transformer, get_model
from utils.output_layout import format_output_paths
from utils.file_utils import cast_features_for_storage, cast_masks_for_storage, is_complete_patch_bag_h5
from utils.resize_ops import apply_mpp_patch_geometry_at, minimal_cover_grid, mpp_patch_sizes
STANDARD_SPLITS = ('train', 'valid', 'test')
IMMUNOCTO_NATIVE_MPP = 0.25
def _shard_samples(samples: list[SplitSample], shard_id: int, num_shards: int, *, by_unique_id: bool) -> list[SplitSample]:
    if num_shards <= 1:
        return samples
    else:
        if by_unique_id:
            unique_ids = []
            for sample in samples:
                if sample.sample_id not in unique_ids:
                    unique_ids.append(sample.sample_id)
            allowed = {sid for idx, sid in enumerate(unique_ids) if idx % num_shards == shard_id}
            return [sample for sample in samples if sample.sample_id in allowed]
        else:
            return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_id]
def plan_split_runs(cli_split: str | None, yaml_split: str | None) -> list[tuple[str | None, str | None]]:
    """\n    Plan extraction runs as ``(split_filter, output_subdir)`` pairs.\n\n    When no split is set on CLI or YAML, run train / valid / test into subfolders.\n    When a split is set explicitly (including ``all``), use a single flat ``feat_dir``.\n    """
    explicit = cli_split if cli_split is not None else yaml_split
    if explicit is None:
        return [(name, name) for name in STANDARD_SPLITS]
    else:
        if str(explicit).lower() in {'', '*', 'all'}:
            return [(None, None)]
        else:
            return [(str(explicit), None)]
def _uses_only_all_split(exp) -> bool:
    if getattr(exp.dataset_config, 'manifest_format', None) == 'immunocto_compact':
        return False
    else:
        samples = load_split_samples(exp.dataset_config, split_filter=None)
        return bool(samples) and {sample.split for sample in samples} == {'all'}
def collate_batch(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    masks = None
    if len(batch[0]) > 2 and batch[0][2] is not None:
            masks = np.stack([item[2] for item in batch], axis=0)
    return (imgs, coords, masks)
class SingleCellPatchDataset(Dataset):
    """One ImmunOcto cell image per item for efficient split-level extraction."""
    def __init__(self, samples, transform, target_patch_size, target_mpp, native_mpp):
        self.samples = list(samples)
        self.transform = transform
        self.target_patch_size = target_patch_size
        self.target_mpp = target_mpp
        self.native_mpp = native_mpp
        self.read_size, self.output_size = mpp_patch_sizes(target_patch_size, target_mpp, native_mpp=native_mpp, default_patch_size=target_patch_size if target_patch_size > 0 else 32)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert('RGB')
        width, height = img.size
        x, y = minimal_cover_grid(height, width, self.read_size)[0]
        img, _ = apply_mpp_patch_geometry_at(img, None, x, y, self.read_size, self.output_size)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return (img, idx, np.array([x, y], dtype=np.int32))
def collate_single_cell_batch(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    indices = np.asarray([item[1] for item in batch], dtype=np.int64)
    coords = np.stack([item[2] for item in batch], axis=0)
    return (imgs, indices, coords)
class WdsSingleCellDataset(IterableDataset):
    """Stream ImmunOcto WebDataset tar shards with worker-level shard partitioning."""
    def __init__(self, shards, transform, target_patch_size, target_mpp, native_mpp, label_mapping, allowed_sample_ids=None):
        self.shards = list(shards)
        self.transform = transform
        self.target_patch_size = target_patch_size
        self.target_mpp = target_mpp
        self.native_mpp = native_mpp
        self.label_mapping = dict(label_mapping)
        self.allowed_sample_ids = None if allowed_sample_ids is None else {str(value) for value in allowed_sample_ids}
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shards = self.shards
        else:
            shards = self.shards[worker.id::worker.num_workers]
        for shard in shards:
            for png_bytes, meta in _iter_wds_shard_samples(shard):
                if self.allowed_sample_ids is not None and str(meta['image_id']) not in self.allowed_sample_ids:
                        continue
                tensor, coord = _wds_image_to_tensor(png_bytes, self.transform, self.target_patch_size, self.target_mpp, self.native_mpp)
                label_name = str(meta['label'])
                yield (tensor, coord, int(self.label_mapping[label_name]), str(meta['image_id']), str(meta['patient_id']), label_name, str(meta['image_path']))
def collate_wds_single_cell_batch(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    coords = np.stack([item[1] for item in batch], axis=0)
    labels = np.asarray([item[2] for item in batch], dtype=np.int64)
    return (imgs, coords, labels, [item[3] for item in batch], [item[4] for item in batch], [item[5] for item in batch], [item[6] for item in batch])
def _build_model(exp, device):
    base_model = get_model(exp.model, device, torch.cuda.device_count(), use_benchmark_config=exp.use_benchmark_config, config_path=str(exp.benchmark_config_path) if exp.benchmark_config_path else None)
    transform = get_custom_transformer(exp.model, use_benchmark_config=exp.use_benchmark_config, config_path=str(exp.benchmark_config_path) if exp.benchmark_config_path else None)
    model = build_feature_extractor(base_model, exp.extraction_task()).to(device).eval()
    return (model, transform)
def _output_paths(exp, sample_id, model_name):
    feat_dir = exp.feat_dir
    if exp.output_layout == 'prepath_legacy':
        pt_path = feat_dir / 'pt_files' / model_name / f'{sample_id}.pt'
        h5_path = feat_dir / 'h5_files' / model_name / f'{sample_id}.h5'
    else:
        paths = format_output_paths(exp.output_layout, str(feat_dir), model_name, sample_id, str(exp.benchmark_config_path) if exp.benchmark_config_path else None)
        pt_path = Path(paths.get('features') or paths['spatial'])
        h5_path = Path(paths['coords'])
    return (pt_path, h5_path)
def _is_complete_output(exp, sample_id, model_name) -> bool:
    layer_paths = layer_h5_paths(exp, sample_id, model_name, output_paths_fn=_output_paths)
    if layer_paths:
        return all(is_complete_patch_bag_h5(path) for path in layer_paths.values())
    pt_path, h5_path = _output_paths(exp, sample_id, model_name)
    if exp.feature_storage == 'patch_h5':
        return is_complete_patch_bag_h5(h5_path)
    else:
        return pt_path.is_file() and pt_path.stat().st_size > 0 and h5_path.is_file() and (h5_path.stat().st_size > 0)
def _should_skip_existing(exp, sample_id, model_name, args) -> bool:
    if not getattr(args, 'skip_existing', True):
        return False
    else:
        if _is_complete_output(exp, sample_id, model_name):
            _, h5_path = _output_paths(exp, sample_id, model_name)
            print(f'skip complete {h5_path}')
            return True
        else:
            return False
def _save_outputs(exp, sample_id, feature_obj, coords, masks, model_name):
    layer_roots = active_layer_roots(exp)
    if layer_roots and isinstance(feature_obj, dict):
        missing = [name for name in layer_roots if name not in feature_obj]
        if missing:
            raise KeyError(f"feature output missing configured layers: {missing}")
        original_dir, original_layer_dirs = exp.feat_dir, exp.layer_feat_dirs
        try:
            exp.layer_feat_dirs = None
            for layer, root in layer_roots.items():
                exp.feat_dir = root
                _save_outputs(
                    exp,
                    sample_id,
                    {layer: feature_obj[layer]},
                    coords,
                    masks,
                    model_name,
                )
        finally:
            exp.feat_dir, exp.layer_feat_dirs = original_dir, original_layer_dirs
        return
    pt_path, h5_path = _output_paths(exp, sample_id, model_name)
    os.makedirs(h5_path.parent, exist_ok=True)
    feature_obj = cast_features_for_storage(feature_obj, exp.feature_dtype)
    masks = cast_masks_for_storage(masks, exp.mask_dtype)
    if exp.feature_storage == 'patch_h5':
        attr_dict = {'model': model_name, 'experiment_id': exp.id, 'feature_dtype': exp.feature_dtype, 'mask_dtype': exp.mask_dtype}
        save_patch_bag_subprocess(str(h5_path), feature_obj, coords, masks if exp.save_masks else None, attr_dict=attr_dict, feature_dtype=exp.feature_dtype, mask_dtype=exp.mask_dtype)
        if isinstance(feature_obj, dict):
            shape = {k: v.shape for k, v in feature_obj.items()}
        else:
            shape = feature_obj.shape
        print(f'saved {h5_path} schema=patch_bag_v1 shape={shape}')
        return
    else:
        os.makedirs(pt_path.parent, exist_ok=True)
        save_tensor_subprocess(str(pt_path), feature_obj)
        asset = {'coords': coords}
        if exp.save_masks and masks is not None:
                mask_dir = exp.feat_dir / 'mask_files' / model_name
                os.makedirs(mask_dir, exist_ok=True)
                asset['masks'] = masks
                save_tensor_subprocess(str(mask_dir / f'{sample_id}.pt'), masks)
        save_hdf5_subprocess(str(h5_path), asset_dict=asset)
        if isinstance(feature_obj, dict):
            shape = {k: v.shape for k, v in feature_obj.items()}
        else:
            shape = feature_obj.shape
        print(f'saved {pt_path} shape={shape}')
def _run_loader(model, loader, device):
    feats, coords_list, masks_list = ([], [], [])
    with torch.no_grad():
        for batch in loader:
            imgs, coords, masks = batch
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            if isinstance(out, dict):
                feats.append({k: v.detach().cpu() for k, v in out.items()})
            else:
                feats.append(out.detach().cpu())
            coords_list.append(coords)
            if masks is not None:
                masks_list.append(masks)
    if not feats:
        raise ValueError('feature loader produced no batches')
    else:
        if isinstance(feats[0], dict):
            feature_obj = {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}
        else:
            feature_obj = torch.cat(feats, dim=0)
        coords = np.concatenate(coords_list, axis=0)
        masks = np.concatenate(masks_list, axis=0) if masks_list else None
        return (feature_obj, coords, masks)
def _append_h5_dataset(handle, name: str, arr: np.ndarray, *, dtype) -> None:
    arr = np.asarray(arr).astype(dtype, copy=False)
    if name not in handle:
        handle.create_dataset(name, shape=(0,) + arr.shape[1:], maxshape=(None,) + arr.shape[1:], chunks=(1,) + arr.shape[1:], dtype=dtype)
    dataset = handle[name]
    start = int(dataset.shape[0])
    dataset.resize((start + int(arr.shape[0]),) + dataset.shape[1:])
    dataset[start:start + int(arr.shape[0])] = arr
def _append_wsi_feature_batch(feature_group, feature_obj, *, feature_dtype: str) -> list[str]:
    feat_dtype = np.dtype(feature_dtype)
    layer_names = []
    if isinstance(feature_obj, dict):
        for name, tensor in feature_obj.items():
            arr = tensor.detach().cpu().numpy().astype(feat_dtype, copy=False)
            _append_h5_dataset(feature_group, str(name), arr, dtype=feat_dtype)
            layer_names.append(str(name))
    else:
        arr = feature_obj.detach().cpu().numpy().astype(feat_dtype, copy=False)
        _append_h5_dataset(feature_group, 'features', arr, dtype=feat_dtype)
        layer_names = ['features']
    return layer_names
def _write_patch_bag_h5(model, loader, device, h5_path: Path, exp, model_name: str) -> int:
    """Stream one WSI bag to patch_h5 batch-by-batch to avoid end-of-slide spikes."""
    part_path = h5_path.with_name(f'{h5_path.name}.part')
    if part_path.is_file():
        part_path.unlink()
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    attr_dict = {'model': model_name, 'experiment_id': exp.id, 'feature_dtype': exp.feature_dtype, 'mask_dtype': exp.mask_dtype}
    total = 0
    layer_names = []
    mask_dtype = np.dtype(exp.mask_dtype)
    with h5py.File(part_path, 'w') as handle:
        handle.attrs['schema_version'] = 'patch_bag_v1'
        handle.attrs['bag_stage'] = 'in_progress'
        for key, value in attr_dict.items():
            handle.attrs[key] = value
        feature_group = handle.create_group('features')
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                imgs, coords, masks = batch
                imgs = imgs.to(device, non_blocking=True)
                out = model(imgs)
                batch_layers = _append_wsi_feature_batch(feature_group, out, feature_dtype=exp.feature_dtype)
                if not layer_names:
                    layer_names = batch_layers
                    handle.attrs['feature_layers'] = json.dumps(layer_names)
                else:
                    if set(layer_names) != set(batch_layers):
                        raise ValueError('inconsistent feature layer names across streamed batches')
                _append_h5_dataset(handle, 'coords', np.asarray(coords, dtype=np.int32), dtype=np.int32)
                if exp.save_masks and masks is not None:
                        _append_h5_dataset(handle, 'masks', masks, dtype=mask_dtype)
                total += int(coords.shape[0])
                if batch_idx % 100 == 0:
                    print(f'  streamed {total} patches to {h5_path.name}.part', flush=True)
        if total <= 0:
            raise ValueError('feature loader produced no batches')
        else:
            handle.attrs['bag_stage'] = 'complete'
            handle.attrs['feature_dtype'] = str(np.dtype(exp.feature_dtype))
            if exp.save_masks:
                handle.attrs['mask_dtype'] = str(mask_dtype)
    os.replace(part_path, h5_path)
    print(f'Successfully saved streamed patch bag: {h5_path}')
    return total


def _run_loader_to_patch_h5(model, loader, device, h5_path: Path, exp, model_name: str) -> int:
    """Stream a WSI bag, splitting its layers when the config assigns separate roots."""
    layer_paths = layer_h5_paths(
        exp, h5_path.stem, model_name, output_paths_fn=_output_paths
    )
    if not layer_paths:
        return _write_patch_bag_h5(model, loader, device, h5_path, exp, model_name)

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="feature_layers_", dir=h5_path.parent) as temp_dir:
        staged_path = Path(temp_dir) / h5_path.name
        total = _write_patch_bag_h5(model, loader, device, staged_path, exp, model_name)
        export_patch_bag_layers(staged_path, layer_paths)
    return total
def _output_sample_id(sample: SplitSample) -> str:
    return Path(sample.sample_id).stem
def _split_h5_path(exp, split_filter: str | None) -> Path:
    split = split_filter or 'all'
    return exp.feat_dir / f'{split}_{exp.model}_features.h5'
def _is_complete_split_h5(path: Path, expected_rows: int) -> bool:
    # irreducible cflow, using cdg fallback
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with h5py.File(path, 'r') as handle:
            return (
                handle.attrs.get('schema_version') == 'single_cell_features_v1'
                and int(handle.attrs.get('n_rows', -1)) == int(expected_rows)
                and 'features' in handle
                and 'labels' in handle
            )
    except OSError:
        return False
def _label_mapping(cfg, samples) -> dict[str, int]:
    label_dict = cfg.label_dict
    if label_dict:
        return {str(name): int(idx) for idx, name in label_dict.items()}
    else:
        return {label: idx for idx, label in enumerate(sorted({sample.label for sample in samples}))}
def _append_dataset(handle, name: str, values, *, dtype=None) -> None:
    arr = np.asarray(values, dtype=dtype)
    if name not in handle:
        maxshape = (None,) + arr.shape[1:]
        chunks = (min(max(1, arr.shape[0]), 8192),) + arr.shape[1:]
        handle.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks, dtype=arr.dtype)
    else:
        dataset = handle[name]
        old = int(dataset.shape[0])
        dataset.resize((old + arr.shape[0],) + dataset.shape[1:])
        dataset[old:old + arr.shape[0]] = arr
def _append_string_dataset(handle, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding='utf-8')
    arr = np.asarray([str(v) for v in values], dtype=object)
    if name not in handle:
        handle.create_dataset(name, data=arr, maxshape=(None,), chunks=(min(len(arr), 8192),), dtype=dtype)
    else:
        dataset = handle[name]
        old = int(dataset.shape[0])
        dataset.resize((old + len(arr),))
        dataset[old:old + len(arr)] = arr
def _feature_group_append(feature_group, name: str, arr: np.ndarray, feature_dtype: np.dtype) -> None:
    if name not in feature_group:
        maxshape = (None,) + arr.shape[1:]
        chunks = (min(max(1, arr.shape[0]), 8192),) + arr.shape[1:]
        feature_group.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks, dtype=feature_dtype)
    else:
        dataset_h5 = feature_group[name]
        old = int(dataset_h5.shape[0])
        dataset_h5.resize((old + arr.shape[0],) + dataset_h5.shape[1:])
        dataset_h5[old:old + arr.shape[0]] = arr
def _append_feature_batch(handle, feature_group, out, feature_dtype: np.dtype) -> list[str]:
    if isinstance(out, dict):
        layer_names = list(out)
        for name, tensor in out.items():
            arr = tensor.detach().cpu().numpy().astype(feature_dtype, copy=False)
            _feature_group_append(feature_group, name, arr, feature_dtype)
        return layer_names
    else:
        arr = out.detach().cpu().numpy().astype(feature_dtype, copy=False)
        _feature_group_append(feature_group, 'features', arr, feature_dtype)
        return ['features']
def _wds_split_dir(exp, split_filter: str | None) -> Path:
    if exp.wds_root is None:
        raise ValueError('run.wds_root must be set when run.input_format=webdataset')
    else:
        return exp.wds_root / (split_filter or 'all')
def _selection_sample_ids(args, split_filter: str | None) -> set[str] | None:
    selection_csv = getattr(args, 'selection_csv', None)
    if selection_csv is None:
        return
    else:
        import csv
        requested_split = str(split_filter or 'all')
        selected = set()
        with Path(selection_csv).open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            required = {'sample_id', 'split'}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f'selection CSV missing columns {sorted(missing)}: {selection_csv}')
            else:
                for row in reader:
                    if requested_split == 'all' or str(row['split']) == requested_split:
                        selected.add(str(row['sample_id']))
        if not selected:
            raise ValueError(f'No selected sample IDs for split={requested_split!r} in {selection_csv}')
        else:
            return selected
def _wds_split_count(exp, split_filter: str | None) -> int:
    split = split_filter or 'all'
    report = _wds_split_dir(exp, split_filter) / f'immunocto-{split}-report.json'
    if report.is_file():
        with report.open(encoding='utf-8') as handle:
            data = json.load(handle)
        return int(data['samples'])
    else:
        manifest = _wds_split_dir(exp, split_filter) / f'immunocto-{split}-shards.csv'
        if manifest.is_file():
            import csv
            total = 0
            with manifest.open(newline='', encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    total += int(row.get('samples', 0))
            return total
        else:
            return 0
def _iter_wds_shards(exp, split_filter: str | None) -> list[Path]:
    split = split_filter or 'all'
    split_dir = _wds_split_dir(exp, split_filter)
    manifest = split_dir / f'immunocto-{split}-shards.csv'
    if manifest.is_file():
        import csv
        shards = []
        with manifest.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                shard = split_dir / str(row['shard'])
                if shard.is_file():
                    shards.append(shard)
        return shards
    else:
        return sorted(split_dir.glob(f'immunocto-{split}-*.tar'))
def _iter_wds_shard_samples(shard: Path):
    pending = {}
    with tarfile.open(shard, 'r') as tar:
        for member in tar:
            if not member.isfile():
                continue
            else:
                key, suffix = (Path(member.name).stem, Path(member.name).suffix)
                slot = pending.setdefault(key, {})
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                else:
                    payload = handle.read()
                    if suffix == '.json':
                        slot['metadata'] = json.loads(payload.decode('utf-8'))
                    else:
                        if suffix == '.png':
                            slot['png'] = payload
                    if 'metadata' in slot and 'png' in slot:
                            item = pending.pop(key)
                            yield (item['png'], item['metadata'])
def _wds_image_to_tensor(png_bytes: bytes, transform, target_patch_size: int, target_mpp, native_mpp: float):
    img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    read_size, output_size = mpp_patch_sizes(target_patch_size, target_mpp, native_mpp=native_mpp, default_patch_size=target_patch_size if target_patch_size > 0 else 32)
    width, height = img.size
    x, y = minimal_cover_grid(height, width, read_size)[0]
    img, _ = apply_mpp_patch_geometry_at(img, None, x, y, read_size, output_size)
    if transform:
        tensor = transform(img)
    else:
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (tensor, np.array([x, y], dtype=np.int32))
def _run_single_cell_split_h5(exp, samples, model, transform, device, args, split_filter: str | None) -> int:
    layer_paths = single_cell_layer_h5_paths(
        exp, split_filter, split_h5_path_fn=_split_h5_path
    )
    if layer_paths:
        n_samples = len(samples)
        if args.skip_existing and all(
            _is_complete_split_h5(path, n_samples) for path in layer_paths.values()
        ):
            print("skip complete per-layer HDFs: " + ", ".join(map(str, layer_paths.values())))
            return n_samples

        original_dir = exp.feat_dir
        original_base = exp.feat_dir_base
        original_layer_dirs = exp.layer_feat_dirs
        staging_parent = Path(original_dir).parent
        staging_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="single_cell_layers_", dir=staging_parent
        ) as temp_dir:
            try:
                exp.feat_dir = Path(temp_dir)
                exp.feat_dir_base = Path(temp_dir)
                exp.layer_feat_dirs = None
                count = _run_single_cell_split_h5(
                    exp, samples, model, transform, device, args, split_filter
                )
                staged_path = _split_h5_path(exp, split_filter)
                export_split_h5_layers(staged_path, layer_paths)
                return count
            finally:
                exp.feat_dir = original_dir
                exp.feat_dir_base = original_base
                exp.layer_feat_dirs = original_layer_dirs

    output_path = _split_h5_path(exp, split_filter)
    if args.skip_existing and _is_complete_split_h5(output_path, len(samples)):
        print(f'skip complete {output_path}')
        return len(samples)
    else:
        tmp_path = output_path.with_name(f'{output_path.name}.part')
        if tmp_path.is_file():
            tmp_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        label_mapping = _label_mapping(exp.dataset_config, samples)
        target_patch_size = args.target_patch_size if args.target_patch_size > 0 else exp.target_patch_size
        target_mpp = exp.target_mpp if exp.use_mpp else None
        dataset = SingleCellPatchDataset(samples, transform, target_patch_size, target_mpp, IMMUNOCTO_NATIVE_MPP)
        loader_kwargs = {'batch_size': args.batch_size, 'num_workers': int(getattr(args, 'num_workers', 8)), 'pin_memory': device.type == 'cuda', 'collate_fn': collate_single_cell_batch}
        if loader_kwargs['num_workers'] > 0:
            loader_kwargs['prefetch_factor'] = 4
        loader = DataLoader(dataset, **loader_kwargs)
        feature_dtype = np.dtype(exp.feature_dtype)
        use_amp = device.type == 'cuda' and str(exp.feature_dtype).lower() == 'float16'
        with h5py.File(tmp_path, 'w') as handle, torch.inference_mode():
                handle.attrs['schema_version'] = 'single_cell_features_v1'
                handle.attrs['experiment_id'] = exp.id
                handle.attrs['model'] = exp.model
                handle.attrs['split'] = split_filter or 'all'
                handle.attrs['feature_dtype'] = str(feature_dtype)
                handle.attrs['label_mapping'] = json.dumps(label_mapping, sort_keys=True)
                feature_group = handle.create_group('features')
                n_rows = 0
                for batch_idx, (imgs, indices, coords) in enumerate(loader, start=1):
                    imgs = imgs.to(device, non_blocking=True)
                    out = model(imgs)
                    batch_samples = [samples[int(i)] for i in indices.tolist()]
                    labels = np.asarray([label_mapping[str(sample.label)] for sample in batch_samples], dtype=np.int64)
                    if isinstance(out, dict):
                        layer_names = list(out)
                        for name, tensor in out.items():
                            arr = tensor.detach().cpu().numpy().astype(feature_dtype, copy=False)
                            if name not in feature_group:
                                maxshape = (None,) + arr.shape[1:]
                                chunks = (min(max(1, arr.shape[0]), 8192),) + arr.shape[1:]
                                feature_group.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks, dtype=feature_dtype)
                            else:
                                dataset_h5 = feature_group[name]
                                old = int(dataset_h5.shape[0])
                                dataset_h5.resize((old + arr.shape[0],) + dataset_h5.shape[1:])
                                dataset_h5[old:old + arr.shape[0]] = arr
                    else:
                        layer_names = ['features']
                        arr = out.detach().cpu().numpy().astype(feature_dtype, copy=False)
                        if 'features' not in feature_group:
                            maxshape = (None,) + arr.shape[1:]
                            chunks = (min(max(1, arr.shape[0]), 8192),) + arr.shape[1:]
                            feature_group.create_dataset('features', data=arr, maxshape=maxshape, chunks=chunks, dtype=feature_dtype)
                        else:
                            dataset_h5 = feature_group['features']
                            old = int(dataset_h5.shape[0])
                            dataset_h5.resize((old + arr.shape[0],) + dataset_h5.shape[1:])
                            dataset_h5[old:old + arr.shape[0]] = arr
                    _append_dataset(handle, 'labels', labels, dtype=np.int64)
                    _append_dataset(handle, 'coords', coords, dtype=np.int32)
                    _append_string_dataset(handle, 'sample_ids', [sample.sample_id for sample in batch_samples])
                    _append_string_dataset(handle, 'case_ids', [sample.case_id for sample in batch_samples])
                    _append_string_dataset(handle, 'label_names', [sample.label for sample in batch_samples])
                    _append_string_dataset(handle, 'image_paths', [sample.extra.get('image_path', str(sample.image_path)) for sample in batch_samples])
                    n_rows += len(batch_samples)
                    handle.attrs['n_rows'] = n_rows
                    handle.attrs['feature_layers'] = json.dumps(layer_names)
                    if batch_idx % 100 == 0:
                        print(f'  wrote {n_rows}/{len(samples)} cells to {tmp_path}')
        os.replace(tmp_path, output_path)
        print(f'saved {output_path} n={len(samples)}')
        return len(samples)
def _run_single_cell_wds_split_h5(exp, model, transform, device, args, split_filter: str | None) -> int:
    allowed_sample_ids = _selection_sample_ids(args, split_filter)
    expected_rows = len(allowed_sample_ids) if allowed_sample_ids is not None else _wds_split_count(exp, split_filter)
    if expected_rows <= 0:
        raise FileNotFoundError(f'No WDS samples found for split={split_filter} under {exp.wds_root}')
    else:
        output_path = _split_h5_path(exp, split_filter)
        if args.skip_existing and _is_complete_split_h5(output_path, expected_rows):
            print(f'skip complete {output_path}')
            return expected_rows
        else:
            tmp_path = output_path.with_name(f'{output_path.name}.part')
            if tmp_path.is_file():
                tmp_path.unlink()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            label_mapping = {str(name): int(idx) for idx, name in exp.dataset_config.label_dict.items()}
            if not label_mapping:
                label_mapping = {'bcell': 0, 'cd4': 1, 'cd8': 2, 'macrophage': 3, 'other': 4}
            target_patch_size = args.target_patch_size if args.target_patch_size > 0 else exp.target_patch_size
            target_mpp = exp.target_mpp if exp.use_mpp else None
            native_mpp = IMMUNOCTO_NATIVE_MPP
            feature_dtype = np.dtype(exp.feature_dtype)
            shards = _iter_wds_shards(exp, split_filter)
            if not shards:
                raise FileNotFoundError(f'No WDS tar shards found for split={split_filter} under {exp.wds_root}')
            else:
                dataset = WdsSingleCellDataset(shards, transform, target_patch_size, target_mpp, native_mpp, label_mapping, allowed_sample_ids=allowed_sample_ids)
                loader_kwargs = {'batch_size': int(args.batch_size), 'num_workers': int(getattr(args, 'num_workers', 8)), 'pin_memory': device.type == 'cuda', 'collate_fn': collate_wds_single_cell_batch}
                if loader_kwargs['num_workers'] > 0:
                    loader_kwargs['prefetch_factor'] = 4
                loader = DataLoader(dataset, **loader_kwargs)
                use_amp = device.type == 'cuda' and str(exp.feature_dtype).lower() == 'float16'
                print(f"  split={split_filter or 'all'} n={expected_rows} shards={len(shards)} workers={loader_kwargs['num_workers']} input={exp.wds_root} output={exp.feat_dir}", flush=True)
                with h5py.File(tmp_path, 'w') as handle, torch.inference_mode():
                    handle.attrs['schema_version'] = 'single_cell_features_v1'
                    handle.attrs['experiment_id'] = exp.id
                    handle.attrs['model'] = exp.model
                    handle.attrs['split'] = split_filter or 'all'
                    handle.attrs['feature_dtype'] = str(feature_dtype)
                    handle.attrs['label_mapping'] = json.dumps(label_mapping, sort_keys=True)
                    handle.attrs['input_format'] = 'webdataset'
                    feature_group = handle.create_group('features')
                    n_rows = 0
                    layer_names = ['features']
                    for imgs, coords, labels, sample_ids, case_ids, label_names, image_paths in loader:
                        imgs = imgs.to(device, non_blocking=True)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                            out = model(imgs)
                        layer_names = _append_feature_batch(handle, feature_group, out, feature_dtype)
                        _append_dataset(handle, 'labels', labels, dtype=np.int64)
                        _append_dataset(handle, 'coords', coords, dtype=np.int32)
                        _append_string_dataset(handle, 'sample_ids', sample_ids)
                        _append_string_dataset(handle, 'case_ids', case_ids)
                        _append_string_dataset(handle, 'label_names', label_names)
                        _append_string_dataset(handle, 'image_paths', image_paths)
                        n_rows += int(imgs.shape[0])
                        handle.attrs['n_rows'] = n_rows
                        handle.attrs['feature_layers'] = json.dumps(layer_names)
                        if n_rows % max(1, int(args.batch_size) * 100) == 0:
                            print(f'  wrote {n_rows}/{expected_rows} cells to {tmp_path}', flush=True)
                    if n_rows != expected_rows:
                        raise RuntimeError(f'WDS row count mismatch for {split_filter}: wrote {n_rows}, expected {expected_rows}')
    os.replace(tmp_path, output_path)
    print(f'saved {output_path} n={expected_rows}')
    return expected_rows
def extract_wsi_slide(exp, slide_id, wsi, h5_path, model, transform, device, args, mask_path=None) -> bool:
    from data.dataset_h5 import Whole_Slide_Bag_FP
    if _should_skip_existing(exp, slide_id, exp.model, args):
        return True
    else:
        target_mpp = exp.target_mpp if exp.use_mpp else None
        target_patch_size = args.target_patch_size if args.target_patch_size > 0 else exp.target_patch_size
        dataset = Whole_Slide_Bag_FP(file_path=str(h5_path), wsi=wsi, custom_transforms=transform, target_patch_size=target_patch_size, fast_read=True, target_mpp=target_mpp, fallback_mpp=exp.fallback_mpp, is_patch_dataset=False, mask_path=mask_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=int(getattr(args, 'num_workers', 0) or 0), pin_memory=device.type == 'cuda', collate_fn=collate_batch)
        if exp.feature_storage == 'patch_h5':
            _, h5_path = _output_paths(exp, slide_id, exp.model)
            n_patches = _run_loader_to_patch_h5(model, loader, device, h5_path, exp, exp.model)
            print(f'saved {h5_path} schema=patch_bag_v1 streamed_rows={n_patches}')
            return True
        else:
            feature_obj, coords, masks = _run_loader(model, loader, device)
            _save_outputs(exp, slide_id, feature_obj, coords, masks, exp.model)
            return True
def _run_wsi_samples(exp, samples, model, transform, device, args) -> int:
    dcfg = exp.dataset_config
    wsi_map = build_wsi_path_map(dcfg)
    if not wsi_map:
        for root in dcfg.wsi_search_roots():
            wsi_map.update(find_wsi_paths(str(root), dcfg.slide_ext))
    seen = set()
    processed = 0
    for sample in samples:
        slide_id = sample.sample_id
        if slide_id in seen:
            continue
        else:
            seen.add(slide_id)
            h5_path = dcfg.coords_dir() / f'{slide_id}.h5'
            if not h5_path.is_file():
                print('missing coord bag', h5_path, '(run preprocessing/create_patches_fp.py first)')
                continue
            else:
                wsi_path = sample.image_path or wsi_map.get(slide_id) or find_wsi_path(slide_id, dcfg)
                if wsi_path is None or not Path(wsi_path).is_file():
                    print('no wsi', slide_id)
                    continue
        try:
            wsi = open_wsi(str(wsi_path))
        except Exception as exc:
            print('failed to open wsi', slide_id, wsi_path, exc)
            continue
        if extract_wsi_slide(exp, slide_id, wsi, h5_path, model, transform, device, args, mask_path=sample.mask_path):
            processed += 1
    return processed
def run_extraction(exp, model, transform, device, args, split_filter: str | None) -> int:
    """Extract features for one split filter. Returns number of samples successfully processed."""
    dcfg = exp.dataset_config
    if exp.feature_storage == 'split_h5' and exp.input_format == 'webdataset':
        expected_rows = _wds_split_count(exp, split_filter)
        if expected_rows <= 0:
            label = split_filter or 'all'
            print(f'  split={label}: no WDS shards, skipping')
            return 0
        else:
            return _run_single_cell_wds_split_h5(exp, model, transform, device, args, split_filter)
    else:
        samples = load_split_samples(dcfg, split_filter=split_filter)
        allowed_sample_ids = _selection_sample_ids(args, split_filter)
        if allowed_sample_ids is not None:
            samples = [sample for sample in samples if str(sample.sample_id) in allowed_sample_ids]
        if not samples:
            label = split_filter or 'all'
            print(f'  split={label}: no samples, skipping')
            return 0
        else:
            shard_id = int(getattr(args, 'shard_id', 0) or 0)
            num_shards = int(getattr(args, 'num_shards', 1) or 1)
            if num_shards > 1:
                n_before = len(samples)
                samples = _shard_samples(samples, shard_id, num_shards, by_unique_id=exp.is_wsi)
                if not samples:
                    label = split_filter or 'all'
                    print(f'  split={label}: shard {shard_id}/{num_shards} has no samples, skipping')
                    return 0
                else:
                    print(f'  shard={shard_id}/{num_shards} samples={len(samples)}/{n_before}')
            split_label = split_filter or 'all'
            print(f'  split={split_label} n={len(samples)} output={exp.feat_dir}')
            if exp.is_wsi:
                return _run_wsi_samples(exp, samples, model, transform, device, args)
            else:
                if dcfg.is_patch_hdf:
                    raise ValueError(f'{exp.dataset} uses mode=\'patch_hdf\'. Offline HDF feature extraction was removed; use training/train.py for PathoCell online segmentation.')
                else:
                    if exp.feature_storage == 'split_h5':
                        return _run_single_cell_split_h5(exp, samples, model, transform, device, args, split_filter)
                    else:
                        raise ValueError(f'Unsupported extraction combination for dataset={exp.dataset!r}: mode={dcfg.mode!r}, feature_storage={exp.feature_storage!r}')
