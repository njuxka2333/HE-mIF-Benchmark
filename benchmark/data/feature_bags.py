
"""Datasets for downstream training on extracted feature tensors."""
from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.experiment_config import ExperimentConfig, load_experiment_config
from utils.file_utils import is_patch_bag_h5
def _immunocto_patient_id_from_path(raw_path: str, csv_path: Path) -> str:
    """Extract the CRC patient ID from a canonical ``.../CRCxx/HE/*.png`` path."""
    path_parts = Path(raw_path).parts
    if len(path_parts) < 3:
        raise ValueError(f'Cannot infer Immunocto patient ID from exclusion path {raw_path!r} in {csv_path}')
    else:
        try:
            he_idx = path_parts.index('HE')
            return path_parts[he_idx - 1]
        except ValueError as exc:
            raise ValueError(f'Expected an Immunocto .../CRCxx/HE/<image>.png path, got {raw_path!r} in {csv_path}') from exc
def _immunocto_relative_path(raw_path: str, csv_path: Path) -> str:
    patient_id = _immunocto_patient_id_from_path(raw_path, csv_path)
    return str(Path(patient_id) / 'HE' / Path(raw_path).name)
def _immunocto_split_patients(split: str) -> set[str] | None:
    """Return canonical patients for an Immunocto split; unknown/all means no filter."""
    split_norm = str(split).strip().lower()
    if split_norm == 'train':
        return {f'CRC{i:02d}' for i in range(1, 29)}
    else:
        if split_norm in {'val', 'valid'}:
            return {f'CRC{i:02d}' for i in range(29, 33)}
        else:
            if split_norm == 'test':
                return {f'CRC{i:02d}' for i in range(33, 41)}
def _load_immunocto_excluded_sample_keys(path: str | Path, *, split: str | None=None) -> set[str]:
    """Return excluded relative image paths, restricted to a requested split."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'Immunocto exclusion CSV not found: {csv_path}')
    else:
        split_patients = _immunocto_split_patients(split or '')
        keys = set()
        with csv_path.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                raw_path = str(row.get('image_path') or row.get('path') or row.get('relative_path') or '').strip()
                patient_id = _immunocto_patient_id_from_path(raw_path, csv_path)
                if split_patients is not None and patient_id not in split_patients:
                        continue
                keys.add(_immunocto_relative_path(raw_path, csv_path))
        return keys
def _resolve_feature_storage(exp: ExperimentConfig, pt_path: Path, h5_path: Path) -> str:
    storage = getattr(exp, 'feature_storage', 'auto')
    if storage == 'patch_h5':
        return 'patch_h5'
    else:
        if storage == 'legacy_pt':
            return 'legacy_pt'
        else:
            if h5_path.is_file() and is_patch_bag_h5(h5_path):
                    return 'patch_h5'
            return 'legacy_pt'
def _feature_data_path(exp: ExperimentConfig, sample_id: str, split: str) -> Path:
    pt_path, h5_path = _feature_paths(exp, sample_id, split)
    storage = _resolve_feature_storage(exp, pt_path, h5_path)
    return h5_path if storage == 'patch_h5' else pt_path
def _load_mil_feature_file(path: Path) -> Any:
    if path.suffix.lower() == '.h5':
        with h5py.File(path, 'r') as handle:
            features = handle['features']
            if isinstance(features, h5py.Group):
                return {name: torch.as_tensor(features[name][()]) for name in features.keys()}
            else:
                return torch.as_tensor(features[()])
    return torch.load(path, map_location='cpu')
def _coordinate_order(handle: h5py.File, n_rows: int) -> np.ndarray:
    """Return stable lexicographic coordinate order for a patch-bag HDF5 file."""
    if 'coords' not in handle:
        return np.arange(n_rows, dtype=np.int64)
    else:
        coords = np.asarray(handle['coords'][:n_rows])
        if coords.ndim != 2 or coords.shape[0] != n_rows:
            raise ValueError(f'Invalid coords shape in {handle.filename}: {coords.shape}')
        else:
            keys = tuple((coords[:, index] for index in reversed(range(coords.shape[1]))))
            return np.lexsort(keys).astype(np.int64, copy=False)
def _load_mil_bag(paths: list[Path], layers: list[str], *, feature_normalize: str | None=None, norm_mean: torch.Tensor | None=None, norm_std: torch.Tensor | None=None, norm_clip_value: float | None=None, max_instances: int | None=None, deterministic_order: str='coordinate') -> torch.Tensor:
    """Load one patient bag, optionally retaining its first coordinate-ordered instances."""
    remaining = None if max_instances is None else int(max_instances)
    if remaining is not None and remaining <= 0:
        raise ValueError('max_instances_per_bag must be positive')
    chunks = []
    for path in sorted(paths, key=lambda item: (item.stem, str(item))):
        if remaining == 0:
            break
        feature_obj = _load_mil_feature_file(path)
        tensors = [_select_mil_feature(feature_obj, layer, feature_normalize=None) for layer in layers]
        chunk = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=(-1))
        if deterministic_order == 'coordinate' and path.suffix.lower() == '.h5':
            with h5py.File(path, 'r') as handle:
                order = _coordinate_order(handle, len(chunk))
            chunk = chunk[torch.as_tensor(order, dtype=torch.long)]
        else:
            if deterministic_order not in {'coordinate', 'stored'}:
                raise ValueError(f"deterministic_instance_order must be 'coordinate' or 'stored', got {deterministic_order!r}")
        if remaining is not None:
            chunk = chunk[:remaining]
            remaining -= len(chunk)
        chunks.append(chunk)
    if not chunks:
        raise ValueError('Cannot construct an empty MIL bag')
    bag = torch.cat(chunks, dim=0)
    bag = _normalize_mil_features(bag, feature_normalize, mean=norm_mean, std=norm_std)
    if norm_clip_value is not None:
        clip = float(norm_clip_value)
        if clip <= 0:
            raise ValueError('feature_norm_clip_value must be positive')
        bag = bag.clamp(min=-clip, max=clip)
    return bag
def _select_mil_feature(feature_obj: Any, layer: str | None, *, feature_normalize: str | None=None, norm_mean: torch.Tensor | None=None, norm_std: torch.Tensor | None=None, norm_clip_value: float | None=None) -> torch.Tensor:
    if isinstance(feature_obj, dict):
        if layer is None:
            if len(feature_obj) != 1:
                raise ValueError(f'feature_layer must be set when feature file has layers: {list(feature_obj)}')
            else:
                tensor = next(iter(feature_obj.values()))
        else:
            if layer not in feature_obj:
                if len(feature_obj) == 1 and 'features' in feature_obj:
                    tensor = feature_obj['features']
                    layer = None
                else:
                    raise KeyError(f'Missing feature layer \'{layer}\', available={list(feature_obj)}')
            if layer is not None:
                tensor = feature_obj[layer]
    else:
        tensor = feature_obj
    tensor = torch.as_tensor(tensor).float()
    if tensor.ndim == 4:
        tensor = tensor.mean(dim=((-1), (-2)))
    else:
        if tensor.ndim > 2:
            tensor = tensor.flatten(start_dim=2).mean(dim=(-1))
    if tensor.ndim != 2:
        raise ValueError(f'Expected MIL feature tensor NxC after pooling, got {tuple(tensor.shape)}')
    else:
        tensor = _normalize_mil_features(tensor, feature_normalize, mean=norm_mean, std=norm_std)
        if norm_clip_value is not None:
            clip = float(norm_clip_value)
            if clip <= 0:
                raise ValueError('feature_norm_clip_value must be positive')
            else:
                tensor = tensor.clamp(min=-clip, max=clip)
        return tensor
def _normalize_mil_features(tensor: torch.Tensor, mode: str | None, *, mean: torch.Tensor | None=None, std: torch.Tensor | None=None, eps: float=1e-06) -> torch.Tensor:
    norm_mode = str(mode or 'none').strip().lower()
    if norm_mode in {'false', 'none', '0', 'off', ''}:
        return tensor
    else:
        if norm_mode == 'l2':
            return torch.nn.functional.normalize(tensor, p=2, dim=(-1))
        else:
            if norm_mode == 'layernorm':
                return torch.nn.functional.layer_norm(tensor, (tensor.shape[(-1)],))
            else:
                if norm_mode in {'zscore', 'zscore_train'}:
                    if mean is None or std is None:
                        raise ValueError('zscore_train normalization requires feature_norm_mean and feature_norm_std')
                    else:
                        return (tensor - mean) / std.clamp_min(eps)
                else:
                    raise ValueError(f'Unknown feature_normalize mode: {mode}')
def _fit_mil_feature_norm_stats(rows: list[dict[str, Any]], *, feature_layer: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dimension mean/std on MIL bags (raw features, no normalization)."""
    chunks = []
    for row in rows:
        bag_feats = []
        for path in row['feature_paths']:
            feature_obj = _load_mil_feature_file(path)
            bag_feats.append(_select_mil_feature(feature_obj, feature_layer, feature_normalize=None))
        chunks.append(torch.cat(bag_feats, dim=0))
    stacked = torch.cat(chunks, dim=0)
    return (stacked.mean(dim=0), stacked.std(dim=0, unbiased=False))
def fit_classification_feature_norm_stats(dataset: 'ClassificationFeatureDataset', *, weighting: str='instance') -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only feature moments with instance- or patient-level weighting."""
    chunks = []
    for row in dataset.rows:
        chunks.append(_load_mil_bag(row['feature_paths'], dataset.feature_layers, max_instances=dataset.max_instances_per_bag, deterministic_order=dataset.deterministic_instance_order))
    norm_weighting = str(weighting).strip().lower()
    if norm_weighting == 'instance':
        stacked = torch.cat(chunks, dim=0)
        return (stacked.mean(dim=0), stacked.std(dim=0, unbiased=False))
    else:
        if norm_weighting == 'patient':
            patient_means = torch.stack([chunk.mean(dim=0) for chunk in chunks])
            patient_second_moments = torch.stack([(chunk * chunk).mean(dim=0) for chunk in chunks])
            mean = patient_means.mean(dim=0)
            variance = (patient_second_moments.mean(dim=0) - mean.square()).clamp_min(0.0)
            return (mean, variance.sqrt())
        else:
            raise ValueError(f'Unknown feature_norm_weighting={weighting!r}; expected \'instance\' or \'patient\'')
def fit_survival_feature_norm_stats(dataset: 'SurvivalFeatureDataset') -> tuple[torch.Tensor, torch.Tensor]:
    chunks = [_load_mil_bag(row['feature_paths'], [str(dataset.feature_layer)], max_instances=dataset.max_instances_per_bag, deterministic_order=dataset.deterministic_instance_order) for row in dataset.rows]
    stacked = torch.cat(chunks, dim=0)
    return (stacked.mean(dim=0), stacked.std(dim=0, unbiased=False))
class LinearProbeFeatureDataset(Dataset):
    """Patch-level classification features from split-level HDF5 files."""
    task_type = 'linear_probe'
    def __init__(self, feature_experiment_config: str | Path, split: str, *, feature_layer: str | None=None, feature_layers: list[str] | None=None, label_mapping: dict[str, int] | None=None, exclude_case_ids: list[str] | None=None, exclude_image_paths_csv: str | Path | None=None, feature_normalize: str | None=None, selection_csv: str | Path | None=None, max_samples: int | None=None, **_: Any):
        self.exp = load_experiment_config(feature_experiment_config)
        self.split = split
        self.feature_layer = feature_layer or (feature_layers[0] if feature_layers else None)
        self.feature_normalize = str(feature_normalize or 'none').lower()
        if self.feature_normalize not in {'zscore_train', 'none', 'l2'}:
            raise ValueError(f'linear-probe feature_normalize must be \'none\', \'l2\', or \'zscore_train\'; got {self.feature_normalize!r}')
        else:
            if self.exp.feature_storage != 'split_h5':
                raise ValueError(f'single-cell classification expects feature_storage=\'split_h5\'; got feature_storage={self.exp.feature_storage!r}')
            else:
                import json
                self.h5_path = self.exp.feat_dir / split / f'{split}_{self.exp.model}_features.h5'
                if not self.h5_path.is_file():
                    raise FileNotFoundError(f'No split-level linear-probe feature file found: {self.h5_path}')
                else:
                    with h5py.File(self.h5_path, 'r') as handle:
                        mapping_raw = handle.attrs.get('label_mapping', '{}')
                        self._label_mapping = {str(k): int(v) for k, v in json.loads(mapping_raw).items()}
                        source_rows = int(handle.attrs.get('n_rows', handle['labels'].shape[0]))
                        self._source_rows = source_rows
                        excluded = {str(case_id) for case_id in exclude_case_ids or []}
                        if excluded:
                            case_ids = np.asarray([value.decode('utf-8') for value in handle['case_ids'][:source_rows]], dtype=object)
                            self._indices = np.flatnonzero(~np.isin(case_ids, list(excluded)))
                        else:
                            self._indices = np.arange(source_rows, dtype=np.int64)
                        if exclude_image_paths_csv is None and self.exp.dataset == 'immunocto':
                                exclude_image_paths_csv = self.exp.dataset_config.excluded_paths_path
                        if exclude_image_paths_csv is not None:
                            excluded_sample_keys = _load_immunocto_excluded_sample_keys(exclude_image_paths_csv, split=split)
                            if excluded_sample_keys:
                                missing = {'image_paths'}.difference(handle.keys())
                                if missing:
                                    raise ValueError(f'{self.h5_path} cannot be filtered by image path; missing datasets: {sorted(missing)}')
                                else:
                                    keep = np.ones(source_rows, dtype=bool)
                                    path_data = handle['image_paths']
                                    chunk_size = 65536
                                    for start in range(0, source_rows, chunk_size):
                                        stop = min(start + chunk_size, source_rows)
                                        paths = path_data[start:stop]
                                        keep[start:stop] = [value.decode('utf-8') not in excluded_sample_keys for value in paths]
                                    self._indices = self._indices[keep[self._indices]]
                        if selection_csv is not None:
                            selection = pd.read_csv(selection_csv)
                            required = {'split', 'patient_id', 'sample_id', 'label', 'source_index'}
                            missing = required.difference(selection.columns)
                            if missing:
                                raise ValueError(f'Pilot selection CSV missing columns {sorted(missing)}: {selection_csv}')
                            else:
                                selection = selection[selection['split'].astype(str) == str(split)].copy()
                                selected_indices = selection['source_index'].astype(np.int64).to_numpy()
                                if len(selected_indices) == 0:
                                    raise ValueError(f'No rows for split={split!r} in {selection_csv}')
                                else:
                                    if selected_indices.min() < 0 or selected_indices.max() >= source_rows:
                                        raise IndexError(f'source_index outside [0, {source_rows}) in {selection_csv}')
                                    else:
                                        actual_samples = np.asarray([value.decode('utf-8') for value in handle['sample_ids'][selected_indices]])
                                        expected_samples = selection['sample_id'].astype(str).to_numpy()
                                        if not np.array_equal(actual_samples, expected_samples):
                                            raise ValueError(f'sample_id/source_index mismatch for split={split!r} in {selection_csv}')
                                        else:
                                            actual_patients = np.asarray([value.decode('utf-8') for value in handle['case_ids'][selected_indices]])
                                            expected_patients = selection['patient_id'].astype(str).to_numpy()
                                            if not np.array_equal(actual_patients, expected_patients):
                                                raise ValueError(f'patient_id/source_index mismatch for split={split!r} in {selection_csv}')
                                            else:
                                                self._indices = selected_indices
                        if max_samples:
                            self._indices = self._indices[:int(max_samples)]
                        self._n_rows = int(self._indices.size)
                    self._label_dict = {v: k for k, v in self._label_mapping.items()}
    @property
    def num_classes(self) -> int:
        return max(self._label_dict) + 1 if self._label_dict else 0
    @property
    def label_dict(self) -> dict[int, str]:
        return self._label_dict
    def getlabel(self, idx: int) -> int:
        with h5py.File(self.h5_path, 'r') as handle:
            return int(handle['labels'][int(self._indices[idx])])
    def __len__(self) -> int:
        return self._n_rows
    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = int(self._indices[idx])
        with h5py.File(self.h5_path, 'r') as handle:
            features = handle['features']
            if self.feature_layer:
                if self.feature_layer not in features:
                    raise KeyError(f'feature_layer={self.feature_layer!r} not found in {self.h5_path}; available={list(features.keys())}')
                else:
                    x = torch.as_tensor(features[self.feature_layer][source_idx:source_idx + 1]).float()
            else:
                if 'features' in features:
                    x = torch.as_tensor(features['features'][source_idx:source_idx + 1]).float()
                else:
                    if len(features.keys()) == 1:
                        key = next(iter(features.keys()))
                        x = torch.as_tensor(features[key][source_idx:source_idx + 1]).float()
                    else:
                        raise KeyError(f'feature_layer must be set for {self.h5_path}; available={list(features.keys())}')
            label = int(handle['labels'][source_idx])
            sample_id = handle['sample_ids'][source_idx].decode('utf-8')
            case_id = handle['case_ids'][source_idx].decode('utf-8')
            label_name = handle['label_names'][source_idx].decode('utf-8')
        if self.feature_normalize == 'l2':
            x = F.normalize(x, p=2, dim=1)
        return {'features': x, 'label': torch.tensor([label], dtype=torch.long), 'sample_id': sample_id, 'case_id': case_id, 'label_name': label_name}
    def tensors(self, *, include_rows: bool=True) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        with h5py.File(self.h5_path, 'r') as handle:
            features = handle['features']
            if self.feature_layer:
                if self.feature_layer not in features:
                    raise KeyError(f'feature_layer={self.feature_layer!r} not found in {self.h5_path}; available={list(features.keys())}')
                else:
                    feature_data = features[self.feature_layer]
            else:
                if 'features' in features:
                    feature_data = features['features']
                else:
                    if len(features.keys()) == 1:
                        key = next(iter(features.keys()))
                        feature_data = features[key]
                    else:
                        raise KeyError(f'feature_layer must be set for {self.h5_path}; available={list(features.keys())}')
            x = torch.empty((self._n_rows, int(feature_data.shape[1])), dtype=torch.float32)
            y = torch.empty(self._n_rows, dtype=torch.long)
            chunk_size = 65536
            output_start = 0
            for source_start in range(0, self._source_rows, chunk_size):
                source_stop = min(source_start + chunk_size, self._source_rows)
                left = int(np.searchsorted(self._indices, source_start, side='left'))
                right = int(np.searchsorted(self._indices, source_stop, side='left'))
                if left == right:
                    continue
                else:
                    local_indices = self._indices[left:right] - source_start
                    feature_chunk = torch.as_tensor(feature_data[source_start:source_stop]).float()
                    label_chunk = torch.as_tensor(handle['labels'][source_start:source_stop], dtype=torch.long)
                    count = right - left
                    x[output_start:output_start + count].copy_(feature_chunk[local_indices])
                    y[output_start:output_start + count].copy_(label_chunk[local_indices])
                    output_start += count
            if output_start != self._n_rows:
                raise RuntimeError(f'Loaded {output_start} rows from {self.h5_path}, expected {self._n_rows}')
            else:
                if include_rows:
                    complete_selection = self._n_rows == self._source_rows and np.array_equal(self._indices, np.arange(self._source_rows))
                    selection = slice(None) if complete_selection else self._indices
                    sample_values = handle['sample_ids'][selection]
                    case_values = handle['case_ids'][selection]
                    label_values = handle['label_names'][selection]
                    sample_ids = [value.decode('utf-8') for value in sample_values]
                    case_ids = [value.decode('utf-8') for value in case_values]
                    label_names = [value.decode('utf-8') for value in label_values]
                else:
                    sample_ids = case_ids = label_names = []
        if self.feature_normalize == 'l2':
            norms = x.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-12)
            x.div_(norms)
        rows = [{'sample_id': sample_id, 'case_id': case_id, 'feature_index': 0, 'label': int(label), 'label_name': label_name} for sample_id, case_id, label, label_name in zip(sample_ids, case_ids, y.tolist(), label_names)] if include_rows else []
        return (x, y, rows)
def _survival_labels(frame: pd.DataFrame, *, time_column: str, event_column: str, n_bins: int, event_positive_values: list[int], bin_frame: pd.DataFrame | None=None, bin_boundaries: list[float] | np.ndarray | None=None) -> tuple[pd.DataFrame, np.ndarray]:
    data = frame.copy()
    data[time_column] = pd.to_numeric(data[time_column], errors='coerce')
    data[event_column] = pd.to_numeric(data[event_column], errors='coerce')
    data = data.dropna(subset=[time_column, event_column]).reset_index(drop=True)
    data['event'] = data[event_column].astype(int).isin(event_positive_values).astype(int)
    data['censorship'] = 1 - data['event']
    bin_data = data if bin_frame is None else bin_frame.copy()
    bin_data[time_column] = pd.to_numeric(bin_data[time_column], errors='coerce')
    bin_data[event_column] = pd.to_numeric(bin_data[event_column], errors='coerce')
    bin_data = bin_data.dropna(subset=[time_column, event_column]).reset_index(drop=True)
    bin_data['event'] = bin_data[event_column].astype(int).isin(event_positive_values).astype(int)
    if bin_boundaries is None:
        uncensored = bin_data[bin_data['event'] == 1]
        if len(uncensored) < n_bins:
            raise ValueError(f'Need at least {n_bins} uncensored patients to build survival bins')
        else:
            _, bins = pd.qcut(uncensored[time_column], q=n_bins, retbins=True, labels=False, duplicates='drop')
            if len(bins) - 1 != n_bins:
                raise ValueError(f'Could not build {n_bins} unique survival bins from {time_column}')
            else:
                bins[0] = bin_data[time_column].min() - 1e-06
                bins[(-1)] = bin_data[time_column].max() + 1e-06
    else:
        bins = np.asarray(bin_boundaries, dtype=float).copy()
        if len(bins) != n_bins + 1 or not np.all(np.diff(bins) > 0):
            raise ValueError(f'Expected {n_bins + 1} strictly increasing survival bin boundaries')
    cut_times = data[time_column].clip(lower=bins[0], upper=np.nextafter(bins[(-1)], -np.inf))
    data['disc_label'] = pd.cut(cut_times, bins=bins, labels=False, right=False, include_lowest=True).astype(int)
    return (data, bins)
class SurvivalFeatureDataset(Dataset):
    """Patient-level WSI survival MIL bags backed by extracted feature tensors."""
    task_type = 'survival'
    def __init__(self, feature_experiment_config: str | Path, split: str, *, split_csv: str | Path, feature_layer: str | None=None, case_column: str='case_id', slide_column: str='slide_id', split_column: str='split', time_column: str='PFI.time.1', event_column: str='PFI.1', n_bins: int=4, event_positive_values: list[int] | None=None, bin_split: str='train', bin_boundaries: list[float] | np.ndarray | None=None, permute_training_outcomes: bool=False, outcome_permutation_seed: int=42, feature_layers: list[str] | None=None, feature_normalize: str | None=None, feature_norm_mean: torch.Tensor | list[float] | None=None, feature_norm_std: torch.Tensor | list[float] | None=None, max_instances_per_bag: int | None=None, deterministic_instance_order: str='coordinate', max_samples: int | None=None, **_: Any):
        self.exp = load_experiment_config(feature_experiment_config)
        self.split = split
        self.feature_layer = feature_layer or (feature_layers[0] if feature_layers else None)
        self.feature_normalize = feature_normalize
        self.feature_norm_mean = torch.as_tensor(feature_norm_mean, dtype=torch.float32) if feature_norm_mean is not None else None
        self.feature_norm_std = torch.as_tensor(feature_norm_std, dtype=torch.float32) if feature_norm_std is not None else None
        self.n_bins = int(n_bins)
        self.max_instances_per_bag = int(max_instances_per_bag) if max_instances_per_bag is not None else None
        self.deterministic_instance_order = str(deterministic_instance_order)
        self.time_column = time_column
        self.event_column = event_column
        positives = [1] if event_positive_values is None else [int(x) for x in event_positive_values]
        if permute_training_outcomes and split != bin_split:
            raise ValueError('Diagnostic survival outcome permutation is permitted only for the training split')
        else:
            frame = pd.read_csv(split_csv)
            split_frame = frame[frame[split_column] == split].copy()
            all_patients = frame.drop_duplicates(case_column).reset_index(drop=True)
            bin_patients = frame[frame[split_column] == bin_split].drop_duplicates(case_column).reset_index(drop=True)
            all_labels, self.bins = _survival_labels(all_patients, time_column=time_column, event_column=event_column, n_bins=self.n_bins, event_positive_values=positives, bin_frame=bin_patients, bin_boundaries=bin_boundaries)
            split_cases = set(split_frame[case_column].astype(str))
            labels = all_labels[all_labels[case_column].astype(str).isin(split_cases)].copy()
            self.outcome_permutation = {'enabled': False}
            if permute_training_outcomes:
                labels = labels.sort_values(case_column).reset_index(drop=True)
                rng = np.random.default_rng(int(outcome_permutation_seed))
                permutation = rng.permutation(len(labels))
                outcome_columns = list(dict.fromkeys([time_column, event_column, 'event', 'censorship', 'disc_label']))
                original_cases = labels[case_column].astype(str).tolist()
                source_cases = labels.iloc[permutation][case_column].astype(str).tolist()
                labels.loc[:, outcome_columns] = labels.iloc[permutation][outcome_columns].to_numpy()
                self.outcome_permutation = {'enabled': True, 'split': split, 'seed': int(outcome_permutation_seed), 'n_patients': len(labels), 'n_changed': int(sum((a != b for a, b in zip(original_cases, source_cases)))), 'case_to_outcome_source': dict(zip(original_cases, source_cases))}
            self.rows = []
            for _, row in labels.iterrows():
                case_id = str(row[case_column])
                slides = split_frame.loc[split_frame[case_column] == case_id, slide_column].dropna()
                feature_paths = [path for slide in slides if (path := _feature_data_path(self.exp, str(slide), split)).is_file()]
                if not feature_paths:
                    continue
                else:
                    self.rows.append({'case_id': case_id, 'feature_paths': feature_paths, 'label': int(row['disc_label']), 'event_time': float(row[time_column]), 'censorship': float(row['censorship'])})
            if max_samples:
                self.rows = self.rows[:int(max_samples)]
            if not self.rows:
                raise FileNotFoundError(f'No survival feature bags found for split=\'{split}\' using {self.exp.id}')
    @property
    def num_classes(self) -> int:
        return self.n_bins
    @property
    def label_dict(self) -> dict[int, str]:
        return {i: f'bin_{i}' for i in range(self.n_bins)}
    def getlabel(self, idx: int) -> int:
        row = self.rows[idx]
        return row['label'] * 2 + int(row['censorship'])
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {'features': self._load_row_bag(row), 'label': torch.tensor(row['label'], dtype=torch.long), 'event_time': float(row['event_time']), 'censorship': torch.tensor(row['censorship'], dtype=torch.float32), 'case_id': row['case_id']}
    def _load_row_bag(self, row: dict[str, Any], *, feature_normalize: str | None=None) -> torch.Tensor:
        mode = self.feature_normalize if feature_normalize is None else feature_normalize
        return _load_mil_bag(row['feature_paths'], [str(self.feature_layer)], feature_normalize=mode, norm_mean=self.feature_norm_mean if mode else None, norm_std=self.feature_norm_std if mode else None, max_instances=self.max_instances_per_bag, deterministic_order=self.deterministic_instance_order)
class ClassificationFeatureDataset(Dataset):
    # ***<module>.ClassificationFeatureDataset: Failure detected at line number 480 and instruction offset 36: Different bytecode
    """Patient/slide-level WSI classification MIL bags from extracted features."""
    task_type = 'classification'
    def __init__(self, feature_experiment_config: str | Path, split: str, *, split_csv: str | Path, feature_layer: str | None=None, case_column: str='case_id', slide_column: str='slide_id', split_column: str='split', label_column: str='label', label_mapping: dict[str, int] | None=None, feature_layers: list[str] | None=None, feature_normalize: str | None=None, feature_norm_mean: torch.Tensor | list[float] | None=None, feature_norm_std: torch.Tensor | list[float] | None=None, feature_norm_clip_value: float | None=None, strict_integrity: bool=False, max_instances_per_bag: int | None=None, deterministic_instance_order: str='coordinate', max_samples: int | None=None, **_: Any):
        self.exp = load_experiment_config(feature_experiment_config)
        self.split = split
        if feature_layers:
            self.feature_layers = [str(name) for name in feature_layers]
        else:
            if feature_layer:
                self.feature_layers = [str(feature_layer)]
            else:
                self.feature_layers = list(self.exp.target_features)
        self.feature_layer = feature_layer or (self.feature_layers[0] if len(self.feature_layers) == 1 else None)
        self.feature_normalize = feature_normalize
        self.feature_norm_mean = torch.as_tensor(feature_norm_mean, dtype=torch.float32) if feature_norm_mean is not None else None
        self.feature_norm_std = torch.as_tensor(feature_norm_std, dtype=torch.float32) if feature_norm_std is not None else None
        self.feature_norm_clip_value = float(feature_norm_clip_value) if feature_norm_clip_value is not None else None
        self.strict_integrity = bool(strict_integrity)
        self.max_instances_per_bag = int(max_instances_per_bag) if max_instances_per_bag is not None else None
        self.deterministic_instance_order = str(deterministic_instance_order)
        mapping = label_mapping or {}
        self._label_dict = {int(v): str(k) for k, v in mapping.items()}
        frame = pd.read_csv(split_csv)
        split_frame = frame[frame[split_column] == split].copy()
        csv_patients = split_frame.drop_duplicates(case_column)
        skipped = []
        self.rows = []
        for _, row in csv_patients.iterrows():
            case_id = str(row[case_column])
            raw_label = str(row[label_column])
            patient_rows = split_frame.loc[split_frame[case_column] == case_id]
            patient_labels = patient_rows[label_column].dropna().astype(str).unique().tolist()
            if len(patient_labels) != 1:
                message = f'inconsistent patient labels {patient_labels}'
                if self.strict_integrity:
                    raise ValueError(f'{case_id}: {message}')
                else:
                    skipped.append({'case_id': case_id, 'reason': message})
            else:
                if raw_label not in mapping:
                    skipped.append({'case_id': case_id, 'reason': f'unmapped label \'{raw_label}\''})
                else:
                    slides = patient_rows[slide_column].dropna().astype(str).drop_duplicates()
                    feature_paths = [path for slide in slides if (path := _feature_data_path(self.exp, slide, split)).is_file()]
                    if self.strict_integrity and len(feature_paths) != len(slides):
                        missing = [slide for slide in slides if not _feature_data_path(self.exp, slide, split).is_file()]
                        raise FileNotFoundError(f'{case_id}: missing feature files for slides={missing} in split={split}')
                    else:
                        if not feature_paths:
                            skipped.append({'case_id': case_id, 'reason': 'missing feature file'})
                        else:
                            self.rows.append({'case_id': case_id, 'feature_paths': feature_paths, 'label': int(mapping[raw_label]), 'label_name': raw_label})
        self.audit = {'split': split, 'csv_patients': int(len(csv_patients)), 'loaded_patients': int(len(self.rows)), 'skipped_patients': skipped}
        if max_samples:
            self.rows = self.rows[:int(max_samples)]
        if not self.rows:
            raise FileNotFoundError(f'No classification feature bags found for split=\'{split}\' using {self.exp.id}')
    @property
    def num_classes(self) -> int:
        if self._label_dict:
            return max(self._label_dict) + 1
        else:
            return 2
    @property
    def label_dict(self) -> dict[int, str]:
        return self._label_dict
    def getlabel(self, idx: int) -> int:
        return int(self.rows[idx]['label'])
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        bag = self._load_row_bag(row)
        if self.strict_integrity:
            if bag.ndim != 2 or bag.shape[0] == 0 or bag.shape[1] == 0:
                raise ValueError(f"{row['case_id']}: empty or invalid bag shape={tuple(bag.shape)}")
            else:
                if not torch.isfinite(bag).all():
                    raise ValueError(f"{row['case_id']}: bag contains NaN or infinite values")
        return {'features': bag, 'label': torch.tensor(row['label'], dtype=torch.long), 'case_id': row['case_id'], 'label_name': row['label_name']}
    def _load_row_bag(self, row: dict[str, Any], *, feature_normalize: str | None=None) -> torch.Tensor:
        mode = self.feature_normalize if feature_normalize is None else feature_normalize
        return _load_mil_bag(row['feature_paths'], self.feature_layers, feature_normalize=mode, norm_mean=self.feature_norm_mean if mode else None, norm_std=self.feature_norm_std if mode else None, norm_clip_value=self.feature_norm_clip_value, max_instances=self.max_instances_per_bag, deterministic_order=self.deterministic_instance_order)
def collate_mil_bag(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError('MIL uses batch_size=1 because WSI bags have variable sizes')
    else:
        return batch[0]
DATASET_REGISTRY = {'linear_probe_features': LinearProbeFeatureDataset, 'survival_features': SurvivalFeatureDataset, 'classification_features': ClassificationFeatureDataset}

# Official feature layouts are split by train/valid/test and may be nested in folds.
def _pt_h5(split_dir: Path, model: str, sample_id: str) -> tuple[Path, Path]:
    return (
        split_dir / "pt_files" / model / f"{sample_id}.pt",
        split_dir / "h5_files" / model / f"{sample_id}.h5",
    )

def _sample_id_aliases(sample_id) -> list[str]:
    raw = str(sample_id)
    aliases = [raw]
    if raw.endswith("_HE"):
        stem = raw[:-3]
        aliases.extend((stem, stem.lstrip("0") or "0"))
    elif raw.isdigit():
        aliases.extend((raw.zfill(3), f"{raw.zfill(3)}_HE"))
    return list(dict.fromkeys(aliases))

def _feature_paths(exp, sample_id, split):
    model = str(getattr(exp, "model"))
    split_name = str(split).lower()
    split_name = {"training": "train", "val": "valid"}.get(split_name, split_name)
    feat_dir = Path(getattr(exp, "feat_dir"))

    split_dirs = [feat_dir / split_name]
    if split_name not in {"train", "test"}:
        split_dirs.append(feat_dir / "test")
    fold_dirs = sorted(feat_dir.glob("fold_*"))

    for sid in _sample_id_aliases(sample_id):
        for split_dir in split_dirs:
            paths = _pt_h5(split_dir, model, sid)
            if paths[1].is_file() or paths[0].is_file():
                return paths
        for fold_dir in fold_dirs:
            paths = _pt_h5(fold_dir / split_name, model, sid)
            if paths[1].is_file() or paths[0].is_file():
                return paths
        if split_name == "valid":
            for fold_dir in fold_dirs:
                paths = _pt_h5(fold_dir / "test", model, sid)
                if paths[1].is_file() or paths[0].is_file():
                    return paths
    return _pt_h5(feat_dir / (split_name or "train"), model, str(sample_id))

# Survival training re-reads the same patch-h5 bag every epoch. Cache float16
# on CPU after the first load. Opt out with HEMIF_SURVIVAL_BAG_CACHE=0.
_SURVIVAL_BAG_CACHE_ENABLED = os.environ.get("HEMIF_SURVIVAL_BAG_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_ORIG_SURVIVAL_GETITEM = SurvivalFeatureDataset.__getitem__


def _clone_cached_survival_item(item: dict):
    cloned = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            tensor = value.float() if key == "features" else value
            cloned[key] = tensor.clone()
        else:
            cloned[key] = value
    return cloned


def _survival_getitem_cached(self, idx: int):
    if not _SURVIVAL_BAG_CACHE_ENABLED:
        return _ORIG_SURVIVAL_GETITEM(self, idx)
    cache = self.__dict__.get("_bag_cache")
    if cache is None:
        cache = {}
        self._bag_cache = cache
    cached = cache.get(idx)
    if cached is not None:
        return _clone_cached_survival_item(cached)
    item = _ORIG_SURVIVAL_GETITEM(self, idx)
    stored = {}
    for key, value in item.items():
        if key == "features" and torch.is_tensor(value):
            stored[key] = value.detach().to(dtype=torch.float16, device="cpu").contiguous()
        elif torch.is_tensor(value):
            stored[key] = value.detach().to("cpu")
        else:
            stored[key] = value
    cache[idx] = stored
    return item


SurvivalFeatureDataset.__getitem__ = _survival_getitem_cached
