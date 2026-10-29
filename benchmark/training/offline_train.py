# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'training/offline_train.py'
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2026-07-21 12:01:22 UTC (1784635282)

"""Train downstream heads from YAML configs."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import random
import traceback
from pathlib import Path
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from training.config import EvalConfig, load_eval_config
from data.feature_bags import DATASET_REGISTRY, collate_mil_bag
from heads import HEAD_REGISTRY, build_segmentation_head, forward_segmentation_head, normalize_head_name, resolve_decoder_mode
from losses import ClassExcludedMeanLoss, EMAInverseClassFrequencyLoss, FocalLoss, NLLSurvLoss
from metrics import SegmentationMetricAccumulator, classification_metrics, concordance_index, concordance_index_bootstrap_ci, kaplan_meier_by_risk_group, linear_probe_metrics, logrank_by_median_risk, tune_binary_threshold
AGGREGATE_METRIC_KEYS = ['accuracy', 'f1_macro', 'f1_micro', 'dice_macro', 'dice_micro', 'iou_macro', 'iou_micro', 'foreground_iou', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'c_index', 'c_index_ci_low', 'c_index_ci_high', 'c_index_bootstrap_n', 'logrank_chi2', 'logrank_p', 'n_patients', 'n_events', 'risk_group_low_n', 'risk_group_high_n', 'auc', 'lin_acc', 'lin_bacc', 'lin_macro_f1', 'lin_weighted_f1', 'lin_auroc', 'loss']
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def _rng_state() -> dict[str, Any]:
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state
def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    else:
        if 'python' in state:
            random.setstate(state['python'])
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda'])
def _configure_torch_threads(cfg: EvalConfig, log_path: Path) -> None:
    num_threads = cfg.training.get('torch_num_threads')
    if num_threads is None:
        return
    else:
        num_threads = int(num_threads)
        torch.set_num_threads(num_threads)
        _log(f'torch_num_threads={num_threads}', log_path)
def _move_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in features.items()}
def _is_survival(cfg: EvalConfig) -> bool:
    return cfg.task == 'prognosis_survival'
def _is_classification(cfg: EvalConfig) -> bool:
    return cfg.task == 'treatment_response'
def _is_linear_probe(cfg: EvalConfig) -> bool:
    return cfg.task == 'single_cell_classification'
def _is_mil(cfg: EvalConfig) -> bool:
    return _is_survival(cfg) or _is_classification(cfg)
def _validate_task_config(cfg: EvalConfig) -> None:
    expected = {'prognosis_survival': {'dataset': 'survival_features', 'heads': {'survival_attention_mil', 'shazam_survival_abmil'}}, 'treatment_response': {'dataset': 'classification_features', 'heads': {'mean_pool_mlp_mil', 'classification_attention_mil', 'max_pool_linear_mil', 'mean_pool_linear_mil'}}, 'single_cell_classification': {'dataset': 'linear_probe_features', 'head': 'linear_probe'}, 'cell_semantic_segmentation': {'dataset': 'pathocell_online', 'head': 'unetr', 'decoder_modes': {'pfmb', 'feature'}}, 'tls_region_segmentation': {'dataset': 'hooknet_tls_online', 'head': 'unetr', 'decoder_mode': 'feature'}}
    if cfg.task not in expected:
        raise ValueError(f'Unknown evaluation task \'{cfg.task}\'')
    else:
        spec = expected[cfg.task]
        if cfg.dataset != spec['dataset']:
            raise ValueError(f"Task \'{cfg.task}\' expects dataset=\'{spec['dataset']}\', got dataset=\'{cfg.dataset}\'")
        else:
            expected_head = spec.get('head')
            expected_heads = spec.get('heads')
            if expected_heads is not None and normalize_head_name(cfg.head) not in {normalize_head_name(name) for name in expected_heads}:
                raise ValueError(f'Task \'{cfg.task}\' expects head in {sorted(expected_heads)}, got head=\'{cfg.head}\'')
            else:
                if expected_head is not None and normalize_head_name(cfg.head) != normalize_head_name(expected_head):
                    raise ValueError(f'Task \'{cfg.task}\' expects head=\'{expected_head}\', got head=\'{cfg.head}\'')
                else:
                    expected_mode = spec.get('decoder_mode')
                    expected_modes = spec.get('decoder_modes')
                    if expected_mode is not None or expected_modes is not None:
                        actual_mode = resolve_decoder_mode(cfg.head, cfg.head_config)
                        if expected_modes is not None and actual_mode not in expected_modes:
                            raise ValueError(f'Task \'{cfg.task}\' expects decoder_mode in {sorted(expected_modes)}, got decoder_mode=\'{actual_mode}\' (head={cfg.head!r})')
                        else:
                            if expected_mode is not None and actual_mode != expected_mode:
                                raise ValueError(f'Task \'{cfg.task}\' expects decoder_mode=\'{expected_mode}\', got decoder_mode=\'{actual_mode}\' (head={cfg.head!r})')
                            else:
                                if cfg.task == 'cell_semantic_segmentation':
                                    if actual_mode == 'feature':
                                        processor = cfg.head_config.get('feature_processor')
                                        processor_name = str(processor.get('name', '')).strip().lower() if isinstance(processor, dict) else ''
                                        if processor_name not in {'cellvit', 'cellvit_pyramid'}:
                                            raise ValueError('PathoCell decoder_mode=\'feature\' requires feature_processor.name=\'cellvit_pyramid\'')
def _positive_class_index(cfg: EvalConfig, num_classes: int) -> int:
    value = cfg.data.get('positive_class', cfg.data.get('positive_label'))
    mapping = cfg.data.get('label_mapping') or {}
    if value is None:
        return max(0, num_classes - 1)
    else:
        if isinstance(value, str) and value in mapping:
            index = int(mapping[value])
        else:
            index = int(value)
        if index < 0 or index >= num_classes:
            raise ValueError(f'positive_class index {index} is outside [0, {num_classes})')
        else:
            return index
def _classification_seeds(cfg: EvalConfig) -> list[int]:
    raw = cfg.training.get('seeds')
    if raw is not None:
        return [int(seed) for seed in raw]
    else:
        return [int(cfg.training.get('seed', 42))]
def _cross_validation_cfg(cfg: EvalConfig) -> dict[str, Any]:
    return dict(cfg.data.get('cross_validation') or {})
def _cross_validation_enabled(cfg: EvalConfig) -> bool:
    return bool(_cross_validation_cfg(cfg).get('enabled', False))
def _cross_validation_folds(cfg: EvalConfig) -> list[int]:
    cv = _cross_validation_cfg(cfg)
    folds = cv.get('folds')
    if folds is not None:
        return [int(fold) for fold in folds]
    else:
        n_folds = int(cv.get('n_folds', 5))
        return list(range(n_folds))
def _resolve_cv_split_csv(cfg: EvalConfig, fold: int) -> str:
    cv = _cross_validation_cfg(cfg)
    pattern = cv.get('split_csv_pattern')
    if pattern:
        base_dir = cv.get('split_csv_dir')
        if base_dir:
            return str(Path(str(base_dir)).expanduser().resolve() / str(pattern).format(fold=fold))
        else:
            parent = Path(cfg.data['split_csv']).expanduser().resolve().parent
            return str(parent / str(pattern).format(fold=fold))
    else:
        template = cv.get('split_csv_template')
        if template:
            return str(template).format(fold=fold)
        else:
            raise ValueError('data.cross_validation requires split_csv_pattern or split_csv_template when enabled.')
def _clone_eval_config_for_fold(cfg: EvalConfig, fold: int, *, multicv: bool) -> EvalConfig:
    data = dict(cfg.data)
    data['split_csv'] = _resolve_cv_split_csv(cfg, fold)
    output_dir = cfg.output_dir / f'fold_{fold}' if multicv else cfg.output_dir
    return EvalConfig(id=cfg.id, task=cfg.task, dataset=cfg.dataset, head=cfg.head, output_dir=output_dir, data=data, head_config=dict(cfg.head_config), training=dict(cfg.training), metrics=dict(cfg.metrics), survival=dict(cfg.survival))
def _apply_cli_fold_override(cfg: EvalConfig, fold: int | None) -> EvalConfig:
    if fold is None:
        return cfg
    else:
        if not _cross_validation_enabled(cfg):
            raise ValueError('--fold requires data.cross_validation.enabled in the eval config.')
        else:
            fold = int(fold)
            data = dict(cfg.data)
            cv = dict(_cross_validation_cfg(cfg))
            cv['folds'] = [fold]
            data['cross_validation'] = cv
            data['split_csv'] = _resolve_cv_split_csv(cfg, fold)
            output_dir = cfg.output_dir / f'fold_{fold}'
            return EvalConfig(id=cfg.id, task=cfg.task, dataset=cfg.dataset, head=cfg.head, output_dir=output_dir, data=data, head_config=dict(cfg.head_config), training=dict(cfg.training), metrics=dict(cfg.metrics), survival=dict(cfg.survival))
def _best_valid_summary_metrics(cfg: EvalConfig) -> dict[str, float]:
    path = cfg.output_dir / 'validation_result' / 'best_validation.csv'
    if not path.is_file():
        return {}
    else:
        with path.open(newline='', encoding='utf-8') as handle:
            row = next(csv.DictReader(handle), None)
        if not row:
            return {}
        else:
            out = {}
            for key, value in row.items():
                if key == 'epoch' or not key.startswith('valid_'):
                    continue
                else:
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            if 'epoch' in row:
                out['best_epoch'] = float(row['epoch'])
            return out
def _run_patient_cross_validation(cfg: EvalConfig, run_once: Callable[[EvalConfig], dict[str, Any]], *, seed: int | None=None) -> None:
    """Shared patient-level k-fold orchestration for downstream MIL tasks."""
    folds = _cross_validation_folds(cfg)
    multicv = len(folds) > 1
    resolved_seed = int(seed if seed is not None else _classification_seeds(cfg)[0])
    summaries = []
    for fold in folds:
        run_cfg = _clone_eval_config_for_fold(cfg, fold, multicv=multicv)
        training = dict(run_cfg.training)
        training['seed'] = resolved_seed
        training.pop('seeds', None)
        run_cfg = EvalConfig(id=run_cfg.id, task=run_cfg.task, dataset=run_cfg.dataset, head=run_cfg.head, output_dir=run_cfg.output_dir, data=run_cfg.data, head_config=run_cfg.head_config, training=training, metrics=run_cfg.metrics, survival=run_cfg.survival)
        summary = run_once(run_cfg)
        summary['fold'] = fold
        summaries.append(summary)
    if multicv:
        _write_cv_summary(cfg.output_dir, summaries)
def _write_cv_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_keys = {'fold', 'seed', 'decision_threshold', 'best_epoch', 'threshold_tuning_score'}
        metric_keys = sorted({key for row in summaries for key in row if key not in skip_keys and (not key.startswith('valid_'))})
        valid_keys = sorted({key for row in summaries for key in row if key.startswith('valid_')})
        fieldnames = ['fold', 'seed', 'best_epoch', 'decision_threshold', *valid_keys, *metric_keys]
        summary_path = output_dir / 'cv_summary.csv'
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row.get(key, '') for key in fieldnames})
        aggregate = {'n_folds': float(len(summaries))}
        for key in [*valid_keys, *metric_keys]:
            values = [float(row[key]) for row in summaries if key in row and row[key] != '']
            if values:
                aggregate[f'{key}_mean'] = float(np.mean(values))
                aggregate[f'{key}_std'] = float(np.std(values))
        with (output_dir / 'cv_summary.json').open('w', encoding='utf-8') as handle:
            json.dump({'folds': summaries, 'aggregate': aggregate}, handle, indent=2)
def _clone_eval_config_for_seed(cfg: EvalConfig, seed: int, *, multiseed: bool) -> EvalConfig:
    training = dict(cfg.training)
    training['seed'] = seed
    training.pop('seeds', None)
    output_dir = cfg.output_dir / f'seed_{seed}' if multiseed else cfg.output_dir
    return EvalConfig(id=cfg.id, task=cfg.task, dataset=cfg.dataset, head=cfg.head, output_dir=output_dir, data=dict(cfg.data), head_config=dict(cfg.head_config), training=training, metrics=dict(cfg.metrics), survival=dict(cfg.survival))
def _resolve_decision_threshold(cfg: EvalConfig) -> float | None:
    value = cfg.training.get('decision_threshold')
    if value is None:
        return
    else:
        return float(value)
def _apply_cli_seed_override(cfg: EvalConfig, seed: int | None) -> EvalConfig:
    if seed is None:
        return cfg
    else:
        had_multiseed = cfg.training.get('seeds') is not None
        training = dict(cfg.training)
        training['seed'] = int(seed)
        training.pop('seeds', None)
        output_dir = cfg.output_dir / f'seed_{seed}' if had_multiseed else cfg.output_dir
        return EvalConfig(id=cfg.id, task=cfg.task, dataset=cfg.dataset, head=cfg.head, output_dir=output_dir, data=dict(cfg.data), head_config=dict(cfg.head_config), training=training, metrics=dict(cfg.metrics), survival=dict(cfg.survival))
def _load_decision_threshold_from_disk(cfg: EvalConfig) -> float | None:
    path = cfg.output_dir / 'decision_threshold.json'
    if not path.is_file():
        return _resolve_decision_threshold(cfg)
    else:
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        threshold = float(payload['decision_threshold'])
        cfg.training['decision_threshold'] = threshold
        return threshold
def _write_multi_seed_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_keys = sorted({key for row in summaries for key in row if key not in {'seed', 'threshold_tuning_score', 'decision_threshold'}})
        summary_path = output_dir / 'multi_seed_summary.csv'
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['seed', 'decision_threshold', *metric_keys])
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row.get(key, '') for key in writer.fieldnames})
        aggregate = {'n_seeds': float(len(summaries))}
        for key in metric_keys:
            values = [float(row[key]) for row in summaries if key in row and row[key] != '']
            if values:
                aggregate[f'{key}_mean'] = float(np.mean(values))
                aggregate[f'{key}_std'] = float(np.std(values))
        with (output_dir / 'multi_seed_summary.json').open('w', encoding='utf-8') as handle:
            json.dump({'seeds': summaries, 'aggregate': aggregate}, handle, indent=2)
def _threshold_tuning_split(cfg: EvalConfig) -> str | None:
    """Return which split to use for threshold calibration: train, valid, or None."""
    if bool(cfg.training.get('tune_threshold_on_train', False)):
        return 'train'
    else:
        if bool(cfg.training.get('tune_threshold_on_valid', False)):
            return 'valid'
def _tune_classification_threshold(cfg: EvalConfig, model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int, train_log: Path, *, split_name: str) -> float | None:
    if num_classes != 2:
        _log('Threshold tuning skipped: only supported for binary classification.', train_log)
        return _resolve_decision_threshold(cfg)
    else:
        split_metrics = evaluate_classification_model(model, loader, criterion, device, num_classes, cfg, decision_threshold=None)
        rows = split_metrics.get('rows', [])
        if not rows:
            _log(f'Threshold tuning skipped: empty {split_name} set.', train_log)
            return _resolve_decision_threshold(cfg)
        else:
            labels = np.asarray([int(row['label']) for row in rows], dtype=int)
            positive_idx = _positive_class_index(cfg, num_classes)
            positive_probs = np.asarray([float(row[f'probability_class_{positive_idx}']) for row in rows], dtype=float)
            metric_name = str(cfg.training.get('threshold_tuning_metric', 'f1_macro'))
            t_min = float(cfg.training.get('threshold_tuning_min', 0.05))
            t_max = float(cfg.training.get('threshold_tuning_max', 0.95))
            grid = int(cfg.training.get('threshold_tuning_grid', 91))
            threshold, score = tune_binary_threshold(labels, positive_probs, metric=metric_name, t_min=t_min, t_max=t_max, grid=grid)
            cfg.training['decision_threshold'] = threshold
            payload = {'decision_threshold': threshold, 'threshold_tuning_split': split_name, 'threshold_tuning_metric': metric_name, 'threshold_tuning_min': t_min, 'threshold_tuning_max': t_max, 'threshold_tuning_grid': grid, 'threshold_tuning_score': score, f'{split_name}_size': int(labels.size)}
            if split_name == 'valid':
                payload['validation_size'] = int(labels.size)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            with (cfg.output_dir / 'decision_threshold.json').open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
            _log(f'Tuned decision threshold={threshold:.4f} on {split_name} ({metric_name}={score:.4f}, n={labels.size})', train_log)
            return threshold
def _feature_norm_settings(cfg: EvalConfig) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
    mode = cfg.training.get('feature_normalize')
    mean = cfg.training.get('feature_norm_mean')
    std = cfg.training.get('feature_norm_std')
    if mean is None and std is None:
            norm_path = cfg.output_dir / 'feature_norm.json'
            if norm_path.is_file():
                with norm_path.open(encoding='utf-8') as handle:
                    payload = json.load(handle)
                mode = payload.get('feature_normalize', mode)
                mean = payload.get('feature_norm_mean')
                std = payload.get('feature_norm_std')
    mean_t = torch.as_tensor(mean, dtype=torch.float32) if mean is not None else None
    std_t = torch.as_tensor(std, dtype=torch.float32) if std is not None else None
    if mode is None:
        return (None, mean_t, std_t)
    else:
        return (str(mode), mean_t, std_t)
def _classification_norm_settings(cfg: EvalConfig) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
    return _feature_norm_settings(cfg)
def _prepare_classification_datasets(cfg: EvalConfig) -> tuple[Any, Any, dict[str, Any]]:
    from data.feature_bags import ClassificationFeatureDataset, fit_classification_feature_norm_stats
    train_split = str(cfg.data.get('train_split', 'train'))
    valid_split = str(cfg.data.get('valid_split', 'valid'))
    norm_mode, mean_t, std_t = _classification_norm_settings(cfg)
    train_dataset = _make_dataset(cfg, split=train_split)
    if not isinstance(train_dataset, ClassificationFeatureDataset):
        raise TypeError('feature normalization requires ClassificationFeatureDataset')
    else:
        if norm_mode in {'zscore', 'zscore_train'} and mean_t is None:
            norm_weighting = str(cfg.training.get('feature_norm_weighting', 'instance'))
            mean_t, std_t = fit_classification_feature_norm_stats(train_dataset, weighting=norm_weighting)
            cfg.training['feature_norm_mean'] = mean_t.tolist()
            cfg.training['feature_norm_std'] = std_t.tolist()
            train_dataset.feature_normalize = norm_mode
            train_dataset.feature_norm_mean = mean_t
            train_dataset.feature_norm_std = std_t
        else:
            if norm_mode and norm_mode not in {'false', 'off', '0', '', 'none'}:
                    train_dataset.feature_normalize = norm_mode
        if train_dataset.feature_normalize:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            with (cfg.output_dir / 'feature_norm.json').open('w', encoding='utf-8') as handle:
                json.dump({'feature_normalize': train_dataset.feature_normalize, 'feature_norm_weighting': str(cfg.training.get('feature_norm_weighting', 'instance')), 'feature_norm_mean': train_dataset.feature_norm_mean.tolist() if train_dataset.feature_norm_mean is not None else None, 'feature_norm_std': train_dataset.feature_norm_std.tolist() if train_dataset.feature_norm_std is not None else None, 'feature_norm_clip_value': cfg.training.get('feature_norm_clip_value')}, handle, indent=2)
        valid_dataset = _make_dataset(cfg, split=valid_split)
        if isinstance(valid_dataset, ClassificationFeatureDataset):
            valid_dataset.feature_normalize = train_dataset.feature_normalize
            valid_dataset.feature_norm_mean = train_dataset.feature_norm_mean
            valid_dataset.feature_norm_std = train_dataset.feature_norm_std
        audit = {'experiment_id': cfg.id, 'feature_experiment': str(cfg.data.get('feature_experiment_config', '')), 'feature_layer': cfg.data.get('feature_layer'), 'feature_layers': cfg.data.get('feature_layers') or getattr(train_dataset, 'feature_layers', None), 'feature_normalize': train_dataset.feature_normalize, 'feature_norm_clip_value': cfg.training.get('feature_norm_clip_value'), 'splits': {train_split: getattr(train_dataset, 'audit', {}), valid_split: getattr(valid_dataset, 'audit', {})}}
        return (train_dataset, valid_dataset, audit)
def _prepare_survival_datasets(cfg: EvalConfig) -> tuple[Any, Any, dict[str, Any]]:
    from data.feature_bags import SurvivalFeatureDataset, fit_survival_feature_norm_stats
    train_split = str(cfg.data.get('train_split', 'train'))
    valid_split = str(cfg.data.get('valid_split', 'valid'))
    norm_mode, mean_t, std_t = _feature_norm_settings(cfg)
    train_dataset = _make_dataset(cfg, split=train_split)
    if not isinstance(train_dataset, SurvivalFeatureDataset):
        raise TypeError('feature normalization requires SurvivalFeatureDataset')
    else:
        fitted_bins = [float(value) for value in train_dataset.bins]
        cfg.survival['bin_boundaries'] = fitted_bins
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.output_dir / 'survival_bins.json').open('w', encoding='utf-8') as handle:
            json.dump({'fit_split': train_split, 'time_column': cfg.survival.get('time_column', 'PFI.time.1'), 'event_column': cfg.survival.get('event_column', 'PFI.1'), 'event_positive_values': cfg.survival.get('event_positive_values', [1]), 'n_bins': int(cfg.survival.get('n_bins', 4)), 'boundaries': fitted_bins}, handle, indent=2)
        if norm_mode in {'zscore', 'zscore_train'} and mean_t is None:
            mean_t, std_t = fit_survival_feature_norm_stats(train_dataset)
            cfg.training['feature_norm_mean'] = mean_t.tolist()
            cfg.training['feature_norm_std'] = std_t.tolist()
            train_dataset.feature_normalize = norm_mode
            train_dataset.feature_norm_mean = mean_t
            train_dataset.feature_norm_std = std_t
        else:
            if norm_mode and norm_mode not in {'false', 'off', '0', '', 'none'}:
                    train_dataset.feature_normalize = norm_mode
        if train_dataset.feature_normalize:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            with (cfg.output_dir / 'feature_norm.json').open('w', encoding='utf-8') as handle:
                json.dump({'feature_normalize': train_dataset.feature_normalize, 'feature_norm_mean': train_dataset.feature_norm_mean.tolist() if train_dataset.feature_norm_mean is not None else None, 'feature_norm_std': train_dataset.feature_norm_std.tolist() if train_dataset.feature_norm_std is not None else None}, handle, indent=2)
        valid_dataset = _make_dataset(cfg, split=valid_split)
        if isinstance(valid_dataset, SurvivalFeatureDataset):
            valid_dataset.feature_normalize = train_dataset.feature_normalize
            valid_dataset.feature_norm_mean = train_dataset.feature_norm_mean
            valid_dataset.feature_norm_std = train_dataset.feature_norm_std
        audit = {'experiment_id': cfg.id, 'feature_experiment': str(cfg.data.get('feature_experiment_config', '')), 'feature_layer': cfg.data.get('feature_layer'), 'feature_layers': cfg.data.get('feature_layers'), 'feature_normalize': train_dataset.feature_normalize, 'survival_bin_fit_split': train_split, 'survival_bin_boundaries': fitted_bins, 'diagnostic_outcome_permutation': train_dataset.outcome_permutation, 'splits': {train_split: {'n_patients': len(train_dataset)}, valid_split: {'n_patients': len(valid_dataset)}}}
        return (train_dataset, valid_dataset, audit)
def _write_dataset_audit(cfg: EvalConfig, audit: dict[str, Any], train_log: Path) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / 'dataset_audit.json').open('w', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2)
    for split_name, split_audit in audit.get('splits', {}).items():
        _log(f"dataset_audit split={split_name} csv_patients={split_audit.get('csv_patients')} loaded={split_audit.get('loaded_patients')} skipped={len(split_audit.get('skipped_patients', []))}", train_log)
        for item in split_audit.get('skipped_patients', []):
            _log(f"dataset_audit skipped case_id={item.get('case_id')} reason={item.get('reason')}", train_log)
def _recall_metric_columns(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if key.startswith('class_') and key.endswith('/recall'):
                cls_id = key.split('/')[0].replace('class_', '')
                out[f'{prefix}class_{cls_id}_recall'] = float(value)
    return out
def _make_dataset(
    cfg: EvalConfig,
    split: str,
    max_samples: int | None = None,
    *,
    feature_layers: list[str] | None = None,
):
    """Build the configured offline feature dataset for one split."""
    dataset_cls = DATASET_REGISTRY[cfg.dataset]
    data = dict(cfg.data)
    override_layers = feature_layers or data.get("feature_layers") or cfg.head_config.get("feature_layers")
    if override_layers is not None:
        data["feature_layers"] = list(override_layers)

    if _is_survival(cfg):
        norm_mode, mean, std = _feature_norm_settings(cfg)
        survival = cfg.survival
        return dataset_cls(
            feature_experiment_config=data["feature_experiment_config"],
            split=split,
            split_csv=data["split_csv"],
            feature_layer=data.get("feature_layer"),
            feature_layers=data.get("feature_layers"),
            case_column=data.get("case_column", "case_id"),
            slide_column=data.get("slide_column", "slide_id"),
            split_column=data.get("split_column", "split"),
            time_column=survival.get("time_column", "PFI.time.1"),
            event_column=survival.get("event_column", "PFI.1"),
            n_bins=int(survival.get("n_bins", 4)),
            event_positive_values=survival.get("event_positive_values", [1]),
            bin_split=str(data.get("train_split", "train")),
            bin_boundaries=survival.get("bin_boundaries"),
            permute_training_outcomes=bool(survival.get("permute_training_outcomes", False)),
            outcome_permutation_seed=int(survival.get("outcome_permutation_seed", 42)),
            feature_normalize=norm_mode,
            feature_norm_mean=mean,
            feature_norm_std=std,
            max_instances_per_bag=data.get("max_instances_per_bag"),
            deterministic_instance_order=str(data.get("deterministic_instance_order", "coordinate")),
            max_samples=max_samples,
        )

    if _is_classification(cfg):
        norm_mode, mean, std = _classification_norm_settings(cfg)
        return dataset_cls(
            feature_experiment_config=data["feature_experiment_config"],
            split=split,
            split_csv=data["split_csv"],
            feature_layer=data.get("feature_layer"),
            feature_layers=data.get("feature_layers"),
            case_column=data.get("case_column", "case_id"),
            slide_column=data.get("slide_column", "slide_id"),
            split_column=data.get("split_column", "split"),
            label_column=data.get("label_column", "label"),
            label_mapping=data.get("label_mapping"),
            feature_normalize=norm_mode,
            feature_norm_mean=mean,
            feature_norm_std=std,
            feature_norm_clip_value=cfg.training.get("feature_norm_clip_value"),
            strict_integrity=bool(cfg.training.get("strict_data_integrity", False)),
            max_instances_per_bag=data.get("max_instances_per_bag"),
            deterministic_instance_order=str(data.get("deterministic_instance_order", "coordinate")),
            max_samples=max_samples,
        )

    if _is_linear_probe(cfg):
        exclusions = dict(data.get("exclude_case_ids_by_split") or {})
        return dataset_cls(
            feature_experiment_config=data["feature_experiment_config"],
            split=split,
            feature_layer=data.get("feature_layer"),
            feature_layers=data.get("feature_layers"),
            label_mapping=data.get("label_mapping"),
            exclude_case_ids=exclusions.get(split, data.get("exclude_case_ids", [])),
            exclude_image_paths_csv=data.get("exclude_image_paths_csv"),
            feature_normalize=cfg.training.get("feature_normalize", "none"),
            selection_csv=data.get("selection_csv"),
            max_samples=max_samples,
        )
    raise ValueError(
        f"Task {cfg.task!r} is not an offline feature task; use the online trainer."
    )
def _weighted_sampler(dataset, num_classes: int, exclude_classes: list[int]) -> WeightedRandomSampler:
    if getattr(dataset, 'task_type', None) in {'survival', 'classification'}:
        labels = np.array([dataset.getlabel(idx) for idx in range(len(dataset))], dtype=np.int64)
        counts = np.bincount(labels)
        weights = np.array([1.0 / max(counts[label], 1) for label in labels], dtype=np.float64)
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights))
    else:
        raise ValueError(f"Weighted sampler is only implemented for MIL datasets, got task_type={getattr(dataset, 'task_type', None)!r}")
def _make_loader(dataset, cfg: EvalConfig, split: str, shuffle: bool, sampler: WeightedRandomSampler | None=None, log_path: Path | None=None) -> DataLoader:
    training = cfg.training
    batch_size = int(training.get('batch_size', 64))
    if getattr(dataset, 'task_type', None) in {'survival', 'classification'}:
        batch_size = 1
    num_workers = int(training.get('num_workers', 4))
    pin_memory = bool(training.get('pin_memory', torch.cuda.is_available()))
    kwargs = {}
    if num_workers > 0:
        kwargs['prefetch_factor'] = int(training.get('prefetch_factor', 8))
    collate_fn = collate_mil_bag if getattr(dataset, 'task_type', None) in {'survival', 'classification'} else None
    if log_path is not None:
        prefetch = kwargs.get('prefetch_factor', 'disabled')
        _log(f'Creating {split} loader: batch_size={batch_size} num_workers={num_workers} prefetch_factor={prefetch} pin_memory={pin_memory}', log_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory, collate_fn=collate_fn, **kwargs)
def _infer_feature_channels(dataset) -> tuple[list[str], list[int]]:
    layers = list(getattr(dataset, 'feature_layers', None) or [])
    if not layers:
        item = dataset[0]
        layers = list(item['features'])
    item = dataset[0]
    channels = [int(item['features'][name].shape[0]) for name in layers]
    return (layers, channels)
def _make_head(cfg: EvalConfig, dataset) -> nn.Module:
    head_cls = HEAD_REGISTRY[cfg.head]
    if _is_linear_probe(cfg):
        feats, _, _ = dataset.tensors()
        return head_cls(input_dim=int(feats.shape[1]), n_classes=dataset.num_classes)
    else:
        if _is_mil(cfg):
            item = dataset[0]
            return head_cls(input_dim=int(item['features'].shape[1]), n_classes=dataset.num_classes, **dict(cfg.head_config))
        else:
            layers, channels = _infer_feature_channels(dataset)
            if cfg.head in {'feature_unetr', 'unetr'}:
                return build_segmentation_head(cfg.head, feature_layers=layers, in_channels=channels, num_classes=dataset.num_classes, head_config=cfg.head_config)
            else:
                head_cfg = dict(cfg.head_config)
                head_cfg.pop('feature_layers', None)
                return head_cls(feature_layers=layers, in_channels=channels, num_classes=dataset.num_classes, **head_cfg)
def _loss_fn(cfg: EvalConfig, num_classes: int) -> nn.Module:
    if _is_survival(cfg):
        loss_cfg = dict(cfg.training.get('loss_fn') or {})
        return NLLSurvLoss(alpha=float(loss_cfg.get('alpha_surv', cfg.training.get('alpha_surv', 0.0))))
    else:
        if _is_classification(cfg):
            loss_cfg = dict(cfg.training.get('loss_fn') or {})
            params = dict(loss_cfg.get('params') or {})
            weight = params.get('weight')
            if weight is not None and (not isinstance(weight, torch.Tensor)):
                    params['weight'] = torch.as_tensor(weight, dtype=torch.float32)
            return nn.CrossEntropyLoss(**params)
        else:
            if _is_linear_probe(cfg):
                return nn.CrossEntropyLoss()
            else:
                loss_cfg = dict(cfg.training.get('loss_fn') or {})
                loss_name = str(loss_cfg.get('name', 'CrossEntropyLoss'))
                params = dict(loss_cfg.get('params') or {})
                params.setdefault('reduction', 'none')
                exclude_classes = loss_cfg.get('exclude_classes', [0])
                class_weighting = bool(loss_cfg.get('class_weighting', True))
                if loss_name == 'FocalLoss':
                    params.setdefault('gamma', 2.0)
                    params.setdefault('label_smoothing', 0.0)
                    base = FocalLoss(**params)
                    use_ema_wrapper = bool(loss_cfg.get('use_ema_wrapper', False))
                    if not use_ema_wrapper:
                        return ClassExcludedMeanLoss(loss_fn=base, exclude_class=exclude_classes)
                else:
                    params.setdefault('label_smoothing', 0.05)
                    base = nn.CrossEntropyLoss(**params)
                return EMAInverseClassFrequencyLoss(loss_fn=base, exclude_class=exclude_classes, num_classes=num_classes, alpha=float(loss_cfg.get('ema_alpha', 0.99)), class_weighting=class_weighting)
def _make_optimizer(cfg: EvalConfig, model: nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg.training.get('optimizer') or {}
    name = str(optim_cfg.get('name', 'AdamW'))
    params = dict(optim_cfg.get('params') or {})
    params.setdefault('lr', 3e-05)
    params.setdefault('weight_decay', 0.03)
    return getattr(torch.optim, name)(model.parameters(), **params)
def _make_scheduler(cfg: EvalConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    training_steps = int(cfg.training.get('training_steps', 20000))
    warmup_steps = int((cfg.training.get('optimizer') or {}).get('warmup_steps', 1000))
    scheduler_cfg = cfg.training.get('scheduler') or {}
    scheduler_name = str(scheduler_cfg.get('name', 'CosineAnnealingLR'))
    scheduler_params = dict(scheduler_cfg.get('params') or {})
    scheduler_params.setdefault('T_max', training_steps)
    scheduler_params.setdefault('eta_min', 1e-06)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps))
    main_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main_scheduler], milestones=[max(1, warmup_steps)])
def train_one_step(model: nn.Module, batch: dict[str, Any], criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, device: torch.device, grad_clip_norm: float, use_amp: bool) -> float:
    model.train()
    features = _move_features(batch['features'], device)
    masks = batch['semantic_mask'].to(device, non_blocking=True)
    images = batch.get('image')
    if images is not None:
        images = images.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = forward_segmentation_head(model, features, masks, images=images)
        loss = criterion(logits, masks)
    if torch.isnan(loss):
        return float('nan')
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return float(loss.detach().cpu())
@torch.no_grad()
def evaluate_survival_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, cfg: EvalConfig) -> dict[str, Any]:
    # ***<module>.evaluate_survival_model: Failure: Different bytecode
    model.eval()
    risks, event_times, censorships, losses = ([], [], [], [])
    attention_diagnostics = []
    rows = []
    for batch in tqdm(loader, desc='eval', leave=False):
        features = batch['features'].to(device, non_blocking=True)
        label = batch['label'].view(1).to(device)
        censorship = batch['censorship'].view(1).to(device)
        hazards, survival, prediction = model(features)
        attention = _attention_diagnostics(model)
        if attention:
            attention_diagnostics.append(attention)
        loss = criterion(hazards, survival, label, censorship)
        risk = float(-torch.sum(survival, dim=1).detach().cpu().item())
        risks.append(risk)
        event_times.append(float(batch['event_time']))
        censorships.append(float(batch['censorship']))
        losses.append(float(loss.detach().cpu()))
        rows.append({'case_id': batch['case_id'], 'risk': risk, 'disc_label': int(label.detach().cpu().item()), 'prediction': int(prediction.detach().cpu().item()), 'event_time': float(batch['event_time']), 'censorship': float(batch['censorship'])})
    risk_array = np.asarray(risks)
    event_time_array = np.asarray(event_times)
    censorship_array = np.asarray(censorships)
    risk_threshold = cfg.survival.get('risk_threshold')
    threshold_source = str(cfg.survival.get('risk_threshold_source', 'split_median'))
    if risk_threshold is not None:
        risk_threshold = float(risk_threshold)
    logrank = logrank_by_median_risk(event_time_array, risk_array, censorship_array, risk_threshold=risk_threshold)
    risk_threshold = float(logrank.get('risk_threshold', logrank.get('risk_median', 0.0)))
    for row in rows:
        row['risk_group'] = 'high' if float(row['risk']) > risk_threshold else 'low'
    bootstrap_enabled = bool(cfg.survival.get('bootstrap_ci', True))
    bootstrap_metrics = concordance_index_bootstrap_ci(event_time_array, risk_array, censorship_array, n_bootstraps=int(cfg.survival.get('bootstrap_n', 1000)), confidence_level=float(cfg.survival.get('bootstrap_confidence_level', 0.95)), seed=int(cfg.survival.get('bootstrap_seed', cfg.training.get('seed', 42)))) if bootstrap_enabled else {}
    metrics = {'risk_threshold_source': {'loss': float(np.mean(losses)) if losses else 0.0, 'c_index': concordance_index(event_time_array, risk_array, censorship_array), **logrank, **bootstrap_metrics}, 'n_patients': threshold_source, 'n_events': float(len(rows)), 'km_rows': float(np.sum(1 - censorship_array.astype(int))) if len(censorship_array) else 0.0, 'rows': rows}
    if attention_diagnostics:
        for key in attention_diagnostics[0]:
            metrics[key] = float(np.mean([item[key] for item in attention_diagnostics]))
    return metrics
@torch.no_grad()
def evaluate_classification_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int, cfg: EvalConfig, *, decision_threshold: float | None=None) -> dict[str, Any]:
    model.eval()
    labels, predictions, probabilities, losses, attention_entropies = ([], [], [], [], [])
    attention_diagnostics = []
    rows = []
    threshold = _resolve_decision_threshold(cfg) if decision_threshold is None else float(decision_threshold)
    positive_idx = _positive_class_index(cfg, num_classes)
    for batch in tqdm(loader, desc='eval', leave=False):
        features = batch['features'].to(device, non_blocking=True)
        label = batch['label'].view(1).to(device)
        logits, probs, prediction = model(features)
        attention = _attention_diagnostics(model)
        if attention:
            attention_diagnostics.append(attention)
        attention_entropy = _normalized_attention_entropy(model)
        if attention_entropy is not None:
            attention_entropies.append(float(attention_entropy.detach().cpu()))
        loss = criterion(logits, label)
        prob_np = probs.detach().cpu().squeeze(0).numpy()
        label_value = int(label.detach().cpu().item())
        if threshold is not None and num_classes == 2:
            pred_value = int(prob_np[positive_idx] >= threshold)
        else:
            pred_value = int(prediction.detach().cpu().item())
        labels.append(label_value)
        predictions.append(pred_value)
        probabilities.append(prob_np)
        losses.append(float(loss.detach().cpu()))
        rows.append({'case_id': batch['case_id'], 'label': label_value, 'label_name': batch.get('label_name', ''), 'prediction': pred_value, 'probability_positive': float(prob_np[positive_idx]), **{f'probability_class_{idx}': float(prob_np[idx]) for idx in range(num_classes)}})
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions), np.asarray(probabilities), num_classes)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    if attention_entropies:
        metrics['attention_entropy'] = float(np.mean(attention_entropies))
    if attention_diagnostics:
        for key in attention_diagnostics[0]:
            metrics[key] = float(np.mean([item[key] for item in attention_diagnostics]))
    if threshold is not None:
        metrics['decision_threshold'] = float(threshold)
    metrics['rows'] = rows
    return metrics
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int, cfg: EvalConfig) -> dict[str, float]:
    if _is_survival(cfg):
        return evaluate_survival_model(model, loader, criterion, device, cfg)
    else:
        if _is_classification(cfg):
            return evaluate_classification_model(model, loader, criterion, device, num_classes, cfg)
        else:
            model.eval()
            pixel_metrics = SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=int(cfg.training['ignore_index']) if cfg.training.get('ignore_index') is not None else None, exclude_background=bool(cfg.metrics.get('exclude_background', True)))
            total_loss = 0.0
            n_batches = 0
            original_class_weighting = getattr(criterion, 'class_weighting', None)
            if original_class_weighting is not None:
                criterion.class_weighting = False
            for batch in tqdm(loader, desc='eval', leave=False):
                features = _move_features(batch['features'], device)
                masks = batch['semantic_mask'].to(device, non_blocking=True)
                images = batch.get('image')
                if images is not None:
                    images = images.to(device, non_blocking=True)
                logits = forward_segmentation_head(model, features, masks, images=images)
                loss = criterion(logits, masks)
                total_loss += float(loss.detach().cpu())
                n_batches += 1
                pred = logits.argmax(dim=1).detach().cpu().numpy()
                target = batch['semantic_mask'].numpy()
                for i in range(pred.shape[0]):
                    pixel_metrics.update(pred[i], target[i])
            if original_class_weighting is not None:
                criterion.class_weighting = original_class_weighting
            metrics = pixel_metrics.compute()
            metrics['loss'] = total_loss / max(1, n_batches)
            return metrics
def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
def _config_payload(cfg: EvalConfig) -> dict[str, Any]:
    return {'evaluation': {'id': cfg.id, 'task': cfg.task, 'dataset': cfg.dataset, 'head': cfg.head}, 'data': cfg.data, 'head': cfg.head_config, 'training': cfg.training, 'metrics': cfg.metrics, 'survival': cfg.survival, 'run': {'output_dir': str(cfg.output_dir)}}
def _prepare_output_dirs(cfg: EvalConfig) -> dict[str, Path]:
    dirs = {'validation': cfg.output_dir / 'validation_result', 'test': cfg.output_dir / 'test_result', 'logs': cfg.output_dir / 'validation_result' / 'logs'}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(cfg)
    with (cfg.output_dir / 'resolved_config.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    with (cfg.output_dir / 'config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return dirs
def _log(message: str, log_path: Path) -> None:
    print(message)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(message + '\n')
def _load_best_model_if_available(model: nn.Module, cfg: EvalConfig, device: torch.device, log_path: Path, *, expected_feature_layers: list[str] | None=None) -> None:
    checkpoint_path = Path(cfg.training.get('checkpoint_path', cfg.output_dir / 'best_model.pt'))
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_layers = checkpoint.get('feature_layers')
        model_layers = list(getattr(model, 'feature_layers', []))
        if expected_feature_layers and ckpt_layers and (list(ckpt_layers) != list(expected_feature_layers)):
            raise RuntimeError(f'{checkpoint_path} was trained with feature_layers={list(ckpt_layers)}, but this run expects {list(expected_feature_layers)}. Delete best_model.pt and retrain, or restore matching feature_layers in the experiment YAML.')
        else:
            if ckpt_layers and model_layers and (list(ckpt_layers) != model_layers):
                raise RuntimeError(f'{checkpoint_path} feature_layers={list(ckpt_layers)} do not match current model layers={model_layers}. Delete best_model.pt and retrain.')
            else:
                incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(f'{checkpoint_path} is incompatible with the current model (missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}). Delete best_model.pt and retrain with the current feature_layers.')
    else:
        _log('best_model.pt was not created; using final model weights for test evaluation.', log_path)
        torch.save({'model_state_dict': model.state_dict(), 'feature_layers': list(getattr(model, 'feature_layers', []))}, checkpoint_path)
def _save_last_checkpoint(cfg: EvalConfig, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, *, step: int, best_score: float, best_step: int, steps_since_last_best: int, validation_index: int, history: list[dict[str, Any]]) -> None:
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'step': int(step), 'best_score': float(best_score), 'best_step': int(best_step), 'steps_since_last_best': int(steps_since_last_best), 'validation_index': int(validation_index), 'history': history, 'feature_layers': list(getattr(model, 'feature_layers', [])), 'rng_state': _rng_state()}, cfg.output_dir / 'last_checkpoint.pt')
def _resume_last_checkpoint_if_requested(cfg: EvalConfig, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, device: torch.device, log_path: Path) -> dict[str, Any]:
    if not bool(cfg.training.get('resume', False)):
        return {}
    else:
        checkpoint_path = cfg.output_dir / 'last_checkpoint.pt'
        if not checkpoint_path.is_file():
            _log(f'resume=True but no checkpoint found at {checkpoint_path}; starting from scratch.', log_path)
            return {}
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            ckpt_layers = checkpoint.get('feature_layers')
            model_layers = list(getattr(model, 'feature_layers', []))
            if ckpt_layers and model_layers and (list(ckpt_layers) != model_layers):
                raise RuntimeError(f'{checkpoint_path} feature_layers={list(ckpt_layers)} do not match current model layers={model_layers}. Resume requires matching UNETR feature layers.')
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                _restore_rng_state(checkpoint.get('rng_state'))
                _log(f"Resumed training checkpoint from step={int(checkpoint.get('step', (-1)))}.", log_path)
                return checkpoint
def _aggregate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Select the stable summary columns written to validation/test CSVs."""
    aliases = {
        "accuracy": ("accuracy_overall", "accuracy"),
        "f1_macro": ("f1_macro",),
        "f1_score_macro": ("f1_score_macro", "f1_micro"),
        "dice_macro": ("dice_macro",),
        "dice_micro": ("dice_micro",),
        "iou_macro": ("iou_macro",),
        "iou_micro": ("iou_micro",),
        "foreground_iou": ("foreground_iou",),
        "precision_macro": ("precision_macro",),
        "precision_micro": ("precision_micro",),
        "recall_macro": ("recall_macro",),
        "recall_micro": ("recall_micro",),
        "c_index": ("c_index",),
        "c_index_ci_low": ("c_index_ci_low",),
        "c_index_ci_high": ("c_index_ci_high",),
        "c_index_bootstrap_n": ("c_index_bootstrap_n",),
        "logrank_chi2": ("logrank_chi2",),
        "logrank_p": ("logrank_p",),
        "n_patients": ("n_patients",),
        "n_events": ("n_events",),
        "risk_group_low_n": ("risk_group_low_n",),
        "risk_group_high_n": ("risk_group_high_n",),
        "auc": ("auc",),
        "lin_acc": ("lin_acc",),
        "lin_bacc": ("lin_bacc",),
        "lin_macro_f1": ("lin_macro_f1",),
        "lin_weighted_f1": ("lin_weighted_f1",),
        "lin_auroc": ("lin_auroc",),
        "loss": ("loss",),
    }
    return {
        output: float(next((metrics[key] for key in candidates if key in metrics), 0.0))
        for output, candidates in aliases.items()
    }
def _validation_row(validation_index: int, step: int, train_loss: float, metrics: dict[str, float]) -> dict[str, float]:
    row = {'epoch': validation_index, 'step': step, 'train_loss': train_loss}
    row.update(_aggregate_metrics(metrics))
    return row
def _write_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ['epoch', 'step', 'train_loss', *AGGREGATE_METRIC_KEYS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
def _pfmb_classwise_test_rows(metrics: dict[str, float], label_dict: dict[int, str], *, exclude_background: bool) -> list[dict[str, Any]]:
    from metrics import aggregate_pfmb_instance_metrics
    aggregate = aggregate_pfmb_instance_metrics(metrics)
    rows = [{'scope': 'overall', 'class_name': 'ALL', 'class_id': (-1), 'support': metrics.get('instance_count', 0.0), **aggregate}]
    for class_id, class_name in sorted(label_dict.items()):
        if exclude_background and class_id == 0:
                continue
        if f'{class_name}/f1_score' not in metrics:
            continue
        else:
            rows.append({'scope': 'class', 'class_name': class_name, 'class_id': class_id, 'support': metrics.get('instance_count', 0.0), 'f1_score_macro': metrics.get(f'{class_name}/f1_score', 0.0), 'precision_macro': metrics.get(f'{class_name}/precision', 0.0), 'recall_macro': metrics.get(f'{class_name}/recall', 0.0), 'accuracy_macro': metrics.get(f'{class_name}/accuracy', 0.0), 'loss': metrics.get('loss', 0.0)})
    return rows
def _classwise_test_rows(
    metrics: dict[str, float],
    label_dict: dict[int, str],
    *,
    exclude_background: bool,
) -> list[dict[str, Any]]:
    """Create one overall row and one row per observed class."""
    class_ids = [
        int(class_id)
        for class_id in label_dict
        if not exclude_background or int(class_id) != 0
    ]
    support = sum(float(metrics.get(f"class_{class_id}/support", 0.0)) for class_id in class_ids)
    rows: list[dict[str, Any]] = [
        {
            "scope": "overall",
            "class_name": "ALL",
            "class_id": -1,
            "support": support,
            **_aggregate_metrics(metrics),
        }
    ]
    for class_id, class_name in sorted(label_dict.items()):
        class_id = int(class_id)
        if exclude_background and class_id == 0:
            continue
        prefix = f"class_{class_id}"
        class_support = float(metrics.get(f"{prefix}/support", 0.0))
        if class_support <= 0:
            continue
        precision = float(metrics.get(f"{prefix}/precision", 0.0))
        recall = float(metrics.get(f"{prefix}/recall", 0.0))
        f1 = float(metrics.get(f"{prefix}/f1", 0.0))
        dice = float(metrics.get(f"{prefix}/dice", 0.0))
        iou = float(metrics.get(f"{prefix}/iou", 0.0))
        accuracy = float(metrics.get(f"{prefix}/accuracy", 0.0))
        rows.append(
            {
                "scope": "class",
                "class_name": str(class_name),
                "class_id": class_id,
                "support": class_support,
                "accuracy_macro": accuracy,
                "accuracy_micro": accuracy,
                "f1_macro": f1,
                "f1_micro": f1,
                "dice_macro": dice,
                "dice_micro": dice,
                "iou_macro": iou,
                "iou_micro": iou,
                "foreground_iou": iou,
                "precision_macro": precision,
                "precision_micro": precision,
                "recall_macro": recall,
                "recall_micro": recall,
            }
        )
    return rows
def _write_test_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ['scope', 'class_name', 'class_id', 'support', 'accuracy_macro', 'accuracy_micro', *AGGREGATE_METRIC_KEYS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
def evaluate(cfg: EvalConfig, model: nn.Module, criterion: nn.Module, device: torch.device, *, split: str | None=None, num_classes: int | None=None, label_dict: dict[int, str] | None=None, feature_layers: list[str] | None=None) -> dict[str, float]:
    """Run the configured evaluation logic and write test artifacts."""
    test_split = split or str(cfg.data.get('test_split', 'test'))
    test_dataset = _make_dataset(cfg, split=test_split, max_samples=None, feature_layers=feature_layers)
    test_loader = _make_loader(test_dataset, cfg, 'test', shuffle=False)
    test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes or test_dataset.num_classes, cfg)
    if _is_survival(cfg):
        test_dir = cfg.output_dir / 'test_result'
        test_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
            json.dump({k: v for k, v in test_metrics.items() if k not in {'km_rows', 'rows'}}, handle, indent=2)
        rows = test_metrics.get('rows', [])
        if rows:
            with (test_dir / 'patient_risk.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        km_rows = test_metrics.get('km_rows', [])
        if km_rows:
            with (test_dir / 'kaplan_meier_by_risk_group.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(km_rows[0]))
                writer.writeheader()
                writer.writerows(km_rows)
        return test_metrics
    else:
        if _is_classification(cfg):
            test_dir = cfg.output_dir / 'test_result'
            test_dir.mkdir(parents=True, exist_ok=True)
            with (cfg.output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
                json.dump({k: v for k, v in test_metrics.items() if k != 'rows'}, handle, indent=2)
            rows = test_metrics.get('rows', [])
            if rows:
                with (test_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            return test_metrics
        else:
            with (cfg.output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
                json.dump(test_metrics, handle, indent=2)
            test_rows = _classwise_test_rows(test_metrics, label_dict or test_dataset.label_dict, exclude_background=bool(cfg.metrics.get('exclude_background', True)))
            _write_test_csv(cfg.output_dir / 'test_result' / f"{cfg.data.get('test_result_name', 'pathocell_test_metrics.csv')}", test_rows)
            return test_metrics
def train_one_survival_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, grad_accum: int) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, risks, event_times, censorships = ([], [], [], [])
    attention_diagnostics = []
    for batch_idx, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        features = batch['features'].to(device, non_blocking=True)
        label = batch['label'].view(1).to(device)
        censorship = batch['censorship'].view(1).to(device)
        hazards, survival, _ = model(features)
        attention = _attention_diagnostics(model)
        if attention:
            attention_diagnostics.append(attention)
        loss = criterion(hazards, survival, label, censorship)
        (loss / grad_accum).backward()
        if (batch_idx + 1) % grad_accum == 0 or batch_idx + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        risks.append(float(-torch.sum(survival, dim=1).detach().cpu().item()))
        event_times.append(float(batch['event_time']))
        censorships.append(float(batch['censorship']))
    risk_array = np.asarray(risks)
    metrics = {'loss': float(np.mean(losses)) if losses else 0.0, 'c_index': concordance_index(np.asarray(event_times), risk_array, np.asarray(censorships)), **logrank_by_median_risk(np.asarray(event_times), risk_array, np.asarray(censorships))}
    if attention_diagnostics:
        for key in attention_diagnostics[0]:
            metrics[key] = float(np.mean([item[key] for item in attention_diagnostics]))
    return metrics
def train_one_classification_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_classes: int, attention_entropy_weight: float=0.0) -> dict[str, float]:
    model.train()
    labels, predictions, probabilities, losses, ce_losses, attention_entropies = ([], [], [], [], [], [])
    attention_diagnostics = []
    for batch in tqdm(loader, desc='train', leave=False):
        features = batch['features'].to(device, non_blocking=True)
        label = batch['label'].view(1).to(device)
        logits, probs, prediction = model(features)
        attention = _attention_diagnostics(model)
        if attention:
            attention_diagnostics.append(attention)
        ce_loss = criterion(logits, label)
        attention_entropy = _normalized_attention_entropy(model)
        loss = ce_loss
        if attention_entropy_weight > 0:
            if attention_entropy is None:
                raise ValueError('attention_entropy_weight requires an attention MIL head')
            else:
                loss = loss + attention_entropy_weight * (1.0 - attention_entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        labels.append(int(label.detach().cpu().item()))
        predictions.append(int(prediction.detach().cpu().item()))
        probabilities.append(probs.detach().cpu().squeeze(0).numpy())
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        if attention_entropy is not None:
            attention_entropies.append(float(attention_entropy.detach().cpu()))
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions), np.asarray(probabilities), num_classes)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    metrics['ce_loss'] = float(np.mean(ce_losses)) if ce_losses else 0.0
    if attention_entropies:
        metrics['attention_entropy'] = float(np.mean(attention_entropies))
    if attention_diagnostics:
        for key in attention_diagnostics[0]:
            metrics[key] = float(np.mean([item[key] for item in attention_diagnostics]))
    return metrics
def _normalized_attention_entropy(model: nn.Module) -> torch.Tensor | None:
    backbone = getattr(model, 'backbone', None)
    weights = getattr(backbone, 'last_attention', None)
    if weights is None:
        return
    else:
        probabilities = weights.clamp_min(torch.finfo(weights.dtype).eps)
        n_instances = probabilities.shape[(-1)]
        if n_instances <= 1:
            return probabilities.new_tensor(1.0)
        else:
            entropy = -(probabilities * probabilities.log()).sum(dim=(-1)).mean()
            return entropy / math.log(n_instances)
def _attention_diagnostics(model: nn.Module) -> dict[str, float]:
    backbone = getattr(model, 'backbone', None)
    weights = getattr(backbone, 'last_attention', None)
    if weights is None:
        return {}
    else:
        probabilities = weights.clamp_min(torch.finfo(weights.dtype).eps)
        n_instances = probabilities.shape[(-1)]
        entropy_raw = -(probabilities * probabilities.log()).sum(dim=(-1)).mean()
        normalized = entropy_raw / math.log(n_instances) if n_instances > 1 else probabilities.new_tensor(1.0)
        result = {'attention_entropy': float(normalized.detach().cpu()), 'attention_effective_fraction': float((entropy_raw.exp() / max(n_instances, 1)).detach().cpu())}
        logits = getattr(backbone, 'last_attention_logits', None)
        if logits is not None:
            result['attention_logit_variance'] = float(logits.var(unbiased=False).detach().cpu())
            result['attention_logit_range'] = float((logits.max() - logits.min()).detach().cpu())
        return result
def _survival_optimizer(cfg: EvalConfig, model: nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg.training.get('optimizer') or {}
    name = str(optim_cfg.get('name', 'Adam'))
    params = dict(optim_cfg.get('params') or {})
    params.setdefault('lr', cfg.training.get('lr', 0.0002))
    params.setdefault('weight_decay', cfg.training.get('weight_decay', 1e-05))
    return getattr(torch.optim, name)(model.parameters(), **params)
def _make_survival_scheduler(cfg: EvalConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not bool(cfg.training.get('use_lr_scheduler', False)) and (not cfg.training.get('scheduler')):
            return
    scheduler_cfg = cfg.training.get('scheduler') or {}
    scheduler_name = str(scheduler_cfg.get('name', 'CosineAnnealingLR'))
    scheduler_params = dict(scheduler_cfg.get('params') or {})
    max_epochs = int(cfg.training.get('max_epochs', 20))
    if scheduler_name == 'CosineAnnealingLR':
        scheduler_params.setdefault('T_max', max_epochs)
        scheduler_params.setdefault('eta_min', 1e-06)
    return getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)
def _save_survival_epoch_checkpoint(cfg: EvalConfig, model: nn.Module, *, epoch: int, score: float, valid_metrics: dict[str, float]) -> None:
    epoch_dir = cfg.output_dir / 'epoch_checkpoints'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'score': score, 'valid_c_index': float(valid_metrics.get('c_index', 0.0)), 'valid_loss': float(valid_metrics.get('loss', 0.0))}, epoch_dir / f'epoch_{epoch:03d}.pt')
def _evaluate_survival_epoch_checkpoints(cfg: EvalConfig, model: nn.Module, criterion: nn.Module, device: torch.device, history: list[dict[str, Any]], train_log: Path) -> dict[str, Any]:
    epoch_dir = cfg.output_dir / 'epoch_checkpoints'
    test_split = cfg.data.get('test_split')
    if not epoch_dir.is_dir() or not test_split:
        return {}
    else:
        test_dataset = _make_dataset(cfg, split=str(test_split))
        test_loader = _make_loader(test_dataset, cfg, 'test', shuffle=False)
        history_by_epoch = {int(row['epoch']): row for row in history}
        rows = []
        for ckpt_path in sorted(epoch_dir.glob('epoch_*.pt')):
            epoch = int(ckpt_path.stem.split('_')[1])
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            test_metrics = evaluate_model(model, test_loader, criterion, device, test_dataset.num_classes, cfg)
            hist = history_by_epoch.get(epoch, {})
            rows.append({'epoch': epoch, 'lr': hist.get('lr'), 'train_c_index': hist.get('train_c_index'), 'valid_c_index': hist.get('valid_c_index'), 'valid_loss': hist.get('valid_loss'), 'test_c_index': float(test_metrics.get('c_index', 0.0)), 'test_loss': float(test_metrics.get('loss', 0.0))})
        summary_path = epoch_dir / 'epoch_test_summary.csv'
        fieldnames = sorted({key for row in rows for key in row})
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        best_by_valid = max(rows, key=lambda row: float(row.get('valid_c_index') or 0.0))
        best_by_test = max(rows, key=lambda row: float(row.get('test_c_index') or 0.0))
        summary = {'n_epochs': len(rows), 'best_by_valid_c_index': best_by_valid, 'best_by_test_c_index_oracle': best_by_test}
        with (epoch_dir / 'epoch_test_summary.json').open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        _log(f"Epoch checkpoint test sweep: best_valid epoch={best_by_valid.get('epoch')} valid_c_index={float(best_by_valid.get('valid_c_index') or 0.0):.4f} test_c_index={float(best_by_valid.get('test_c_index') or 0.0):.4f}; oracle_test epoch={best_by_test.get('epoch')} test_c_index={float(best_by_test.get('test_c_index') or 0.0):.4f}", train_log)
        return summary
def _classification_optimizer(cfg: EvalConfig, model: nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg.training.get('optimizer') or {}
    name = str(optim_cfg.get('name', 'Adam'))
    params = dict(optim_cfg.get('params') or {})
    params.setdefault('lr', cfg.training.get('lr', 0.0001))
    params.setdefault('weight_decay', cfg.training.get('weight_decay', 1e-05))
    return getattr(torch.optim, name)(model.parameters(), **params)
def _linear_probe_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None=None, reduction: str='mean') -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction=reduction)
def _linear_probe_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.detach().cpu(), minlength=int(num_classes)).to(torch.float64)
    if bool((counts <= 0).any()):
        raise ValueError(f'Cannot compute balanced class weights with counts={counts.tolist()}')
    else:
        return counts.sum() / (float(num_classes) * counts)
def _linear_probe_weighting(labels: torch.Tensor, num_classes: int, mode: str) -> torch.Tensor | None:
    key = str(mode or 'none').lower()
    if key in {'none', 'off', 'false', '0'}:
        return
    else:
        balanced = _linear_probe_class_weights(labels, num_classes)
        if key == 'balanced':
            return balanced
        else:
            if key in {'sqrt_balanced', 'sqrt_inverse_frequency'}:
                return balanced.sqrt()
            else:
                raise ValueError(f'Unknown linear-probe class_weight_mode={mode!r}')
def _linear_probe_regularization(model: nn.Module, cost: float, penalty: str='unsquared_l2') -> torch.Tensor:
    weight = model.classifier.weight
    if penalty == 'unsquared_l2':
        return 0.5 / float(cost) * weight.norm(p=2)
    else:
        if penalty == 'squared_l2':
            return 0.5 / float(cost) * weight.square().sum()
        else:
            raise ValueError(f'Unknown linear-probe penalty: {penalty!r}')
def _fit_linear_probe(*_args, **_kwargs):
    raise RuntimeError("LBFGS linear probe was removed; use the configured AdamW linear probe.")
def _eval_linear_probe(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], *, num_classes: int, device: torch.device, batch_size: int=65536, label_dict: dict[int, str] | None=None, include_rows: bool=True) -> dict[str, Any]:
    model.eval()
    parameter_dtype = next(model.parameters()).dtype
    prob_chunks = []
    loss_sum = 0.0
    for start in range(0, int(features.shape[0]), int(batch_size)):
        batch = features[start:start + int(batch_size)].to(device=device, dtype=parameter_dtype)
        batch_labels = labels[start:start + int(batch_size)].to(device)
        logits = model(batch)
        loss_sum += float(nn.functional.cross_entropy(logits, batch_labels, reduction='sum').item())
        prob_chunks.append(torch.softmax(logits, dim=(-1)).detach().cpu())
    probs = torch.cat(prob_chunks, dim=0).numpy()
    preds = probs.argmax(axis=1)
    y = labels.detach().cpu().numpy()
    class_names = [str(label_dict[idx]) for idx in range(num_classes)] if label_dict else None
    metrics = linear_probe_metrics(y, preds, probs, num_classes, class_names=class_names)
    metrics['loss'] = loss_sum / max(1, int(labels.shape[0]))
    safe_probs = np.clip(probs, 1e-12, 1.0)
    metrics['prediction_entropy_mean'] = float(np.mean(-np.sum(safe_probs * np.log(safe_probs), axis=1)))
    metrics['prediction_counts'] = np.bincount(preds, minlength=num_classes).astype(int).tolist()
    if include_rows:
        metrics['rows'] = [{**row, 'prediction': int(pred), **{f'probability_class_{idx}': float(prob[idx]) for idx in range(num_classes)}} for row, pred, prob in zip(rows, preds.tolist(), probs)]
    return metrics
def _write_linear_probe_test_outputs(cfg: EvalConfig, metrics: dict[str, Any]) -> None:
    test_dir = cfg.output_dir / 'test_result'
    test_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != 'rows'}
    with (cfg.output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(serializable, handle, indent=2)
    rows = metrics.get('rows', [])
    if rows:
        with (test_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report_rows = _linear_probe_test_rows(metrics)
    if report_rows:
        _write_test_csv(test_dir / 'linear_probe_report.csv', report_rows)
def _linear_probe_test_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    report = metrics.get("lin_report")
    if not isinstance(report, dict):
        return []
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    class_items = [
        (name, values)
        for name, values in report.items()
        if isinstance(values, dict) and name not in {"macro avg", "weighted avg"}
    ]
    support = float(weighted.get("support", sum(float(item.get("support", 0)) for _, item in class_items)))
    rows = [
        {
            "scope": "overall",
            "class_name": "ALL",
            "class_id": -1,
            "support": support,
            "accuracy_macro": float(metrics.get("lin_acc", 0.0)),
            "accuracy_micro": float(metrics.get("lin_bacc", 0.0)),
            "f1_macro": float(metrics.get("lin_macro_f1", 0.0)),
            "f1_micro": float(macro.get("f1-score", 0.0)),
            "precision_macro": float(macro.get("precision", 0.0)),
            "precision_micro": float(weighted.get("precision", 0.0)),
            "recall_macro": float(macro.get("recall", 0.0)),
            "recall_micro": float(weighted.get("recall", 0.0)),
        }
    ]
    for class_id, (name, values) in enumerate(class_items):
        rows.append(
            {
                "scope": "class",
                "class_name": str(name),
                "class_id": class_id,
                "support": float(values.get("support", 0.0)),
                "accuracy_macro": float(values.get("recall", 0.0)),
                "accuracy_micro": float(values.get("recall", 0.0)),
                "f1_macro": float(values.get("f1-score", 0.0)),
                "f1_micro": float(values.get("f1-score", 0.0)),
                "precision_macro": float(values.get("precision", 0.0)),
                "precision_micro": float(values.get("precision", 0.0)),
                "recall_macro": float(values.get("recall", 0.0)),
                "recall_micro": float(values.get("recall", 0.0)),
            }
        )
    return rows
def run_linear_probe(cfg: EvalConfig) -> None:
    """Train the configured AdamW linear probe."""
    from training.linear_probe_adamw import run_linear_probe as run_adamw

    run_adamw(cfg)
def run_classification(cfg: EvalConfig) -> None:
    if _cross_validation_enabled(cfg):
        _run_patient_cross_validation(cfg, _run_classification_once)
        return
    else:
        seeds = _classification_seeds(cfg)
        multiseed = len(seeds) > 1
        summaries = []
        for seed in seeds:
            run_cfg = _clone_eval_config_for_seed(cfg, seed, multiseed=multiseed)
            summary = _run_classification_once(run_cfg)
            if multiseed:
                summaries.append(summary)
        if multiseed:
            _write_multi_seed_summary(cfg.output_dir, summaries)
def _run_classification_once(cfg: EvalConfig) -> dict[str, Any]:
    seed = int(cfg.training.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg.training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dirs = _prepare_output_dirs(cfg)
    train_log = output_dirs["logs"] / "train.log"
    _log(f"experiment_dir={cfg.output_dir} seed={seed}", train_log)
    _configure_torch_threads(cfg, train_log)
    train_dataset, valid_dataset, audit = _prepare_classification_datasets(cfg)
    _write_dataset_audit(cfg, audit, train_log)
    sampler = (
        _weighted_sampler(train_dataset, train_dataset.num_classes, [])
        if bool(cfg.training.get("weighted_sample", False))
        else None
    )
    train_loader = _make_loader(train_dataset, cfg, "train", shuffle=True, sampler=sampler, log_path=train_log)
    valid_loader = _make_loader(valid_dataset, cfg, "valid", shuffle=False, log_path=train_log)
    model = _make_head(cfg, train_dataset).to(device)
    criterion = _loss_fn(cfg, train_dataset.num_classes).to(device)
    optimizer = _classification_optimizer(cfg, model)
    primary_metric = str(cfg.training.get("primary_metric", "auc"))
    metric_direction = str(cfg.training.get("primary_metric_mode", "max")).lower()
    best_score = float("-inf")
    epochs_without_improve = 0
    patience = int(cfg.training.get("early_stopping_patience", 0))
    min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    early_start = int(cfg.training.get("early_stopping_start_epoch", 0))
    checkpoint_start = int(cfg.training.get("checkpoint_start_epoch", 0))
    smooth_window = max(1, int(cfg.training.get("checkpoint_smoothing_window", 1)))
    history: list[dict[str, Any]] = []
    _log(f"Starting WSI classification training (dataset={cfg.dataset}, head={cfg.head})", train_log)

    for epoch in range(int(cfg.training.get("max_epochs", 200))):
        train_metrics = train_one_classification_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_dataset.num_classes,
            attention_entropy_weight=float(cfg.training.get("attention_entropy_weight", 0.0)),
        )
        valid_metrics = evaluate_model(model, valid_loader, criterion, device, valid_dataset.num_classes, cfg)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics.get("accuracy"),
            "train_auc": train_metrics.get("auc"),
            "train_auprc": train_metrics.get("auprc"),
            "train_ce_loss": train_metrics.get("ce_loss", train_metrics["loss"]),
            "train_attention_entropy": train_metrics.get("attention_entropy"),
            "train_attention_effective_fraction": train_metrics.get("attention_effective_fraction"),
            "train_attention_logit_variance": train_metrics.get("attention_logit_variance"),
            "train_attention_logit_range": train_metrics.get("attention_logit_range"),
            "valid_attention_entropy": valid_metrics.get("attention_entropy"),
            "valid_attention_effective_fraction": valid_metrics.get("attention_effective_fraction"),
            "valid_attention_logit_variance": valid_metrics.get("attention_logit_variance"),
            "valid_attention_logit_range": valid_metrics.get("attention_logit_range"),
            "f1_macro": valid_metrics.get("f1_macro"),
            "valid_f1_macro": valid_metrics.get("f1_macro"),
            **_recall_metric_columns(train_metrics, "train_"),
            **_recall_metric_columns(valid_metrics, "valid_"),
        }
        history.append(row)
        raw_score = float(valid_metrics.get(primary_metric, valid_metrics.get("auc", 0.0)))
        recent = [
            float(item.get(f"valid_{primary_metric}", item.get("valid_auc", 0.0)))
            for item in history[-smooth_window:]
        ]
        smoothed_score = float(np.mean(recent))
        score = -smoothed_score if metric_direction == "min" else smoothed_score
        row["checkpoint_score"] = smoothed_score
        _write_metrics(cfg.output_dir / "history.csv", history)
        _write_metrics(output_dirs["validation"] / "validation_metrics.csv", history)
        _log(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_auc={train_metrics.get('auc', 0.0):.4f} "
            f"train_auprc={train_metrics.get('auprc', 0.0):.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_auc={valid_metrics.get('auc', 0.0):.4f} "
            f"valid_auprc={valid_metrics.get('auprc', 0.0):.4f} "
            f"valid_acc={valid_metrics.get('accuracy', 0.0):.4f}",
            train_log,
        )
        eligible = epoch >= checkpoint_start
        improved = score > best_score + min_delta
        if eligible and improved:
            best_score = score
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "primary_metric": primary_metric,
                    "primary_metric_mode": metric_direction,
                    "score": raw_score,
                    "checkpoint_score": smoothed_score,
                    "checkpoint_smoothing_window": smooth_window,
                    "seed": seed,
                },
                cfg.output_dir / "best_model.pt",
            )
            _write_metrics(output_dirs["validation"] / "best_validation.csv", [row])
        elif eligible:
            epochs_without_improve += 1
            if patience > 0 and epoch >= early_start and epochs_without_improve >= patience:
                _log(f"Early stopping at epoch {epoch}: {primary_metric} did not improve for {patience} epochs.", train_log)
                break

    summary = {"seed": seed}
    summary.update(_best_valid_summary_metrics(cfg))
    _load_best_model_if_available(model, cfg, device, train_log)
    selected_valid = evaluate_classification_model(
        model, valid_loader, criterion, device, valid_dataset.num_classes, cfg, decision_threshold=0.5
    )
    validation_dir = cfg.output_dir / "validation_result"
    validation_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_valid.pop("rows", [])
    (validation_dir / "best_validation_metrics.json").write_text(
        json.dumps(selected_valid, indent=2), encoding="utf-8"
    )
    if selected_rows:
        with (validation_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)

    test_split = cfg.data.get("test_split")
    if test_split and bool(cfg.training.get("evaluate_test_after_training", True)):
        tuning_split = _threshold_tuning_split(cfg)
        threshold = None
        if tuning_split == "train":
            threshold = _tune_classification_threshold(
                cfg, model, train_loader, criterion, device, train_dataset.num_classes, train_log,
                split_name="train",
            )
        elif tuning_split == "valid":
            threshold = _tune_classification_threshold(
                cfg, model, valid_loader, criterion, device, valid_dataset.num_classes, train_log,
                split_name="valid",
            )
        else:
            threshold = _resolve_decision_threshold(cfg)
        if threshold is not None:
            summary["decision_threshold"] = threshold
        test_dataset = _make_dataset(cfg, split=str(test_split))
        audit["splits"][str(test_split)] = getattr(test_dataset, "audit", {})
        _write_dataset_audit(cfg, audit, train_log)
        test_metrics = evaluate(cfg, model, criterion, device, split=str(test_split))
        summary.update({key: value for key, value in test_metrics.items() if key != "rows"})
        _log(f"test_metrics={_aggregate_metrics(test_metrics)}", train_log)
    return summary
def run_survival(cfg: EvalConfig) -> None:
    if _cross_validation_enabled(cfg):
        _run_patient_cross_validation(cfg, _run_survival_once)
    else:
        _run_survival_once(cfg)
def _run_survival_once(cfg: EvalConfig) -> dict[str, Any]:
    seed = int(cfg.training.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg.training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dirs = _prepare_output_dirs(cfg)
    train_log = output_dirs["logs"] / "train.log"
    _log(f"experiment_dir={cfg.output_dir} seed={seed}", train_log)
    _configure_torch_threads(cfg, train_log)
    train_dataset, valid_dataset, audit = _prepare_survival_datasets(cfg)
    _write_dataset_audit(cfg, audit, train_log)
    sampler = (
        _weighted_sampler(train_dataset, train_dataset.num_classes, [])
        if bool(cfg.training.get("weighted_sample", True))
        else None
    )
    train_loader = _make_loader(train_dataset, cfg, "train", shuffle=True, sampler=sampler, log_path=train_log)
    valid_loader = _make_loader(valid_dataset, cfg, "valid", shuffle=False, log_path=train_log)
    model = _make_head(cfg, train_dataset).to(device)
    criterion = _loss_fn(cfg, train_dataset.num_classes)
    optimizer = _survival_optimizer(cfg, model)
    scheduler = _make_survival_scheduler(cfg, optimizer)
    max_epochs = int(cfg.training.get("max_epochs", 20))
    grad_accum = max(1, int(cfg.training.get("gradient_accumulation", cfg.training.get("gc", 32))))
    primary_metric = str(cfg.training.get("primary_metric", "c_index"))
    direction = str(cfg.training.get("primary_metric_mode", "max")).lower()
    if direction not in {"max", "min"}:
        direction = "min" if primary_metric == "loss" else "max"
    min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    patience = int(cfg.training.get("early_stopping_patience", 0))
    early_start = int(cfg.training.get("early_stopping_start_epoch", 0))
    checkpoint_start = int(cfg.training.get("checkpoint_selection_start_epoch", 0))
    if checkpoint_start < 0 or checkpoint_start >= max_epochs:
        raise ValueError("training.checkpoint_selection_start_epoch must be in [0, max_epochs)")
    save_epoch_checkpoints = bool(cfg.training.get("save_epoch_checkpoints", False))
    evaluate_epoch_checkpoints = bool(cfg.training.get("evaluate_epoch_checkpoints_on_test", False))
    best_score = float("inf") if direction == "min" else float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        train_metrics = train_one_survival_epoch(model, train_loader, criterion, optimizer, device, grad_accum)
        valid_metrics = evaluate_model(model, valid_loader, criterion, device, valid_dataset.num_classes, cfg)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_c_index": train_metrics["c_index"],
            "valid_loss": valid_metrics["loss"],
            "valid_c_index": valid_metrics["c_index"],
            "train_attention_entropy": train_metrics.get("attention_entropy"),
            "train_attention_effective_fraction": train_metrics.get("attention_effective_fraction"),
            "train_attention_logit_variance": train_metrics.get("attention_logit_variance"),
            "train_attention_logit_range": train_metrics.get("attention_logit_range"),
            "valid_attention_entropy": valid_metrics.get("attention_entropy"),
            "valid_attention_effective_fraction": valid_metrics.get("attention_effective_fraction"),
            "valid_attention_logit_variance": valid_metrics.get("attention_logit_variance"),
            "valid_attention_logit_range": valid_metrics.get("attention_logit_range"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _write_metrics(cfg.output_dir / "history.csv", history)
        _write_metrics(output_dirs["validation"] / "validation_metrics.csv", history)
        score = float(valid_metrics.get(primary_metric, valid_metrics["c_index"]))
        eligible = epoch >= checkpoint_start
        improved = eligible and (
            score < best_score - min_delta if direction == "min" else score > best_score + min_delta
        )
        _log(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_c_index={train_metrics['c_index']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_c_index={valid_metrics['c_index']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            train_log,
        )
        if save_epoch_checkpoints:
            _save_survival_epoch_checkpoint(cfg, model, epoch=epoch, score=score, valid_metrics=valid_metrics)
        if improved:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "primary_metric": primary_metric,
                    "primary_metric_mode": direction,
                    "score": score,
                    "seed": seed,
                },
                cfg.output_dir / "best_model.pt",
            )
            _write_metrics(output_dirs["validation"] / "best_validation.csv", [row])
        elif eligible and epoch >= early_start:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                _log(f"Early stopping at epoch {epoch}: {primary_metric} did not improve for {patience} epochs.", train_log)
                break
        if scheduler is not None:
            scheduler.step()

    if evaluate_epoch_checkpoints:
        _evaluate_survival_epoch_checkpoints(cfg, model, criterion, device, history, train_log)
    summary = {"seed": seed}
    summary.update(_best_valid_summary_metrics(cfg))
    test_split = cfg.data.get("test_split")
    if test_split and bool(cfg.training.get("evaluate_test_after_training", True)):
        _load_best_model_if_available(model, cfg, device, train_log)
        threshold_split = str(cfg.survival.get("risk_threshold_split", cfg.data.get("valid_split", "valid")))
        threshold_loader = valid_loader
        if threshold_split != str(cfg.data.get("valid_split", "valid")):
            threshold_dataset = _make_dataset(cfg, split=threshold_split)
            threshold_loader = _make_loader(
                threshold_dataset, cfg, threshold_split, shuffle=False, log_path=train_log
            )
        threshold_cfg = EvalConfig(
            id=cfg.id,
            task=cfg.task,
            dataset=cfg.dataset,
            head=cfg.head,
            output_dir=cfg.output_dir,
            data=dict(cfg.data),
            head_config=dict(cfg.head_config),
            training=dict(cfg.training),
            metrics=dict(cfg.metrics),
            survival={**dict(cfg.survival), "risk_threshold": None, "risk_threshold_source": "split_median", "bootstrap_ci": False},
        )
        threshold_metrics = evaluate_model(
            model, threshold_loader, criterion, device, valid_dataset.num_classes, threshold_cfg
        )
        cfg.survival["risk_threshold"] = float(
            threshold_metrics.get("risk_threshold", threshold_metrics.get("risk_median", 0.0))
        )
        cfg.survival["risk_threshold_source"] = threshold_split
        with (cfg.output_dir / "risk_threshold.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "risk_threshold": cfg.survival["risk_threshold"],
                    "risk_threshold_source": threshold_split,
                    "source_c_index": float(threshold_metrics.get("c_index", 0.0)),
                    "source_n_patients": float(threshold_metrics.get("n_patients", 0.0)),
                    "source_n_events": float(threshold_metrics.get("n_events", 0.0)),
                },
                handle,
                indent=2,
            )
        test_metrics = evaluate(cfg, model, criterion, device, split=str(test_split))
        summary.update({key: value for key, value in test_metrics.items() if key != "rows"})
        _log(f"test_metrics={_aggregate_metrics(test_metrics)}", train_log)
    return summary
def run(cfg: EvalConfig) -> None:
    _validate_task_config(cfg)
    if _is_linear_probe(cfg):
        run_linear_probe(cfg)
        return
    else:
        if _is_survival(cfg):
            run_survival(cfg)
            return
        else:
            if _is_classification(cfg):
                run_classification(cfg)
                return
            else:
                seed = int(cfg.training.get('seed', 42))
                set_seed(seed)
                device = torch.device(cfg.training.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
                output_dirs = _prepare_output_dirs(cfg)
                train_log = output_dirs['logs'] / 'train.log'
                _log(f'experiment_dir={cfg.output_dir}', train_log)
                _configure_torch_threads(cfg, train_log)
                max_samples = cfg.data.get('max_samples')
                _log(f"Loading train dataset split={cfg.data.get('train_split', 'train')}", train_log)
                train_dataset = _make_dataset(cfg, split=str(cfg.data.get('train_split', 'train')), max_samples=int(max_samples) if max_samples else None)
                _log(f"Loaded train dataset: patches={len(train_dataset)} samples={len(getattr(train_dataset, 'samples', []))} feature_layers={list(train_dataset.feature_layers)} feature_cache_size={getattr(train_dataset, 'feature_cache_size', 'n/a')}", train_log)
                _log(f"Loading valid dataset split={cfg.data.get('valid_split', 'valid')}", train_log)
                valid_dataset = _make_dataset(cfg, split=str(cfg.data.get('valid_split', 'valid')), max_samples=int(max_samples) if max_samples else None)
                _log(f"Loaded valid dataset: patches={len(valid_dataset)} samples={len(getattr(valid_dataset, 'samples', []))} feature_cache_size={getattr(valid_dataset, 'feature_cache_size', 'n/a')}", train_log)
                loss_cfg = dict(cfg.training.get('loss_fn') or {})
                exclude_classes = list(loss_cfg.get('exclude_classes', [0]))
                sampler = None
                if bool(cfg.training.get('uniform_class_sampling', True)):
                    sample_excluded_classes = bool(cfg.training.get('sample_excluded_classes', False))
                    sampler_exclude = exclude_classes if sample_excluded_classes else []
                    sampler_classes = [cls for cls in range(train_dataset.num_classes) if cls not in set(sampler_exclude)]
                    _log(f'Building uniform class sampler from aligned masks (patches={len(train_dataset)}, classes={sampler_classes})', train_log)
                    sampler = _weighted_sampler(train_dataset, train_dataset.num_classes, sampler_exclude)
                    _log(f'segmentation_dataloader uniform_class_sampling=True weighted_sampler_classes={sampler_classes} sample_excluded_classes={sample_excluded_classes}', train_log)
                train_loader = _make_loader(train_dataset, cfg, 'train', shuffle=True, sampler=sampler, log_path=train_log)
                valid_loader = _make_loader(valid_dataset, cfg, 'valid', shuffle=False, log_path=train_log)
                model = _make_head(cfg, train_dataset).to(device)
                criterion = _loss_fn(cfg, train_dataset.num_classes)
                optimizer = _make_optimizer(cfg, model)
                scheduler = _make_scheduler(cfg, optimizer)
                use_amp = bool(cfg.training.get('amp', True)) and device.type == 'cuda'
                scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
                _log(f'Starting segmentation training (dataset={cfg.dataset})', train_log)
                _log(f'device={device} num_classes={train_dataset.num_classes}', train_log)
                _log(f"lr_scheduler={(cfg.training.get('scheduler') or {}).get('name', 'CosineAnnealingLR')} warmup_steps={(cfg.training.get('optimizer') or {}).get('warmup_steps', 1000)} scheduler_step_each_batch=True total_plan_optimizer_steps={int(cfg.training.get('training_steps', 20000))}", train_log)
                primary_metric = str(cfg.training.get('primary_metric', 'f1_macro'))
                best_score = -float('inf')
                best_step = 0
                steps_since_last_best = 0
                history = []
                training_steps = int(cfg.training.get('training_steps', 20000))
                val_interval = int(cfg.training.get('val_interval', 200))
                log_interval = int(cfg.training.get('log_interval', 10))
                early_stopping = int(cfg.training.get('early_stopping', 1200))
                grad_clip_norm = float(cfg.training.get('grad_clip_norm', 2.0))
                loss_window = []
                validation_index = 0
                resume_state = _resume_last_checkpoint_if_requested(cfg, model, optimizer, scheduler, scaler, device, train_log)
                if resume_state:
                    best_score = float(resume_state.get('best_score', best_score))
                    best_step = int(resume_state.get('best_step', best_step))
                    steps_since_last_best = int(resume_state.get('steps_since_last_best', steps_since_last_best))
                    validation_index = int(resume_state.get('validation_index', validation_index))
                    history = list(resume_state.get('history', history))
                start_step = int(resume_state.get('step', (-1))) + 1 if resume_state else 0
                checkpoint_interval = int(cfg.training.get('checkpoint_interval', val_interval))
                steps_per_run = cfg.training.get('steps_per_run')
                run_stop_step = training_steps
                if steps_per_run:
                    run_stop_step = min(training_steps, start_step + int(steps_per_run))
                    _log(f'steps_per_run={int(steps_per_run)}; this run will train steps {start_step}..{max(start_step, run_stop_step) - 1} of total {training_steps}.', train_log)
                last_step_seen = start_step - 1
                train_iter = iter(train_loader)
                for step in range(start_step, run_stop_step):
                    last_step_seen = step
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    loss = train_one_step(model, batch, criterion, optimizer, scheduler, scaler, device, grad_clip_norm, use_amp)
                    if not np.isnan(loss):
                        loss_window.append(loss)
                    if step % log_interval == 0:
                        mean_loss = float(np.mean(loss_window)) if loss_window else float('nan')
                        _log(f"step={step} train_loss={mean_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g}", train_log)
                    should_validate = step > 0 and (step % val_interval == 0 or step == training_steps - 1)
                    if should_validate:
                        train_loss = float(np.mean(loss_window)) if loss_window else float('nan')
                        metrics = evaluate_model(model, valid_loader, criterion, device, train_dataset.num_classes, cfg)
                        validation_index += 1
                        row = _validation_row(validation_index, step, train_loss, metrics)
                        row['lr'] = optimizer.param_groups[0]['lr']
                        history.append(row)
                        _write_metrics(cfg.output_dir / 'history.csv', history)
                        _write_validation_csv(output_dirs['validation'] / 'validation_metrics.csv', history)
                        score = float(metrics.get(primary_metric, -metrics['loss']))
                        _log(f"step={step} train_loss={train_loss:.4f} val_f1_macro={metrics.get('f1_macro', 0.0):.4f} val_iou_macro={metrics.get('iou_macro', 0.0):.4f} foreground_iou={metrics.get('foreground_iou', 0.0):.4f}", train_log)
                        steps_since_last_best += val_interval
                        if score > best_score:
                            best_score = score
                            best_step = step
                            steps_since_last_best = 0
                            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'primary_metric': primary_metric, 'score': score, 'feature_layers': list(getattr(model, 'feature_layers', []))}, cfg.output_dir / 'best_model.pt')
                            _write_validation_csv(output_dirs['validation'] / 'best_validation.csv', [row])
                        loss_window = []
                        if steps_since_last_best >= early_stopping:
                            _log(f'Early stopping at step {step}: validation {primary_metric} did not improve for {early_stopping} steps (best {primary_metric}={best_score:.6f} at step {best_step}).', train_log)
                            _save_last_checkpoint(cfg, model, optimizer, scheduler, scaler, step=step, best_score=best_score, best_step=best_step, steps_since_last_best=steps_since_last_best, validation_index=validation_index, history=history)
                            break
                    if checkpoint_interval > 0 and step > start_step and (step % checkpoint_interval == 0):
                                _save_last_checkpoint(cfg, model, optimizer, scheduler, scaler, step=step, best_score=best_score, best_step=best_step, steps_since_last_best=steps_since_last_best, validation_index=validation_index, history=history)
                if run_stop_step > 0:
                    _save_last_checkpoint(cfg, model, optimizer, scheduler, scaler, step=max(last_step_seen, start_step - 1), best_score=best_score, best_step=best_step, steps_since_last_best=steps_since_last_best, validation_index=validation_index, history=history)
                test_split = cfg.data.get('test_split')
                training_complete = last_step_seen >= training_steps - 1
                if test_split and training_complete:
                    pinned_layers = list(train_dataset.feature_layers)
                    _load_best_model_if_available(model, cfg, device, train_log, expected_feature_layers=pinned_layers)
                    test_metrics = evaluate(cfg, model, criterion, device, split=str(test_split), feature_layers=pinned_layers)
                    _log(f'test_metrics={_aggregate_metrics(test_metrics)}', train_log)
                else:
                    if test_split:
                        _log(f'Skipping test evaluation because batched training is not complete (last_step={last_step_seen}, training_steps={training_steps}).', train_log)
def run_eval_only(cfg: EvalConfig, split: str | None=None) -> None:
    _validate_task_config(cfg)
    if _is_linear_probe(cfg):
        raise ValueError('eval-only is not supported for linear_probe; rerun the linear probe config.')
    else:
        seed = int(cfg.training.get('seed', 42))
        set_seed(seed)
        device = torch.device(cfg.training.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
        output_dirs = _prepare_output_dirs(cfg)
        train_log = output_dirs['logs'] / 'train.log'
        train_dataset = _make_dataset(cfg, split=str(cfg.data.get('train_split', 'train')))
        model = _make_head(cfg, train_dataset).to(device)
        criterion = _loss_fn(cfg, train_dataset.num_classes)
        feature_layers = None
        if not _is_mil(cfg):
            feature_layers = list(train_dataset.feature_layers)
        _load_best_model_if_available(model, cfg, device, train_log, expected_feature_layers=feature_layers)
        eval_split = split or str(cfg.data.get('test_split', 'test'))
        if _is_classification(cfg):
            tuning_split = _threshold_tuning_split(cfg)
            if tuning_split:
                if tuning_split == 'train':
                    tune_loader = _make_loader(train_dataset, cfg, 'train', shuffle=False, log_path=train_log)
                    tune_num_classes = train_dataset.num_classes
                else:
                    tune_dataset = _make_dataset(cfg, split=str(cfg.data.get('valid_split', 'valid')))
                    tune_loader = _make_loader(tune_dataset, cfg, 'valid', shuffle=False, log_path=train_log)
                    tune_num_classes = tune_dataset.num_classes
                _tune_classification_threshold(cfg, model, tune_loader, criterion, device, tune_num_classes, train_log, split_name=tuning_split)
            else:
                if (cfg.output_dir / 'decision_threshold.json').is_file():
                    _load_decision_threshold_from_disk(cfg)
        test_metrics = evaluate(cfg, model, criterion, device, split=eval_split, num_classes=train_dataset.num_classes, label_dict=getattr(train_dataset, 'label_dict', None), feature_layers=feature_layers)
        _log(f'eval_only split={eval_split} metrics={_aggregate_metrics(test_metrics)}', train_log)
def main() -> None:
    parser = argparse.ArgumentParser(description='Train downstream heads on extracted features')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--eval-only', action='store_true', help='Load best_model.pt and run evaluation without training.')
    parser.add_argument('--eval-split', default=None, help='Split to evaluate when --eval-only is set (default: config data.test_split).')
    parser.add_argument('--seed', type=int, default=None, help='Run one seed only; overrides training.seeds and writes to output_dir/seed_<seed> when seeds is set.')
    parser.add_argument('--fold', type=int, default=None, help='Run one CV fold only; requires data.cross_validation.enabled.')
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    cfg = _apply_cli_fold_override(cfg, args.fold)
    cfg = _apply_cli_seed_override(cfg, args.seed)
    train_log = cfg.output_dir / 'validation_result' / 'logs' / 'train.log'
    train_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.eval_only:
            run_eval_only(cfg, split=args.eval_split)
        else:
            run(cfg)
    except KeyboardInterrupt:
        _log('Training interrupted by KeyboardInterrupt.', train_log)
        raise
    except Exception:
        _log('Training failed with exception:\n' + traceback.format_exc(), train_log)
        raise
if __name__ == '__main__':
    main()
from training.pca_features import install_pca_hook

install_pca_hook()
