# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: 'training/online_segmentation_train.py'
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2026-07-20 06:43:30 UTC (1784529810)

"""Shared online frozen-encoder UNETR training for dense segmentation tasks."""
from __future__ import annotations
import csv
import json
import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Protocol
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from dataclasses import asdict, is_dataclass
from data.prefetching_batch_source import PrefetchingBatchSource
from data.loader import load_split_samples
from training.augmentations import InputNormalization, build_augmenter, find_input_normalization
from training.config import EvalConfig, load_eval_config, resolve_path
from training.distributed import DistributedContext, broadcast_bool, destroy_distributed, initialize_distributed
from heads import build_segmentation_head, forward_segmentation_head, resolve_decoder_mode
from heads.unetr import classify_encoder_pyramid
from metrics import InstanceLevelMetricAccumulator, SegmentationMetricAccumulator, aggregate_pfmb_instance_metrics, format_pfmb_instance_validation_log, uses_pfmb_instance_primary_metric
from training.offline_train import _aggregate_metrics, _classwise_test_rows, _loss_fn, _make_optimizer, _make_scheduler, _write_test_csv
from utils.experiment_config import ExperimentConfig, load_experiment_config
from feature_extraction import build_feature_extractor
from models import get_custom_transformer, get_model
class OnlineBatchSource(Protocol):
    def next_batch(self) -> dict[str, Any]:
        return
    def reset_epoch(self, epoch: int | None=None) -> None:
        return
    def close(self) -> None:
        return
def load_online_eval_bundle(path: Path) -> tuple[ExperimentConfig, EvalConfig]:
    """Load a unified online YAML with embedded ``experiment`` and ``evaluation`` blocks."""
    config_path = resolve_path(path)
    with config_path.open(encoding='utf-8') as handle:
        raw = yaml.safe_load(handle) or {}
    exp = load_experiment_config(config_path)
    evaluation = raw.get('evaluation') or {}
    run = raw.get('run') or {}
    eval_id = str(evaluation.get('id') or config_path.stem)
    output_dir = resolve_path(run.get('output_dir', f'../Evaluation/{eval_id}'))
    data = dict(raw.get('data') or {})
    cfg = EvalConfig(id=eval_id, task=str(evaluation.get('task', 'tls_region_segmentation')), dataset=str(evaluation.get('dataset', 'hooknet_tls_online')), head=str(evaluation.get('head', 'unetr')), output_dir=output_dir, data=data, head_config=dict(raw.get('head') or {}), training=dict(raw.get('training') or {}), metrics=dict(raw.get('metrics') or {}), survival=dict(raw.get('survival') or {}))
    if exp.dense_interface == 'default':
        from heads.interfaces import validate_dense_config
        validate_dense_config(exp.model, exp.target_features, cfg.head_config, cfg.training)
    return (exp, cfg)
def _json_ready(value: Any) -> Any:
    """Convert paths and nested config containers into JSON/YAML-safe values."""
    if isinstance(value, Path):
        return str(value)
    elif is_dataclass(value):
        return _json_ready(asdict(value))
    else:
        if isinstance(value, dict):
            return {str(key): _json_ready(item) for key, item in value.items()}
        else:
            if isinstance(value, (list, tuple)):
                return [_json_ready(item) for item in value]
            else:
                return value
def _official_sampling_payload(exp: ExperimentConfig) -> dict[str, Any] | None:
    if exp.official_sampling is None:
        return
    else:
        return _json_ready(vars(exp.official_sampling))
def _online_config_payload(exp: ExperimentConfig, cfg: EvalConfig) -> dict[str, Any]:
    """Resolve the YAML and runtime settings into a portable config snapshot."""
    experiment_fields = (
        "id", "head", "dataset", "model", "target_mpp", "target_patch_size",
        "target_features", "pathocell_tiling",
    )
    run_fields = (
        "output_dir", "feat_dir", "feature_storage", "feature_extract_batch_size",
        "use_benchmark_config", "benchmark_config_path", "fallback_mpp",
        "output_layout", "grid_stride", "feature_dtype", "mask_dtype",
        "input_format", "wds_root", "extraction_mode", "backend", "cpu_threads",
        "iterator_cpus", "progress_every", "coords_manifest", "fixed_grid_sampling",
    )
    return _json_ready(
        {
            "description": "Resolved config for this online segmentation run.",
            "evaluation": {key: getattr(cfg, key, None) for key in ("id", "task", "dataset", "head")},
            "experiment": {key: getattr(exp, key, None) for key in experiment_fields},
            "masks": {"save": bool(getattr(exp, "save_masks", False))},
            "data": cfg.data,
            "head": cfg.head_config,
            "training": cfg.training,
            "metrics": cfg.metrics,
            "survival": cfg.survival,
            "run": {key: getattr(exp, key, None) for key in run_fields},
        }
    )
def _prepare_online_output_dirs(exp: ExperimentConfig, cfg: EvalConfig) -> dict[str, Path]:
    dirs = {'validation': cfg.output_dir / 'validation_result', 'test': cfg.output_dir / 'test_result', 'logs': cfg.output_dir / 'validation_result' / 'logs'}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    payload = _online_config_payload(exp, cfg)
    with (cfg.output_dir / 'resolved_config.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    with (cfg.output_dir / 'config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return dirs
def resolve_num_classes(cfg: EvalConfig, exp: ExperimentConfig) -> int:
    if cfg.task == 'tls_region_segmentation':
        return 4
    else:
        label_dict = exp.dataset_config.label_dict_path
        if label_dict and label_dict.is_file():
            with label_dict.open(encoding='utf-8') as handle:
                labels = json.load(handle)
            return max((int(k) for k in labels)) + 1
        else:
            return 15
def build_frozen_encoder(exp: ExperimentConfig, device: torch.device):
    base_model = get_model(exp.model, device, torch.cuda.device_count(), use_benchmark_config=exp.use_benchmark_config, config_path=str(exp.benchmark_config_path) if exp.benchmark_config_path else None)
    transform = get_custom_transformer(exp.model, use_benchmark_config=exp.use_benchmark_config, config_path=str(exp.benchmark_config_path) if exp.benchmark_config_path else None)
    encoder = build_feature_extractor(base_model, exp.extraction_task()).to(device).eval()
    return (encoder, transform)
def encoder_features(encoder, images: torch.Tensor, *, device: torch.device, use_amp: bool, input_size: int | None=None) -> dict[str, torch.Tensor]:
    images = images.to(device, non_blocking=True)
    if input_size is not None and tuple(images.shape[(-2):]) != (input_size, input_size):
            images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False, antialias=True)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            features = encoder(images)
    if not isinstance(features, dict):
        raise TypeError(f'Expected frozen encoder to return feature dict, got {type(features)}')
    else:
        return {name: value.detach() for name, value in features.items()}
def build_decoder(cfg: EvalConfig, exp: ExperimentConfig, first_features: dict[str, torch.Tensor], *, num_classes: int) -> nn.Module:
    layers = list(cfg.head_config.get('feature_layers') or exp.target_features)
    channels = [int(first_features[name].shape[1]) for name in layers]
    return build_segmentation_head(cfg.head, feature_layers=layers, in_channels=channels, num_classes=num_classes, head_config=cfg.head_config)
def maybe_augment(augmenter: nn.Module | None, images: torch.Tensor, masks: torch.Tensor, instance_masks: torch.Tensor | None=None, *, input_normalization: InputNormalization | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if augmenter is None:
        return (images, masks, instance_masks)
    else:
        if input_normalization is not None:
            images = input_normalization.decode(images)
        if instance_masks is not None:
            aug_images, aug_masks, aug_instances = augmenter(images, masks, instance_masks)
            if input_normalization is not None:
                aug_images = input_normalization.encode(aug_images)
            return (aug_images, aug_masks, aug_instances)
        else:
            aug_images, aug_masks = augmenter(images, masks)
            if input_normalization is not None:
                aug_images = input_normalization.encode(aug_images)
            return (aug_images, aug_masks, instance_masks)
def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    else:
        keys = sorted({key for row in rows for key in row})
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
def _pathocell_label_dict(cfg: EvalConfig, exp: ExperimentConfig) -> dict[int, str]:
    label_dict = {}
    if exp.dataset_config.label_dict_path and exp.dataset_config.label_dict_path.is_file():
            with exp.dataset_config.label_dict_path.open(encoding='utf-8') as handle:
                label_dict = {int(k): str(v) for k, v in json.load(handle).items()}
    if not label_dict:
        num_classes = resolve_num_classes(cfg, exp)
        label_dict = {cls_id: f'class_{cls_id}' for cls_id in range(num_classes)}
    return label_dict
def _clone_online_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            if isinstance(value, list):
                out[key] = list(value)
            else:
                out[key] = value
    return out
def materialize_online_valid_batches(batch_source: OnlineBatchSource, *, steps: int, log: Callable[[str], None], label: str='validation') -> list[dict[str, Any]]:
    """Draw batches once at startup; replay the same subset every checkpoint."""
    reset = getattr(batch_source, 'reset_epoch', None)
    if callable(reset):
        reset()
    batches = [_clone_online_batch(batch_source.next_batch()) for _ in range(int(steps))]
    patches = sum((int(b['semantic_mask'].shape[0]) for b in batches))
    log(f'stage=setup frozen {label} subset batches={len(batches)} patches={patches} (replayed each eval)')
    return batches
def save_online_draws_manifest(path: Path, batches: list[dict[str, Any]], *, split: str, seed: int, seed_offset: int) -> None:
    rows = []
    for batch_idx, batch in enumerate(batches):
        names = batch.get('sample_name') or []
        coords = batch.get('coord')
        coord_list = coords.tolist() if torch.is_tensor(coords) else coords
        for patch_idx, slide_id in enumerate(names):
            row = {'split': split, 'seed': seed, 'seed_offset': seed_offset, 'batch_idx': batch_idx, 'patch_idx': patch_idx, 'slide_id': slide_id}
            if coord_list is not None:
                xy = coord_list[patch_idx]
                row['x'] = int(xy[0])
                row['y'] = int(xy[1])
            rows.append(row)
    write_rows(path, rows)
def online_sampled_protocol(cfg: EvalConfig, *, scope: str) -> str:
    """Return sampled|full_grid for validation or test."""
    if scope == 'valid':
        return str(cfg.training.get('online_valid_protocol', 'sampled')).lower()
    else:
        if scope == 'test':
            return str(cfg.data.get('test_eval_protocol') or cfg.training.get('test_eval_protocol') or 'full_grid').lower()
        else:
            raise ValueError(f'unknown sampled protocol scope {scope!r}')
def uses_fixed_sampled_protocol(protocol: str) -> bool:
    return protocol in {'fixed_sampled', 'online_sampler', 'sampled', 'fast'}
def online_test_seed_offset(cfg: EvalConfig) -> int:
    if cfg.data.get('test_sampler_seed_offset') is not None:
        return int(cfg.data['test_sampler_seed_offset'])
    else:
        if cfg.training.get('test_sampler_seed_offset') is not None:
            return int(cfg.training['test_sampler_seed_offset'])
        else:
            return 20000
def online_official_seed(cfg: EvalConfig, exp: ExperimentConfig) -> int:
    if exp.official_sampling is not None:
        return int(exp.official_sampling.seed)
    else:
        return int(cfg.training.get('seed', 42))
def online_test_steps(cfg: EvalConfig) -> int:
    if cfg.data.get('online_test_steps') is not None:
        return int(cfg.data['online_test_steps'])
    else:
        if cfg.training.get('online_test_steps') is not None:
            return int(cfg.training['online_test_steps'])
        else:
            return int(cfg.training.get('online_valid_steps', 50))
@torch.no_grad()
def run_fixed_sampled_split_eval(decoder: nn.Module, encoder, criterion, cfg: EvalConfig, exp: ExperimentConfig, *, fixed_batches: list[dict[str, Any]], num_classes: int, device: torch.device, use_amp: bool, split: str, log: Callable[[str], None], write_outputs: bool=True) -> dict[str, float]:
    """Evaluate a frozen official-sampler batch list (valid or test)."""
    label_dict = {0: 'Background', 1: 'TLS', 2: 'Germinal Center', 3: 'Rest'} if cfg.task == 'tls_region_segmentation' else _pathocell_label_dict(cfg, exp)
    log(f"stage=eval protocol=fixed_sampled split={split} batches={len(fixed_batches)} patches={sum((int(b['semantic_mask'].shape[0]) for b in fixed_batches))}")
    metrics = evaluate_sampled(decoder, encoder, _NullBatchSource(), criterion, cfg, exp, num_classes=num_classes, device=device, steps=len(fixed_batches), use_amp=use_amp, fixed_batches=fixed_batches)
    slide_ids = {str(slide_id) for batch in fixed_batches for slide_id in batch.get('sample_name') or []}
    out = dict(metrics)
    out['slides'] = float(len(slide_ids))
    out['patches'] = float(sum((int(batch['semantic_mask'].shape[0]) for batch in fixed_batches)))
    out['protocol'] = 'fixed_sampled'
    if write_outputs:
        is_test = split == str(cfg.data.get('test_split', 'test'))
        result_dir = cfg.output_dir / ('test_result' if is_test else 'validation_result')
        result_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = cfg.output_dir / ('test_metrics.json' if is_test else f'{split}_metrics.json')
        with metrics_path.open('w', encoding='utf-8') as handle:
            json.dump(out, handle, indent=2)
        csv_name = str(cfg.data.get('test_result_name', 'hooknet_tls_test_metrics.csv')) if is_test else f'{split}_metrics.csv'
        _write_test_csv(result_dir / csv_name, _classwise_test_rows(out, label_dict, exclude_background=bool(cfg.metrics.get('exclude_background', True))))
    return out
class _NullBatchSource:
    def reset_epoch(self) -> None:
        return
    def close(self) -> None:
        return
    def next_batch(self) -> dict[str, Any]:
        raise RuntimeError('_NullBatchSource should not draw batches during fixed replay eval')
@torch.no_grad()
def evaluate_sampled(decoder: nn.Module, encoder, batch_source: OnlineBatchSource, criterion, cfg: EvalConfig, exp: ExperimentConfig, *, num_classes: int, device: torch.device, steps: int, use_amp: bool, fixed_batches: list[dict[str, Any]] | None=None) -> dict[str, float]:
    decoder.eval()
    original_class_weighting = getattr(criterion, 'class_weighting', None)
    if original_class_weighting is not None:
        criterion.class_weighting = False
    pfmb_instance = uses_pfmb_instance_primary_metric(cfg)
    pixel_metrics = None if pfmb_instance else SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=int(cfg.training['ignore_index']) if cfg.training.get('ignore_index') is not None else None, exclude_background=bool(cfg.metrics.get('exclude_background', True)))
    instance_metrics = InstanceLevelMetricAccumulator(label_dict=_pathocell_label_dict(cfg, exp)) if pfmb_instance else None
    losses = []
    if fixed_batches is not None:
        batches = fixed_batches
    else:
        batch_source.reset_epoch()
        batches = [batch_source.next_batch() for _ in range(int(steps))]
    for batch in batches:
        masks = batch['semantic_mask'].to(device, non_blocking=True)
        images = batch['image'].to(device, non_blocking=True)
        head_images = batch.get('task_image', batch['image']).to(device, non_blocking=True)
        features = encoder_features(encoder, images, device=device, use_amp=use_amp, input_size=cfg.training.get('encoder_input_size'))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = forward_segmentation_head(decoder, features, masks, images=head_images)
            loss = criterion(logits, masks)
        losses.append(float(loss.detach().cpu()))
        if pfmb_instance:
            instance_masks = batch.get('instance_mask')
            if instance_masks is None:
                raise ValueError('PathoCell PFMB validation requires instance_mask in sampled batches')
            else:
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                target = masks.detach().cpu().numpy()
                instances = instance_masks.detach().cpu().numpy()
                for i in range(probs.shape[0]):
                    instance_metrics.update(probs[i], target[i], instances[i])
        else:
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            target = masks.detach().cpu().numpy()
            for i in range(pred.shape[0]):
                pixel_metrics.update(pred[i], target[i])
    if original_class_weighting is not None:
        criterion.class_weighting = original_class_weighting
    out = instance_metrics.compute() if pfmb_instance else pixel_metrics.compute()
    out['loss'] = float(np.mean(losses)) if losses else 0.0
    return out
def train_one_online_step(decoder: nn.Module, encoder, batch: dict[str, Any], criterion, optimizer, scheduler, scaler: torch.amp.GradScaler, augmenter: nn.Module | None, *, device: torch.device, use_amp: bool, grad_clip_norm: float, loss_divisor: int=1, optimizer_step: bool=True, encoder_input_size: int | None=None, input_normalization: InputNormalization | None=None) -> tuple[float, float]:
    decoder.train()
    masks = batch['semantic_mask'].to(device, non_blocking=True)
    images = batch['image'].to(device, non_blocking=True)
    head_images = batch.get('task_image', batch['image']).to(device, non_blocking=True)
    instance_masks = batch.get('instance_mask')
    if instance_masks is not None:
        instance_masks = instance_masks.to(device, non_blocking=True)
    images, masks, instance_masks = maybe_augment(augmenter, images, masks, instance_masks, input_normalization=input_normalization)
    if augmenter is not None and 'task_image' in batch:
        raise ValueError('Shared-resolution task_image requires a joint dual-resolution augmentation implementation')
    else:
        sync_context = decoder.no_sync() if isinstance(decoder, DDP) and (not optimizer_step) else nullcontext()
        with sync_context:
            features = encoder_features(encoder, images, device=device, use_amp=use_amp, input_size=encoder_input_size)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = forward_segmentation_head(decoder, features, masks, images=head_images)
                loss = criterion(logits, masks)
            if torch.isnan(loss):
                raise FloatingPointError('NaN loss would break optimizer-update step semantics')
            else:
                scaler.scale(loss / float(loss_divisor)).backward()
        grad_norm = float('nan')
        if optimizer_step:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        return (float(loss.detach().cpu()), grad_norm)
def run_online_training(
    *,
    cfg: EvalConfig,
    exp: ExperimentConfig,
    train_source: OnlineBatchSource,
    valid_source: OnlineBatchSource,
    test_runner: Callable[..., dict[str, float]] | None,
    device: torch.device,
    train_split: str,
    valid_split: str,
    log: Callable[[str], None],
    test_only: bool = False,
    checkpoint: Path | None = None,
    valid_runner: Callable[..., dict[str, float]] | None = None,
    test_source: OnlineBatchSource | None = None,
    distributed: DistributedContext = DistributedContext(),
) -> dict[str, float] | None:
    """Train a frozen encoder with the configured dense-prediction head."""
    num_classes = resolve_num_classes(cfg, exp)
    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    log("stage=setup loading frozen encoder and preprocessing transform")
    encoder, transform = build_frozen_encoder(exp, device)
    input_normalization = find_input_normalization(transform)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    log("stage=setup frozen encoder ready")
    if input_normalization is None:
        log("stage=setup augmentation_domain=transformed_input normalization=none")
    else:
        log(
            "stage=setup augmentation_domain=raw_rgb_then_model_normalization "
            f"mean={input_normalization.mean} std={input_normalization.std}"
        )
    probe_batch = train_source.next_batch()
    encoder_input_size = cfg.training.get("encoder_input_size")
    first_features = encoder_features(
        encoder, probe_batch["image"], device=device, use_amp=use_amp, input_size=encoder_input_size
    )
    feature_shapes = {name: tuple(value.shape) for name, value in first_features.items()}
    log(
        f"stage=setup encoder_input_size={encoder_input_size or exp.target_patch_size} "
        f"task_output_size={exp.target_patch_size} feature_shapes={feature_shapes}"
    )
    decoder_layers = list(cfg.head_config.get("feature_layers") or exp.target_features)
    feature_processor = cfg.head_config.get("feature_processor")
    if feature_processor:
        processor_name = feature_processor.get("name") if isinstance(feature_processor, dict) else str(feature_processor)
        grids = [tuple(first_features[name].shape[-2:]) for name in decoder_layers]
        log(f"stage=setup feature_processor={processor_name} layers={decoder_layers} encoder_grids={grids}")
    elif str(cfg.head_config.get("decoder_mode", "")).lower() == "pfmb":
        grids = [tuple(first_features[name].shape[-2:]) for name in decoder_layers]
        log(f"stage=setup pfmb_decoder_routing={classify_encoder_pyramid(grids)} layers={decoder_layers} grids={grids}")
    log(f"stage=setup initializing head={cfg.head}")
    decoder = build_decoder(cfg, exp, first_features, num_classes=num_classes).to(device)
    if distributed.enabled:
        decoder = DDP(
            decoder,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
            output_device=distributed.local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    augmenter = build_augmenter(cfg.training)
    if augmenter is not None:
        augmenter = augmenter.to(device)
    criterion = _loss_fn(cfg, num_classes).to(device)
    out_dir = cfg.output_dir
    test_split = str(cfg.data.get("test_split", "test"))
    test_protocol = online_sampled_protocol(cfg, scope="test")
    sampled_test = uses_fixed_sampled_protocol(test_protocol)
    test_steps = online_test_steps(cfg) if sampled_test else 0
    frozen_test_batches: list[dict[str, Any]] | None = None

    def materialize_test_batches() -> list[dict[str, Any]]:
        if test_source is None:
            raise ValueError("fixed_sampled test evaluation requires a configured test data source")
        batches = materialize_online_valid_batches(
            test_source, steps=test_steps, log=log, label="test"
        )
        manifest_dir = out_dir / "test_result"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        save_online_draws_manifest(
            manifest_dir / "test_draws_manifest.csv", batches, split=test_split,
            seed=online_official_seed(cfg, exp), seed_offset=online_test_seed_offset(cfg),
        )
        return batches

    def evaluate_test(*, write_outputs: bool = True) -> dict[str, float]:
        nonlocal frozen_test_batches
        target_decoder = decoder.module if isinstance(decoder, DDP) else decoder
        original_sync = getattr(criterion, "distributed_sync", None)
        if original_sync is not None:
            criterion.distributed_sync = False
        try:
            if sampled_test:
                if frozen_test_batches is None:
                    frozen_test_batches = materialize_test_batches()
                return run_fixed_sampled_split_eval(
                    target_decoder, encoder, criterion, cfg, exp,
                    fixed_batches=frozen_test_batches, num_classes=num_classes,
                    device=device, use_amp=use_amp, split=test_split, log=log,
                    write_outputs=write_outputs,
                )
            if test_runner is None:
                raise ValueError("full_grid test evaluation requires a test runner")
            return test_runner(
                decoder=target_decoder, encoder=encoder, transform=transform, exp=exp,
                cfg=cfg, criterion=criterion, device=device, test_split=test_split,
                use_amp=use_amp, log=log,
            )
        finally:
            if original_sync is not None:
                criterion.distributed_sync = original_sync

    def close_sources() -> None:
        seen: set[int] = set()
        for batch_source in (train_source, valid_source, test_source):
            if batch_source is not None and id(batch_source) not in seen:
                seen.add(id(batch_source))
                close = getattr(batch_source, "close", None)
                if callable(close):
                    close()

    if test_only:
        best_path = checkpoint or out_dir / "best_model.pt"
        if not best_path.is_file():
            close_sources()
            raise FileNotFoundError(f"--test-only requires checkpoint at {best_path}")
        payload = torch.load(best_path, map_location=device)
        target_decoder = decoder.module if isinstance(decoder, DDP) else decoder
        target_decoder.load_state_dict(payload["model_state_dict"])
        log(f"stage=test_only checkpoint={best_path} step={payload.get('step')} protocol={test_protocol}")
        try:
            result = evaluate_test() if distributed.is_main else None
            distributed.barrier()
            return result
        finally:
            close_sources()

    optimizer = _make_optimizer(cfg, decoder)
    scheduler = _make_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_clip_norm = float(cfg.training.get("grad_clip_norm", 2.0))
    grad_accum = max(1, int(cfg.training.get("gradient_accumulation", cfg.training.get("gc", 1))))
    training_steps = max(1, int(cfg.training.get("training_steps", 5000)))
    val_interval = max(1, int(cfg.training.get("val_interval", 250)))
    log_interval = max(1, int(cfg.training.get("log_interval", 10)))
    valid_steps = max(1, int(cfg.training.get("online_valid_steps", 50)))
    primary_metric = str(cfg.training.get("primary_metric", "f1_macro"))
    pfmb_instance = uses_pfmb_instance_primary_metric(cfg)
    best_score = -float("inf")
    best_step = 0
    steps_since_last_best = 0
    early_stopping = int(cfg.training.get("early_stopping", 0))
    checkpoint_start = int(cfg.training.get("checkpoint_start_step", 0))
    skip_test = bool(cfg.training.get("skip_test", False))
    profile_steps = bool(cfg.training.get("online_profile_steps", False))
    history_path = out_dir / "history.csv"
    history: list[dict[str, Any]] = []
    frozen_valid_batches = None
    if valid_runner is None:
        frozen_valid_batches = materialize_online_valid_batches(
            valid_source, steps=valid_steps, log=log, label="validation"
        )
    if sampled_test:
        frozen_test_batches = materialize_test_batches()
    train_source.reset_epoch(0)
    losses: list[float] = []
    grad_norms: list[float] = []
    learning_rate_curve: list[dict[str, float]] = []
    scheduler_updates = validation_runs = checkpoint_evaluations = completed_steps = 0
    log(
        f"stage=train start task={cfg.task} head={cfg.head} model={exp.model} "
        f"training_steps={training_steps} val_interval={val_interval} "
        f"online_valid_steps={valid_steps} early_stopping={early_stopping} "
        f"gradient_accumulation={grad_accum} world_size={distributed.world_size} "
        f"test_protocol={test_protocol}"
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        for global_step in range(1, training_steps + 1):
            t_start = time.perf_counter() if profile_steps else 0.0
            micro_losses: list[float] = []
            grad_norm = float("nan")
            for micro_step in range(grad_accum):
                batch = train_source.next_batch()
                loss_value, step_grad_norm = train_one_online_step(
                    decoder, encoder, batch, criterion, optimizer, scheduler, scaler, augmenter,
                    device=device, use_amp=use_amp, grad_clip_norm=grad_clip_norm,
                    loss_divisor=grad_accum, optimizer_step=micro_step == grad_accum - 1,
                    encoder_input_size=encoder_input_size, input_normalization=input_normalization,
                )
                micro_losses.append(loss_value)
                if np.isfinite(step_grad_norm):
                    grad_norm = step_grad_norm
            t_data = time.perf_counter() if profile_steps else 0.0
            completed_steps = global_step
            scheduler_updates += 1
            loss_value = float(np.mean(micro_losses))
            learning_rate_curve.append(
                {"global_step": global_step, "learning_rate": float(optimizer.param_groups[0]["lr"])}
            )
            if profile_steps and device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter() if profile_steps else 0.0
            losses.append(loss_value)
            if np.isfinite(grad_norm):
                grad_norms.append(grad_norm)
            if global_step % log_interval == 0:
                message = f"stage=train global_step={global_step} train_loss={np.mean(losses):.4f} lr={optimizer.param_groups[0]['lr']:.6g}"
                if profile_steps:
                    message += f" data_s={t_data-t_start:.2f} train_s={t_end-t_data:.2f} total_s={t_end-t_start:.2f}"
                log(message)
            if global_step % val_interval:
                continue
            validation_runs += 1
            checkpoint_evaluations += 1
            if valid_runner is not None:
                log(f"stage=validation global_step={global_step} protocol=full_grid split={valid_split}")
                valid_metrics = valid_runner(
                    decoder=decoder, encoder=encoder, exp=exp, cfg=cfg, criterion=criterion,
                    device=device, use_amp=use_amp, log=log,
                )
            else:
                valid_metrics = evaluate_sampled(
                    decoder, encoder, valid_source, criterion, cfg, exp,
                    num_classes=num_classes, device=device, steps=valid_steps,
                    use_amp=use_amp, fixed_batches=frozen_valid_batches,
                )
            fields = aggregate_pfmb_instance_metrics(valid_metrics) if pfmb_instance else _aggregate_metrics(valid_metrics)
            if not pfmb_instance:
                fields.update({key: float(value) for key, value in valid_metrics.items() if key.startswith("class_")})
            row = {
                "epoch": len(history), "step": global_step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "learning_rate": float(optimizer.param_groups[0]["lr"]), **fields,
            }
            history.append(row)
            if distributed.is_main:
                write_rows(history_path, history)
                write_rows(out_dir / "validation_result" / "validation_metrics.csv", history)
            score = float(valid_metrics.get(primary_metric, -valid_metrics.get("loss", 0.0)))
            if distributed.enabled:
                score_tensor = torch.tensor([score], dtype=torch.float64, device=device)
                dist.broadcast(score_tensor, src=0)
                score = float(score_tensor.item())
            valid_log = (
                format_pfmb_instance_validation_log(valid_metrics)
                if pfmb_instance
                else f"valid_{primary_metric}={valid_metrics.get(primary_metric, 0.0):.4f} valid_f1_macro={valid_metrics.get('f1_macro', 0.0):.4f} valid_iou_macro={valid_metrics.get('iou_macro', 0.0):.4f}"
            )
            log(f"stage=validation global_step={global_step} train_loss={row['train_loss']:.4f} {valid_log}")
            steps_since_last_best += val_interval
            eligible = global_step >= checkpoint_start
            stop = False
            if eligible and score > best_score:
                best_score = score
                best_step = global_step
                steps_since_last_best = 0
                if distributed.is_main:
                    target_decoder = decoder.module if isinstance(decoder, DDP) else decoder
                    torch.save(
                        {"model_state_dict": target_decoder.state_dict(), "step": global_step,
                         "score": score, "head": cfg.head, "task": cfg.task,
                         "feature_layers": exp.target_features,
                         "protocol": "online_frozen_encoder_unetr"},
                        out_dir / "best_model.pt",
                    )
                    write_rows(out_dir / "validation_result" / "best_validation.csv", [row])
                log(f"stage=validation new_best global_step={global_step} {primary_metric}={score:.4f}")
            elif early_stopping > 0 and steps_since_last_best >= early_stopping:
                stop = True
                log(f"stage=train early_stop global_step={global_step} {primary_metric} best={best_score:.4f} at step={best_step}")
            losses = []
            grad_norms = []
            if broadcast_bool(stop, device=device):
                break
    finally:
        for batch_source in (train_source, valid_source, test_source):
            close = getattr(batch_source, "close", None) if batch_source is not None else None
            if callable(close):
                close()

    distributed.barrier()
    if not distributed.is_main:
        distributed.barrier()
        return None
    ema_value = getattr(criterion, "ema_updates", torch.tensor(0))
    ema_updates = int(ema_value.item()) if hasattr(ema_value, "item") else int(ema_value)
    loss_cfg = dict(cfg.training.get("loss_fn") or {})
    ema_enabled = bool(loss_cfg.get("class_weighting", False))
    online = dict(cfg.data.get("online_sampling") or {})
    configured_batch = int(
        cfg.training.get("global_batch_size") or online.get("global_batch_size")
        or int(cfg.training.get("batch_size", 1)) * grad_accum * distributed.world_size
    )
    semantics = {
        "world_size": distributed.world_size,
        "global_batch_size": configured_batch,
        "per_gpu_batch_size": configured_batch // distributed.world_size,
        "optimizer_updates": completed_steps,
        "scheduler_updates": scheduler_updates,
        "ema_updates": ema_updates,
        "ema_updates_expected": ema_enabled,
        "validation_runs": validation_runs,
        "checkpoint_evaluations": checkpoint_evaluations,
        "configured_training_steps": training_steps,
        "step_unit": "optimizer_update",
        "gradient_accumulation": grad_accum,
        "micro_batches_consumed": completed_steps * grad_accum,
        "invariant_holds": completed_steps == scheduler_updates and (ema_updates == completed_steps * grad_accum if ema_enabled else ema_updates == 0),
    }
    (out_dir / "trainer_semantics_verification.json").write_text(json.dumps(semantics, indent=2), encoding="utf-8")
    write_rows(out_dir / "learning_rate_curve.csv", learning_rate_curve)
    if not semantics["invariant_holds"]:
        raise RuntimeError(f"Trainer step invariant failed: {semantics}")
    best_path = out_dir / "best_model.pt"
    if not best_path.is_file():
        target_decoder = decoder.module if isinstance(decoder, DDP) else decoder
        torch.save({"model_state_dict": target_decoder.state_dict(), "step": completed_steps, "score": best_score, "head": cfg.head, "task": cfg.task, "feature_layers": exp.target_features, "protocol": "online_frozen_encoder_unetr"}, best_path)
    if skip_test:
        log("skip_test=true; finished after training/validation")
        distributed.barrier()
        return None
    if not sampled_test and test_runner is None:
        log("no full_grid test runner configured; finished after training/validation")
        distributed.barrier()
        return None
    payload = torch.load(best_path, map_location=device)
    target_decoder = decoder.module if isinstance(decoder, DDP) else decoder
    target_decoder.load_state_dict(payload["model_state_dict"])
    log(f"starting test split={test_split} checkpoint={best_path} protocol={test_protocol}")
    result = evaluate_test()
    distributed.barrier()
    return result
def wrap_prefetch(source: OnlineBatchSource, prefetch_batches: int) -> OnlineBatchSource:
    if isinstance(source, PrefetchingBatchSource):
        return source
    else:
        return PrefetchingBatchSource(source, prefetch_batches=prefetch_batches)
def _limit_cpu_threads(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(num_threads)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, num_threads)))
def _online_valid_protocol(cfg: EvalConfig) -> str:
    return "full_grid"
def _maybe_valid_runner(cfg: EvalConfig, runner):
    return runner
_TLS_SEGMENTATION_HEADS = frozenset({'feature_unetr', 'unetr'})
_PATHOCELL_SEGMENTATION_HEADS = frozenset({'unetr'})
def _assert_tls_segmentation_head(cfg: EvalConfig) -> None:
    if cfg.task != "tls_region_segmentation":
        return
    if cfg.head != "unetr":
        raise ValueError("HookNet-TLS segmentation requires head='unetr'.")
    if resolve_decoder_mode(cfg.head, cfg.head_config) != "feature":
        raise ValueError("HookNet-TLS segmentation requires decoder_mode='feature'.")


def _assert_pathocell_segmentation_head(cfg: EvalConfig) -> None:
    if cfg.task != "cell_semantic_segmentation":
        return
    if cfg.head != "unetr":
        raise ValueError("PathoCell segmentation requires head='unetr'.")
    mode = resolve_decoder_mode(cfg.head, cfg.head_config)
    processor = cfg.head_config.get("feature_processor")
    processor_name = str(processor.get("name", "")).strip().lower() if isinstance(processor, dict) else ""
    if mode == "pfmb" or (mode == "feature" and processor_name in {"cellvit_pyramid", "cellvit"}):
        return
    raise ValueError("PathoCell segmentation requires decoder_mode='pfmb', or feature mode with cellvit_pyramid.")
def _resolve_bundle(args) -> tuple[ExperimentConfig, EvalConfig]:
    if args.config is None:
        raise ValueError("Pass the unified experiment/evaluation YAML with --config.")
    return load_online_eval_bundle(args.config)
def _build_tls_sources(exp, cfg, transform, device, train_split: str, valid_split: str):
    from data.hooknet_tls.sampler import FixedGridTileSampler

    del device
    sampling = exp.fixed_grid_sampling
    if sampling is None or sampling.fixed_grid_root is None:
        raise ValueError("TLS training requires fixed_grid_sampling.fixed_grid_root.")
    training = dict(cfg.training or {})
    balanced = bool(training.get("uniform_class_sampling", False))
    eval_workers = max(1, min(4, int(exp.iterator_cpus)))

    def make_sampler(split, seed_offset, workers, *, class_balanced):
        return FixedGridTileSampler(
            exp, sampling, image_transform=transform, split=split,
            seed_offset=seed_offset, patch_workers=workers,
            class_balanced=class_balanced,
        )

    train_sampler = make_sampler(train_split, 0, exp.iterator_cpus, class_balanced=balanced)
    valid_sampler = make_sampler(valid_split, 10000, eval_workers, class_balanced=False)
    test_split = str(cfg.data.get("test_split") or "test")
    test_sampler = make_sampler(test_split, online_test_seed_offset(cfg), eval_workers, class_balanced=False)

    def valid_runner(**kwargs):
        return _tls_png_full_grid_eval(
            kwargs["decoder"], kwargs["encoder"], kwargs.get("transform"),
            kwargs["exp"], kwargs["cfg"], kwargs["criterion"], kwargs["device"],
            valid_split, kwargs["use_amp"], kwargs["log"],
            sampler=valid_sampler, write_outputs=False,
        )

    def test_runner(decoder, encoder, image_transform, experiment, evaluation_config,
                    criterion, eval_device, split, use_amp, log):
        return _tls_png_full_grid_eval(
            decoder, encoder, image_transform, experiment, evaluation_config,
            criterion, eval_device, split, use_amp, log,
            sampler=test_sampler, write_outputs=True,
        )

    return (
        wrap_prefetch(train_sampler, sampling.prefetch_batches),
        valid_sampler,
        _maybe_valid_runner(cfg, valid_runner),
        test_runner,
        test_sampler,
    )
def _build_pathocell_sources(exp, cfg, transform, device, train_split: str,
                             valid_split: str, distributed: DistributedContext):
    from data.pathocell.sampler import HDFTileBatchSampler
    from data.loader import load_split_samples
    from training.online_segmentation_test import run_pathocell_grid_eval, run_pathocell_grid_test

    del device
    online = dict(cfg.data.get("online_sampling") or {})
    training = dict(cfg.training or {})
    world_size = max(1, int(distributed.world_size))
    global_batch = int(online.get("global_batch_size") or training.get("global_batch_size") or online.get("batch_size") or training.get("batch_size") or 32)
    if distributed.enabled and global_batch % world_size:
        raise ValueError(f"global_batch_size={global_batch} must divide world_size={world_size}")
    batch_size = global_batch // world_size if distributed.enabled else int(online.get("batch_size") or training.get("batch_size") or global_batch)
    loss_cfg = dict(training.get("loss_fn") or {})
    exclude_classes = list(loss_cfg.get("exclude_classes") or [0])
    balanced = bool(training.get("uniform_class_sampling", False))
    sample_excluded = bool(training.get("sample_excluded_classes", False))
    num_classes = resolve_num_classes(cfg, exp)
    seed = int(online.get("seed") or training.get("seed") or 42)
    cache_size = int(online.get("volume_cache_size") or 2)
    train_samples = load_split_samples(exp.dataset_config, split_filter=str(train_split))
    test_split = str(cfg.data.get("test_split") or "test")
    eval_batch_size = int(cfg.data.get("test_batch_size") or max(1, min(batch_size, 4)))
    eval_workers = max(1, min(4, int(getattr(exp, "iterator_cpus", 1) or 1)))

    def make_sampler(split, seed_offset, *, is_training=False):
        samples = train_samples if is_training else load_split_samples(exp.dataset_config, split_filter=str(split))
        return HDFTileBatchSampler(
            exp, samples, image_transform=transform,
            batch_size=batch_size if is_training else eval_batch_size,
            seed=seed + int(seed_offset), max_cached_volumes=cache_size,
            patch_workers=int(getattr(exp, "iterator_cpus", 1) or 1) if is_training else eval_workers,
            uniform_class_sampling=balanced if is_training else False,
            num_classes=num_classes if is_training else None,
            exclude_classes=exclude_classes if is_training else None,
            sample_excluded_classes=sample_excluded if is_training else False,
            rank=distributed.rank if distributed.enabled else 0,
            world_size=world_size if distributed.enabled else 1,
        )

    train_sampler = make_sampler(train_split, 0, is_training=True)
    valid_sampler = make_sampler(valid_split, 10000)
    test_sampler = make_sampler(test_split, online_test_seed_offset(cfg))

    def valid_runner(**kwargs):
        return run_pathocell_grid_eval(
            kwargs["decoder"], kwargs["encoder"], kwargs.get("transform"),
            kwargs["exp"], kwargs["cfg"], kwargs["criterion"], kwargs["device"],
            valid_split, kwargs["use_amp"], kwargs["log"],
            write_outputs=False, sampler=valid_sampler,
        )

    def test_runner(decoder, encoder, image_transform, experiment, evaluation_config,
                    criterion, eval_device, split, use_amp, log):
        return run_pathocell_grid_test(
            decoder, encoder, image_transform, experiment, evaluation_config,
            criterion, eval_device, split, use_amp, log,
            write_outputs=True, sampler=test_sampler,
        )

    return (
        wrap_prefetch(train_sampler, int(online.get("prefetch_batches") or 1))
        if not distributed.enabled else train_sampler,
        valid_sampler, _maybe_valid_runner(cfg, valid_runner), test_runner, test_sampler,
    )
def build_task_sources(task: str, exp, cfg, device, train_split: str,
                       valid_split: str, distributed: DistributedContext = DistributedContext()):
    transform = get_custom_transformer(
        exp.model,
        use_benchmark_config=exp.use_benchmark_config,
        config_path=str(exp.benchmark_config_path) if exp.benchmark_config_path else None,
    )
    if task == "tls_region_segmentation":
        return _build_tls_sources(exp, cfg, transform, device, train_split, valid_split)
    if task == "cell_semantic_segmentation":
        return _build_pathocell_sources(exp, cfg, transform, device, train_split, valid_split, distributed)
    raise ValueError(f"Unsupported online task {task!r}")
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Unified downstream YAML")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--valid-split", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    exp, cfg = _resolve_bundle(args)
    train_split = str(args.train_split or cfg.data.get("train_split") or "train")
    valid_split = str(args.valid_split or cfg.data.get("valid_split") or "valid")
    _assert_tls_segmentation_head(cfg)
    _assert_pathocell_segmentation_head(cfg)
    if cfg.task == "cell_semantic_segmentation":
        tiling = str(cfg.data.get("pathocell_tiling", exp.pathocell_tiling)).strip().lower()
        if tiling != "pfmb":
            raise ValueError("PathoCell benchmark protocol requires pathocell_tiling='pfmb'.")
    _limit_cpu_threads(exp.cpu_threads)
    distributed, device = initialize_distributed(args.device or cfg.training.get("device"))
    seed = int(cfg.training.get("seed", 42)) + distributed.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        if distributed.is_main:
            _prepare_online_output_dirs(exp, cfg)
        distributed.barrier()
        output_dirs = {
            "validation": cfg.output_dir / "validation_result",
            "test": cfg.output_dir / "test_result",
            "logs": cfg.output_dir / "validation_result" / "logs",
        }
        train_log = output_dirs["logs"] / "train_online.log"

        def log(message: str) -> None:
            if distributed.is_main:
                print(message, flush=True)
                with train_log.open("a", encoding="utf-8") as handle:
                    handle.write(message + "\n")

        log(f"online_segmentation output_dir={cfg.output_dir}")
        log(
            f"protocol=online_frozen_encoder_unetr task={cfg.task} head={cfg.head} "
            f"model={exp.model} target_features={exp.target_features} "
            f"valid_protocol={_online_valid_protocol(cfg)} "
            f"test_protocol={online_sampled_protocol(cfg, scope='test')}"
        )
        train_source, valid_source, valid_runner, test_runner, test_source = build_task_sources(
            cfg.task, exp, cfg, device, train_split, valid_split, distributed
        )
        result = run_online_training(
            cfg=cfg, exp=exp, train_source=train_source, valid_source=valid_source,
            test_runner=test_runner, test_source=test_source, valid_runner=valid_runner,
            device=device, train_split=train_split, valid_split=valid_split, log=log,
            test_only=args.test_only, checkpoint=args.checkpoint, distributed=distributed,
        )
        if result is not None and distributed.is_main:
            if uses_pfmb_instance_primary_metric(cfg):
                log(f"finished online training+test {format_pfmb_instance_validation_log(result)}")
            else:
                log(
                    f"finished online training+test f1_macro={result.get('f1_macro', 0.0):.4f} "
                    f"iou_macro={result.get('iou_macro', 0.0):.4f} patches={int(result.get('patches', 0.0))}"
                )
    finally:
        destroy_distributed(distributed)


if __name__ == "__main__":
    main()
if __name__ == '__main__':
    main()
