"""PathoCell / Lizard MifUNet training (HE → frozen generator → UNet).

Same generator path as virtual staining and TLS output-layer training: RGB
tiles in, frozen HE→mIF model, then a trainable UNet. No pre-extracted
``fov_mif`` cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data.loader import coerce_label_dict_keys, load_split_samples
from data.pathocell_hdf import (
    PATHOCELL_NATIVE_MPP,
    extract_pathocell_tile_triple_at_np,
    is_pfmb_tiling,
    load_pathocell_volume,
    pathocell_tile_geometry,
    plan_pathocell_tiles,
    resolve_pathocell_tiling,
)
from data.pathocell_online_sampler import PathoCellOnlineSampler
from heads import build_segmentation_head, forward_segmentation_head
from heads.mif_unet import MifUNet
from mif.common import (
    estimate_channel_stats,
    eval_config_from_raw,
    is_finite_loss,
    load_channel_stats,
    load_config_dict,
    make_ce_criterion,
    make_scheduler,
    save_channel_stats,
    warmup_factor,
)
from models.mif_output import FrozenMifGenerator, load_frozen_mif_generator, resolve_mif_channels
from training.augmentations import build_augmenter
from utils.experiment_config import ExperimentConfig, load_experiment_config

LOGGER = logging.getLogger("mif_unet_train")

PATHOCELL_NUM_CLASSES = 15


def _he_hwc_to_tensor(image_hwc: np.ndarray, transform) -> torch.Tensor:
    rgb = np.ascontiguousarray(image_hwc[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if transform is None:
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return transform(Image.fromarray(rgb))


def build_mif_unet_from_cfg(
    head_config: dict[str, Any],
    *,
    model_name: str,
    num_classes: int,
    generator: FrozenMifGenerator | None,
) -> MifUNet:
    cfg = dict(head_config)
    cfg.setdefault("decoder_mode", "mif_unet")
    cfg.setdefault("in_channels", resolve_mif_channels(model_name, cfg.get("in_channels")))
    cfg["num_classes"] = int(num_classes)
    cfg["generator"] = generator
    return build_segmentation_head(
        "unetr",
        feature_layers=cfg.get("feature_layers"),
        in_channels=int(cfg["in_channels"]),
        num_classes=int(num_classes),
        head_config=cfg,
    )


def _maybe_augment(batch: dict[str, Any], augmenter, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["semantic_mask"].to(device, non_blocking=True)
    if augmenter is None:
        return images, masks
    out = augmenter(images, masks)
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return images, masks


@torch.no_grad()
def estimate_generated_channel_stats(
    model: MifUNet,
    sampler,
    *,
    device: torch.device,
    num_batches: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.generator is None:
        raise ValueError("generator required to estimate online channel stats")
    batches: list[torch.Tensor] = []
    model.eval()
    model.generator.train(False)
    for _ in range(max(1, int(num_batches))):
        batch = sampler.next_batch()
        rgb = batch["image"].to(device, non_blocking=True)
        stack = model.generator(rgb)
        batches.append(stack.detach().float().cpu())
    return estimate_channel_stats(batches)


@torch.no_grad()
def run_pathocell_mif_grid_eval(
    model: nn.Module,
    exp: ExperimentConfig,
    *,
    transform,
    split: str,
    device: torch.device,
    batch_size: int = 16,
    use_amp: bool = True,
    exclude_background: bool = True,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Fixed-grid FOV eval with PathoCell/Lizard instance F1 (same as encoder cellseg)."""
    from metrics import InstanceLevelMetricAccumulator
    from training.config import EvalConfig
    from training.online_segmentation_train import resolve_num_classes

    del exclude_background
    cfg = EvalConfig(
        "tmp",
        "cell_semantic_segmentation",
        exp.dataset,
        "unetr",
        Path("/tmp"),
        {},
        {},
        {"num_classes": PATHOCELL_NUM_CLASSES},
        {},
        {},
    )
    try:
        num_classes = int(resolve_num_classes(cfg, exp))
    except Exception:
        num_classes = PATHOCELL_NUM_CLASSES

    label_dict = coerce_label_dict_keys(dict(exp.dataset_config.label_dict))
    if not label_dict or not all(isinstance(k, int) for k in label_dict):
        raise RuntimeError(
            "cellseg instance eval needs int class-id keys in label_dict "
            f"(JSON string keys yield f1_score_macro=0); got {list(label_dict)[:8]!r}"
        )
    metrics = InstanceLevelMetricAccumulator(label_dict=label_dict)
    samples = load_split_samples(exp.dataset_config, split_filter=split)
    read_size, output_size = pathocell_tile_geometry(
        exp.target_patch_size,
        exp.target_mpp if exp.use_mpp else None,
        native_mpp=(
            exp.dataset_config.native_mpp
            if exp.dataset_config.native_mpp is not None
            else PATHOCELL_NATIVE_MPP
        ),
    )
    tiling = resolve_pathocell_tiling(exp)
    renumber = is_pfmb_tiling(tiling)
    model.eval()
    if getattr(model, "generator", None) is not None:
        model.generator.train(False)
    losses: list[float] = []

    batch_imgs: list[torch.Tensor] = []
    batch_masks: list[np.ndarray] = []
    batch_instances: list[np.ndarray] = []

    def flush() -> None:
        if not batch_imgs:
            return
        images = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        target = torch.as_tensor(np.stack(batch_masks, axis=0), dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = forward_segmentation_head(model, features=None, masks=target, images=images)
            if criterion is not None:
                loss = criterion(logits, target)
                losses.append(float(loss.detach().mean().cpu()))
            probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
        semantic_np = target.detach().cpu().numpy()
        for i in range(probs.shape[0]):
            metrics.update(probs[i], semantic_np[i], batch_instances[i])
        batch_imgs.clear()
        batch_masks.clear()
        batch_instances.clear()

    n_tiles = 0
    for sample in samples:
        if sample.image_path is None or not Path(sample.image_path).is_file():
            continue
        image_hwc, semantic_hw, instance_hw = load_pathocell_volume(sample.image_path)
        if semantic_hw is None or instance_hw is None:
            continue
        _, tiles = plan_pathocell_tiles(
            image_hwc.shape[0], image_hwc.shape[1], read_size, protocol=tiling
        )
        for tile in tiles:
            he, semantic, instance = extract_pathocell_tile_triple_at_np(
                image_hwc,
                semantic_hw,
                instance_hw,
                tile,
                read_size,
                output_size,
                protocol=tiling,
                renumber_instances=renumber,
            )
            if semantic is None or instance is None:
                continue
            batch_imgs.append(_he_hwc_to_tensor(he, transform))
            batch_masks.append(semantic.astype(np.int64, copy=False))
            batch_instances.append(instance.astype(np.int64, copy=False))
            n_tiles += 1
            if len(batch_imgs) >= batch_size:
                flush()
    flush()
    if n_tiles == 0:
        raise RuntimeError(f"no eval tiles for split={split} dataset={exp.dataset}")

    raw_metrics = metrics.compute()
    out = {str(k): float(v) for k, v in raw_metrics.items()}
    if losses:
        out["loss"] = float(sum(losses) / len(losses))
    if "f1_score_macro" not in out:
        raise RuntimeError(f"instance eval missing f1_score_macro; keys={sorted(out)}")
    LOGGER.info(
        "pfmb instance eval split=%s n_tiles=%s num_classes=%s f1_score_macro=%.4f instance_count=%s",
        split,
        n_tiles,
        num_classes,
        out["f1_score_macro"],
        out.get("instance_count"),
    )
    return out


def train_mif_unet(raw: dict[str, Any], *, log: logging.Logger | None = None) -> dict[str, Any]:
    log = log or LOGGER
    exp_path = raw.get("_experiment_config_path")
    if exp_path:
        exp = load_experiment_config(exp_path)
    else:
        from training.online_segmentation_train import load_online_eval_bundle
        import yaml as _yaml

        tmp = Path("/tmp") / f"_mif_unet_exp_{int(time.time() * 1000)}.yaml"
        tmp.write_text(_yaml.safe_dump(raw), encoding="utf-8")
        try:
            exp, cfg_bundle = load_online_eval_bundle(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        _ = cfg_bundle

    cfg = eval_config_from_raw(raw, fallback_id=str(raw.get("evaluation", {}).get("id", "mif_unet")))
    training = dict(cfg.training)
    data_cfg = dict(cfg.data)
    head_cfg = dict(cfg.head_config)
    run = dict(raw.get("run") or {})

    device = torch.device(str(training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")))
    seed = int(training.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = str(getattr(exp, "model", "") or head_cfg.get("mif_model") or "").lower()
    if not model_name:
        raise ValueError("experiment.model (or head.mif_model) is required")
    in_channels = resolve_mif_channels(model_name, head_cfg.get("in_channels"))
    head_cfg["in_channels"] = in_channels
    head_cfg.setdefault("decoder_mode", "mif_unet")

    try:
        from training.online_segmentation_train import resolve_num_classes

        num_classes = int(training.get("num_classes") or resolve_num_classes(cfg, exp))
    except Exception:
        num_classes = int(training.get("num_classes") or PATHOCELL_NUM_CLASSES)
    log.info("num_classes=%s", num_classes)
    output_dir = Path(run.get("output_dir") or cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = str(data_cfg.get("train_split") or "train")
    valid_split = str(data_cfg.get("valid_split") or "valid")
    test_split = str(data_cfg.get("test_split") or "test")
    online = dict(data_cfg.get("online_sampling") or {})
    batch_size = int(training.get("batch_size") or online.get("batch_size") or 4)

    generator, transform = load_frozen_mif_generator(
        model_name,
        device,
        layer=head_cfg.get("mif_layer") or data_cfg.get("mif_layer"),
        gpu_num=0,
        use_benchmark_config=False,
        config_path=run.get("benchmark_config_path") or getattr(exp, "benchmark_config_path", None),
    )

    train_samples = load_split_samples(exp.dataset_config, split_filter=train_split)
    exclude = list((training.get("loss_fn") or {}).get("exclude_classes") or [0])
    train_sampler = PathoCellOnlineSampler(
        exp,
        train_samples,
        transform=transform,
        batch_size=batch_size,
        seed=seed,
        volume_cache_size=int(online.get("volume_cache_size") or 4),
        uniform_class_sampling=bool(training.get("uniform_class_sampling", True)),
        num_classes=num_classes,
        exclude_classes=exclude,
        sample_excluded_classes=bool(training.get("sample_excluded_classes", False)),
    )

    model = build_mif_unet_from_cfg(
        head_cfg, model_name=model_name, num_classes=num_classes, generator=generator
    )
    model.to(device)
    if model.generator is not None:
        model.generator.train(False)

    stats_path = output_dir / "channel_stats.json"
    if stats_path.is_file() and not bool(training.get("recompute_channel_stats", False)):
        mean, std = load_channel_stats(stats_path)
        log.info("loaded channel stats from %s", stats_path)
    else:
        n_stat = int(training.get("channel_stats_batches", 16))
        log.info("estimating generated channel stats over %s train batches", n_stat)
        mean, std = estimate_generated_channel_stats(
            model, train_sampler, device=device, num_batches=n_stat
        )
        save_channel_stats(stats_path, mean, std)
    model.set_channel_stats(mean, std)

    try:
        from training.online_segmentation_train import _loss_fn

        criterion = _loss_fn(cfg, num_classes)
    except Exception:
        criterion = make_ce_criterion(training, num_classes)
    criterion.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = dict(training.get("optimizer") or {})
    name = str(opt_cfg.get("name") or "AdamW")
    params = dict(opt_cfg.get("params") or {"lr": 3e-4, "weight_decay": 0.01})
    optimizer = getattr(torch.optim, name)(trainable, **params)
    scheduler = make_scheduler(training, optimizer)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    warmup_steps = int(opt_cfg.get("warmup_steps") or 0)

    augmenter = build_augmenter(training)
    steps = int(training.get("training_steps") or 1000)
    val_interval = int(training.get("val_interval") or 200)
    log_interval = int(training.get("log_interval") or 10)
    early_stopping = training.get("early_stopping")
    early_stopping = int(early_stopping) if early_stopping is not None else None
    grad_clip = training.get("grad_clip_norm")
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    primary = str(training.get("primary_metric") or "f1_score_macro")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid = -1.0
    best_path = output_dir / "best.pt"
    last_improve_step = 0
    history: list[dict[str, Any]] = []

    model.train()
    if model.generator is not None:
        model.generator.train(False)

    t0 = time.time()
    for step in range(steps):
        batch = train_sampler.next_batch()
        images, masks = _maybe_augment(batch, augmenter, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = forward_segmentation_head(model, features=None, masks=masks, images=images)
            loss = criterion(logits, masks)
        if not is_finite_loss(loss):
            log.warning("non-finite loss at step %s; skipping", step)
            continue
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        warmup_factor(step, warmup_steps, base_lrs, optimizer)
        if scheduler is not None and step >= warmup_steps:
            scheduler.step()

        if step % log_interval == 0 or step == steps - 1:
            log.info("step %s/%s loss=%.4f lr=%.2e", step, steps, float(loss), optimizer.param_groups[0]["lr"])

        if val_interval > 0 and (step % val_interval == 0 or step == steps - 1):
            valid_metrics = run_pathocell_mif_grid_eval(
                model,
                exp,
                transform=transform,
                split=valid_split,
                device=device,
                batch_size=int(data_cfg.get("test_batch_size") or batch_size),
                use_amp=use_amp,
                exclude_background=bool(cfg.metrics.get("exclude_background", True)),
                criterion=criterion,
            )
            if primary not in valid_metrics:
                raise RuntimeError(
                    f"validation missing {primary!r} (refusing TLS pixel fallback); "
                    f"keys={sorted(valid_metrics)}"
                )
            score = float(valid_metrics[primary])
            history.append({"step": step, "train_loss": float(loss), "valid": valid_metrics})
            log.info("valid step=%s %s=%.4f metrics=%s", step, primary, score, valid_metrics)
            if score > best_valid:
                best_valid = score
                last_improve_step = step
                torch.save(
                    {
                        "model_state_dict": {
                            k: v
                            for k, v in model.state_dict().items()
                            if not k.startswith("generator.")
                        },
                        "step": step,
                        "best_valid": best_valid,
                        "primary_metric": primary,
                        "channel_mean": mean,
                        "channel_std": std,
                        "head_config": {k: v for k, v in head_cfg.items() if k != "generator"},
                    },
                    best_path,
                )
            model.train()
            if model.generator is not None:
                model.generator.train(False)
            if early_stopping is not None and (step - last_improve_step) >= early_stopping:
                log.info("early stopping at step %s (no improve for %s)", step, early_stopping)
                break

    train_sampler.close()

    test_metrics: dict[str, float] = {}
    if not bool(training.get("skip_test", False)) and best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log.info("loaded best ckpt missing=%s unexpected=%s", len(missing), len(unexpected))
        test_metrics = run_pathocell_mif_grid_eval(
            model,
            exp,
            transform=transform,
            split=test_split,
            device=device,
            batch_size=int(data_cfg.get("test_batch_size") or batch_size),
            use_amp=use_amp,
            exclude_background=bool(cfg.metrics.get("exclude_background", True)),
            criterion=criterion,
        )
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        log.info("test metrics: %s", test_metrics)

    summary = {
        "best_valid": best_valid,
        "primary_metric": primary,
        "metric_protocol": "pfmb_instance",
        "test": test_metrics,
        "elapsed_sec": time.time() - t0,
        "output_dir": str(output_dir),
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Online PathoCell/Lizard MifUNet training (HE → frozen generator → UNet)"
    )
    parser.add_argument("--config", required=True, help="Unified YAML (evaluation/experiment/training/head/run)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raw = load_config_dict(args.config)
    summary = train_mif_unet(raw)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
