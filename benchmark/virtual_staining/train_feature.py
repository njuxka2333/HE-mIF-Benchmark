"""Thin TLS online MifFeatureUNETR training (HE → frozen generator → UNet).

Reuses HookNet-TLS fixed-grid PNG sampling (``FixedGridTileSampler``). No
pre-extracted FOV mIF. Config is a unified YAML dict; materialization deferred.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.mif_output import FrozenMifGenerator, load_frozen_mif_generator, resolve_mif_channels
from heads import build_segmentation_head, forward_segmentation_head
from heads.mif_feature import MifFeatureUNETR
from virtual_staining.common import (
    eval_config_from_raw,
    is_finite_loss,
    load_channel_stats,
    load_config_dict,
    make_ce_criterion,
    make_scheduler,
    save_channel_stats,
    warmup_factor,
)
from virtual_staining.training_utils import estimate_generator_channel_stats, prepare_segmentation_batch

LOGGER = logging.getLogger("mif_feature_unetr_train")

TLS_NUM_CLASSES = 4  # background, TLS, GC, Rest (255 ignored)


def _as_next_batch_sampler(obj: Any) -> Any | None:
    nxt = getattr(obj, "next_batch", None)
    return obj if callable(nxt) else None


def _split_tls_sources(sources: Any) -> tuple[Any, Any, Any | None]:
    """Unpack ``_build_tls_sources`` into (train, valid, test) batch samplers."""
    if isinstance(sources, dict):
        train = sources["train"]
        valid = sources["valid"]
        test = _as_next_batch_sampler(sources.get("test"))
        return train, valid, test
    train, valid, *rest = sources
    test = None
    for item in rest:
        cand = _as_next_batch_sampler(item)
        if cand is not None:
            test = cand
    return train, valid, test


def build_frozen_mif_generator(
    model_name: str,
    *,
    device: torch.device,
    layer: str | None = None,
    benchmark_config_path: str | Path | None = None,
) -> tuple[FrozenMifGenerator, Any]:
    """Load the HE→mIF wrapper and wrap it in ``FrozenMifGenerator``."""
    return load_frozen_mif_generator(
        model_name,
        device,
        layer=layer,
        gpu_num=0,
        use_benchmark_config=False,
        config_path=benchmark_config_path,
    )


def build_mif_feature_from_cfg(
    head_config: dict[str, Any],
    *,
    model_name: str,
    num_classes: int,
    generator: FrozenMifGenerator | None,
) -> MifFeatureUNETR:
    cfg = dict(head_config)
    cfg.setdefault("decoder_mode", "mif_feature")
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


@torch.no_grad()
def _protocol_is_full_grid(protocol: Any) -> bool:
    return str(protocol or "").strip().lower() in {"full_grid", "png_full_grid"}


def _full_grid_batches(sampler) -> int:
    inner = sampler
    wrapped = getattr(inner, "source", None)
    if wrapped is not None and callable(getattr(wrapped, "next_batch", None)):
        inner = wrapped
    begin = getattr(inner, "begin_full_pass", None)
    if callable(begin):
        begin()
    n = len(inner)
    bs = max(1, int(getattr(inner, "batch_size", 1) or 1))
    return max(1, (n + bs - 1) // bs)


def evaluate_tls_sampled(
    model: nn.Module,
    sampler,
    *,
    device: torch.device,
    num_batches: int,
    num_classes: int = TLS_NUM_CLASSES,
    use_amp: bool = True,
    ignore_index: int = 255,
    exclude_background: bool = True,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    from metrics import SegmentationMetricAccumulator

    metrics = SegmentationMetricAccumulator(
        num_classes=num_classes,
        ignore_index=ignore_index,
        exclude_background=exclude_background,
    )
    model.eval()
    losses: list[float] = []
    for _ in range(max(1, int(num_batches))):
        batch = sampler.next_batch()
        images = batch["image"].to(device, non_blocking=True)
        target = batch["semantic_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = forward_segmentation_head(model, features=None, masks=target, images=images)
            if criterion is not None:
                losses.append(float(criterion(logits, target).detach().mean().cpu()))
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            metrics.update(pred[i], target_np[i])
    out = {str(k): float(v) for k, v in metrics.compute().items()}
    if losses:
        out["loss"] = float(sum(losses) / len(losses))
    return out


def train_mif_feature_unetr(
    raw: dict[str, Any],
    *,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    log = log or LOGGER
    from training.online_segmentation_train import _build_tls_sources, load_online_eval_bundle

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(raw, handle)
        tmp_path = Path(handle.name)
    try:
        exp, cfg_bundle = load_online_eval_bundle(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    cfg = eval_config_from_raw(raw, fallback_id=str(raw.get("evaluation", {}).get("id", "mif_feature")))
    # Prefer fields from the bytecode bundle when present.
    if cfg_bundle is not None:
        cfg = cfg_bundle

    training = dict(cfg.training)
    data_cfg = dict(cfg.data)
    head_cfg = dict(cfg.head_config)
    run = dict(raw.get("run") or {})

    device = torch.device(str(training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")))
    seed = int(training.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = str(getattr(exp, "model", "") or "").lower()
    if not model_name:
        raise ValueError("experiment.model is required")
    in_channels = resolve_mif_channels(model_name, head_cfg.get("in_channels"))
    head_cfg["in_channels"] = in_channels
    head_cfg.setdefault("decoder_mode", "mif_feature")
    num_classes = int(training.get("num_classes") or TLS_NUM_CLASSES)
    ignore_index = int(training.get("ignore_index", 255))
    output_dir = Path(run.get("output_dir") or cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator, transform = build_frozen_mif_generator(
        model_name,
        device=device,
        layer=head_cfg.get("mif_layer") or data_cfg.get("mif_layer"),
        benchmark_config_path=run.get("benchmark_config_path") or getattr(exp, "benchmark_config_path", None),
    )
    model = build_mif_feature_from_cfg(
        head_cfg,
        model_name=model_name,
        num_classes=num_classes,
        generator=generator,
    )
    model.to(device)

    train_split = str(data_cfg.get("train_split") or "train")
    valid_split = str(data_cfg.get("valid_split") or "valid")
    sources = _build_tls_sources(exp, cfg, transform, device, train_split, valid_split)
    # Sources are (train, valid, valid_runner, png_full_grid_test, test_sampler).
    # Only objects with next_batch are usable by evaluate_tls_sampled.
    train_sampler, valid_sampler, test_sampler = _split_tls_sources(sources)

    stats_path = output_dir / "channel_stats.json"
    if stats_path.is_file() and not bool(training.get("recompute_channel_stats", False)):
        mean, std = load_channel_stats(stats_path)
        log.info("loaded channel stats from %s", stats_path)
    else:
        n_stat = int(training.get("channel_stats_batches", 16))
        log.info("estimating generated channel stats over %s batches", n_stat)
        mean, std = estimate_generator_channel_stats(
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

    # Optimize UNet params only (generator already frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = dict(training.get("optimizer") or {})
    name = str(opt_cfg.get("name") or "AdamW")
    params = dict(opt_cfg.get("params") or {"lr": 3e-4, "weight_decay": 0.01})
    optimizer = getattr(torch.optim, name)(trainable, **params)
    scheduler = make_scheduler(training, optimizer)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    warmup_steps = int(opt_cfg.get("warmup_steps") or 0)

    # TLS RGB path: keep HED if configured.
    from training.augmentations import build_augmenter

    augmenter = build_augmenter(training)
    steps = int(training.get("training_steps") or 1000)
    val_interval = int(training.get("val_interval") or 200)
    log_interval = int(training.get("log_interval") or 10)
    early_stopping = training.get("early_stopping")
    early_stopping = int(early_stopping) if early_stopping is not None else None
    grad_clip = training.get("grad_clip_norm")
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    primary = str(training.get("primary_metric") or "f1_macro")
    test_batches = int(
        training.get("test_batches")
        or data_cfg.get("test_batches")
        or data_cfg.get("online_test_steps")
        or 100
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid = -1.0
    best_path = output_dir / "best.pt"
    last_improve_step = 0
    history: list[dict[str, Any]] = []

    model.train()
    # Re-assert generator stays in eval after model.train()
    if model.generator is not None:
        model.generator.train(False)

    t0 = time.time()
    for step in range(steps):
        batch = train_sampler.next_batch()
        images, masks = prepare_segmentation_batch(batch, augmenter, device)
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
            n_val = _full_grid_batches(valid_sampler)
            valid_metrics = evaluate_tls_sampled(
                model,
                valid_sampler,
                device=device,
                num_batches=n_val,
                num_classes=num_classes,
                use_amp=use_amp,
                ignore_index=ignore_index,
                exclude_background=bool(cfg.metrics.get("exclude_background", True)),
                criterion=criterion,
            )
            score = float(valid_metrics.get(primary, valid_metrics.get("f1_score_macro", -1.0)))
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
                log.info("early stopping at step %s", step)
                break

    test_metrics: dict[str, float] = {}
    eval_sampler = test_sampler or valid_sampler
    if not bool(training.get("skip_test", False)) and best_path.is_file() and eval_sampler is not None:
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            log.info("loaded best ckpt missing=%s unexpected=%s", len(missing), len(unexpected))
            test_proto = data_cfg.get("test_eval_protocol")
            n_test = _full_grid_batches(eval_sampler) if _protocol_is_full_grid(test_proto) else test_batches
            test_metrics = evaluate_tls_sampled(
                model,
                eval_sampler,
                device=device,
                num_batches=n_test,
                num_classes=num_classes,
                use_amp=use_amp,
                ignore_index=ignore_index,
                exclude_background=bool(cfg.metrics.get("exclude_background", True)),
                criterion=criterion,
            )
            (output_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, indent=2), encoding="utf-8"
            )
            log.info("test metrics: %s", test_metrics)
        except Exception:
            log.exception("test eval failed; writing train_summary without test metrics")

    summary = {
        "best_valid": best_valid,
        "primary_metric": primary,
        "test": test_metrics,
        "elapsed_sec": time.time() - t0,
        "output_dir": str(output_dir),
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Online TLS MifFeatureUNETR training")
    parser.add_argument("--config", required=True, help="Unified YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raw = load_config_dict(args.config)
    summary = train_mif_feature_unetr(raw)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
