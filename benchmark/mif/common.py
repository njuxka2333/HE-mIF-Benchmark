"""Shared helpers for virtual-stain UNet training (cellseg MifUNet + TLS MifFeatureUNETR)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import yaml

from training.augmentations import build_augmenter
from training.config import EvalConfig, load_eval_config


def load_config_dict(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return raw


def eval_config_from_raw(raw: Mapping[str, Any], *, fallback_id: str = "mif_seg") -> EvalConfig:
    """Build ``EvalConfig`` from an in-memory YAML dict (no extra disk round-trip)."""
    evaluation = dict(raw.get("evaluation") or {})
    run = dict(raw.get("run") or {})
    eval_id = str(evaluation.get("id") or fallback_id)
    output_dir = Path(run.get("output_dir") or f"/tmp/{eval_id}")
    return EvalConfig(
        eval_id,
        str(evaluation.get("task") or "cell_semantic_segmentation"),
        str(evaluation.get("dataset") or "pathocell_online"),
        str(evaluation.get("head") or "unetr"),
        output_dir,
        dict(raw.get("data") or {}),
        dict(raw.get("head") or {}),
        dict(raw.get("training") or {}),
        dict(raw.get("metrics") or {}),
        dict(raw.get("survival") or {}),
    )


def strip_hed_augmentations(training_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop HEDNormalize — marker stacks are not H&E."""
    cfg = dict(training_cfg or {})
    augs = dict(cfg.get("augmentations") or {})
    augs.pop("HEDNormalize", None)
    cfg["augmentations"] = augs
    return cfg


def build_mif_augmenter(training_cfg: Mapping[str, Any] | None):
    return build_augmenter(strip_hed_augmentations(training_cfg))


def make_optimizer(training_cfg: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = dict(training_cfg.get("optimizer") or {})
    name = str(opt_cfg.get("name") or "AdamW")
    params = dict(opt_cfg.get("params") or {"lr": 3e-4, "weight_decay": 0.01})
    cls = getattr(torch.optim, name)
    return cls(model.parameters(), **params)


def make_scheduler(
    training_cfg: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_cfg = dict(training_cfg.get("scheduler") or {})
    if not sched_cfg:
        return None
    name = str(sched_cfg.get("name") or "CosineAnnealingLR")
    params = dict(sched_cfg.get("params") or {})
    cls = getattr(torch.optim.lr_scheduler, name)
    return cls(optimizer, **params)


def make_ce_criterion(
    training_cfg: Mapping[str, Any],
    num_classes: int,
) -> nn.Module:
    """CrossEntropy (+ optional class weighting) without depending on bytecode loss factory."""
    from losses import ClassExcludedMeanLoss

    loss_cfg = dict(training_cfg.get("loss_fn") or {})
    params = dict(loss_cfg.get("params") or {})
    label_smoothing = float(params.get("label_smoothing", 0.0))
    ignore_index = training_cfg.get("ignore_index")
    ce_kwargs: dict[str, Any] = {"reduction": "none", "label_smoothing": label_smoothing}
    if ignore_index is not None:
        ce_kwargs["ignore_index"] = int(ignore_index)
    base = nn.CrossEntropyLoss(**ce_kwargs)
    exclude = list(loss_cfg.get("exclude_classes") or [])
    class_weighting = bool(loss_cfg.get("class_weighting", False))
    return ClassExcludedMeanLoss(
        base,
        num_classes=int(num_classes),
        exclude_classes=exclude,
        class_weighting=class_weighting,
    )


@torch.no_grad()
def estimate_channel_stats(
    batches: list[torch.Tensor],
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over a list of ``[B,C,H,W]`` stacks."""
    if not batches:
        raise ValueError("need at least one batch to estimate channel stats")
    c = int(batches[0].shape[1])
    total = 0
    sum_c = torch.zeros(c, dtype=torch.float64)
    sumsq_c = torch.zeros(c, dtype=torch.float64)
    for batch in batches:
        if batch.dim() != 4 or batch.shape[1] != c:
            raise ValueError(f"expected [B,{c},H,W], got {tuple(batch.shape)}")
        x = batch.detach().float().cpu()
        flat = x.permute(1, 0, 2, 3).reshape(c, -1)
        sum_c += flat.sum(dim=1).double()
        sumsq_c += (flat * flat).sum(dim=1).double()
        total += flat.shape[1]
    mean = (sum_c / max(total, 1)).float()
    var = (sumsq_c / max(total, 1) - mean.double().pow(2)).clamp_min(0.0)
    std = var.sqrt().float().clamp_min(eps)
    return mean, std


def save_channel_stats(path: Path, mean: torch.Tensor, std: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_channel_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    return mean, std


def warmup_factor(
    step: int,
    warmup_steps: int,
    base_lrs: list[float],
    optimizer: torch.optim.Optimizer,
) -> None:
    if warmup_steps <= 0:
        return
    if step >= warmup_steps:
        return
    scale = float(step + 1) / float(warmup_steps)
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def is_finite_loss(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss).all().item()) and not math.isnan(float(loss.detach().cpu()))


__all__ = [
    "load_config_dict",
    "eval_config_from_raw",
    "load_eval_config",
    "strip_hed_augmentations",
    "build_mif_augmenter",
    "make_optimizer",
    "make_scheduler",
    "make_ce_criterion",
    "estimate_channel_stats",
    "save_channel_stats",
    "load_channel_stats",
    "warmup_factor",
    "is_finite_loss",
]
