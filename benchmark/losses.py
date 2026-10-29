"""Survival and segmentation losses used by offline and online trainers."""

from __future__ import annotations

from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def exclude_classes(
    exclude_classes: Union[int, list[int]],
    loss: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded = [exclude_classes] if isinstance(exclude_classes, int) else list(exclude_classes)
    mask = torch.ones_like(loss)
    for cls in excluded:
        hit = target == int(cls)
        loss = loss.masked_fill(hit, 0)
        mask = mask.masked_fill(hit, 0)
    return loss, mask


class EMAInverseClassFrequencyLoss(nn.Module):
    """EMA class weighting over supervised labels only.

    TLS masks use 255 for unannotated pixels and exclude class 0 (background)
    from the loss.  Neither may contribute to the class-frequency estimate.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        exclude_class: Union[int, list[int]],
        num_classes: int,
        alpha: float = 0.99,
        class_weighting: bool = False,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.exclude_class = (
            [exclude_class] if isinstance(exclude_class, int) else list(exclude_class)
        )
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.class_weighting = bool(class_weighting)
        self.distributed_sync = True
        self.register_buffer("ema_frequencies", torch.ones(self.num_classes) * 1e-06)
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def update_frequencies(self, target: torch.Tensor) -> None:
        with torch.no_grad():
            labels = target.detach().flatten()
            valid = (labels >= 0) & (labels < self.num_classes)
            for cls in self.exclude_class:
                valid &= labels != int(cls)
            counts = torch.bincount(
                labels[valid], minlength=self.num_classes
            ).float().to(self.ema_frequencies.device)
            if self.distributed_sync and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            self.ema_frequencies.mul_(self.alpha).add_(counts, alpha=1 - self.alpha)
            self.ema_updates.add_(1)

    def calculate_weights(self) -> torch.Tensor:
        inverse = 1.0 / (self.ema_frequencies + 1e-06)
        return inverse / inverse.sum()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.class_weighting:
            self.update_frequencies(target)
            weights = self.calculate_weights().to(pred.device)
            for cls in self.exclude_class:
                weights[int(cls)] = 0
            self.loss_fn.weight = weights.softmax(dim=0)

        loss = self.loss_fn(pred, target)
        loss, mask = exclude_classes(self.exclude_class, loss, target)
        numerator = loss.sum()
        denominator = mask.sum(dtype=torch.float32).to(device=pred.device)
        if self.distributed_sync and dist.is_initialized():
            global_denominator = denominator.detach().clone()
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
            return numerator * dist.get_world_size() / global_denominator.clamp_min(1)
        return numerator / denominator.clamp_min(1)


class NLLSurvLoss(nn.Module):
    """Discrete-time negative log-likelihood survival loss (CLAM / NLL-surv)."""

    def __init__(self, alpha: float = 0.0, eps: float = 1e-07):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        hazards: torch.Tensor,
        survival: torch.Tensor,
        label: torch.Tensor,
        censorship: torch.Tensor,
    ) -> torch.Tensor:
        label = label.view(-1, 1)
        censorship = censorship.view(-1, 1).float()
        survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
        uncensored = -(1 - censorship) * (
            torch.log(torch.gather(survival_padded, 1, label).clamp(min=self.eps))
            + torch.log(torch.gather(hazards, 1, label).clamp(min=self.eps))
        )
        censored = -censorship * torch.log(
            torch.gather(survival_padded, 1, label + 1).clamp(min=self.eps)
        )
        return ((1 - self.alpha) * (censored + uncensored) + self.alpha * uncensored).mean()


class FocalLoss(nn.Module):
    """Pixel-wise focal CE with optional label smoothing; reduction usually 'none'."""

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = str(reduction)
        self.ignore_index = int(ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        num_classes = pred.shape[1]
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
                valid = target != self.ignore_index
                safe_target = target.clone()
                safe_target[~valid] = 0
                true_dist.scatter_(1, safe_target.unsqueeze(1), 1.0 - self.label_smoothing)
                true_dist = true_dist * valid.unsqueeze(1).to(true_dist.dtype)
            ce = -(true_dist * log_probs).sum(dim=1)
        else:
            ce = F.nll_loss(
                log_probs,
                target,
                reduction="none",
                ignore_index=self.ignore_index,
            )
        probs = log_probs.exp()
        safe_target = target.clamp(min=0)
        pt = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - pt).clamp(min=0.0) ** self.gamma) * ce
        if target.dtype != torch.bool:
            loss = loss.masked_fill(target == self.ignore_index, 0.0)
        if self.reduction == "mean":
            denom = (target != self.ignore_index).sum().clamp_min(1)
            return loss.sum() / denom
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassExcludedMeanLoss(nn.Module):
    """Mean a per-pixel loss while excluding one or more target classes.

    When ``class_weighting`` is set this is the EMA-weighted TLS/cellseg loss.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        exclude_class: Union[int, list[int]] = 0,
        *,
        exclude_classes: Union[int, list[int], None] = None,
        num_classes: int | None = None,
        class_weighting: bool = False,
        alpha: float = 0.99,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        excluded = exclude_class if exclude_classes is None else exclude_classes
        self.loss_fn = loss_fn
        if class_weighting:
            if num_classes is None:
                raise ValueError("class_weighting requires num_classes")
            self._weighted = EMAInverseClassFrequencyLoss(
                loss_fn,
                exclude_class=excluded,
                num_classes=int(num_classes),
                alpha=alpha,
                class_weighting=True,
            )
            self.exclude_class = self._weighted.exclude_class
            return
        self._weighted = None
        if isinstance(excluded, int):
            self.exclude_class = [int(excluded)]
        else:
            self.exclude_class = [int(x) for x in excluded]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._weighted is not None:
            return self._weighted(pred, target)
        per_pixel = self.loss_fn(pred, target)
        mask = torch.ones_like(target, dtype=torch.bool)
        for cls in self.exclude_class:
            mask &= target != cls
        denom = mask.sum().clamp_min(1)
        return (per_pixel * mask.to(per_pixel.dtype)).sum() / denom
