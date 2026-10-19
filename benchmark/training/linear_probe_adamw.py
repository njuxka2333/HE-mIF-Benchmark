"""AdamW frozen-feature linear probe. Replaces the LBFGS ``run_linear_probe`` path."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PCA_MAX_CELLS = 50_000
PCA_PROJECT_BATCH = 100_000

LBFGS_KEYS = (
    "cost",
    "max_iter",
    "penalty",
    "tolerance_grad",
    "tolerance_change",
    "history_size",
    "solver_dtype",
    "progress_interval",
)


def _jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "rows":
            continue
        if hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def assert_no_lbfgs(training: dict[str, Any]) -> None:
    present = [key for key in LBFGS_KEYS if key in training]
    if present:
        raise ValueError(
            "LBFGS linear-probe settings are removed: "
            f"{present}. Use AdamW + max_epochs (same recipe as Immunocto encoder)."
        )


def _require_adamw(training: dict[str, Any]) -> tuple[float, float]:
    optim_cfg = dict(training.get("optimizer") or {})
    name = str(optim_cfg.get("name") or "").strip()
    if name.lower() != "adamw":
        raise ValueError(f"Linear probe requires optimizer AdamW, got {name!r}")
    params = dict(optim_cfg.get("params") or {})
    return float(params.get("lr", 1e-3)), float(params.get("weight_decay", 1e-4))


def _make_cosine(
    optimizer: torch.optim.Optimizer, training: dict[str, Any], max_epochs: int
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    sched_cfg = dict(training.get("scheduler") or {})
    name = str(sched_cfg.get("name") or "").strip()
    if name != "CosineAnnealingLR":
        raise ValueError(f"Linear probe requires CosineAnnealingLR, got {name!r}")
    params = dict(sched_cfg.get("params") or {})
    t_max = int(params.get("T_max", max_epochs))
    eta_min = float(params.get("eta_min", 0.0))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def _metric_from_eval(metrics: dict[str, Any], name: str) -> float:
    if name in metrics:
        return float(metrics[name])
    mapped = {
        "macro_f1": "lin_macro_f1",
        "auroc": "lin_auroc",
        "auprc": "lin_auprc",
        "bacc": "lin_bacc",
        "acc": "lin_acc",
        "loss": "loss",
    }
    key = mapped.get(name, f"lin_{name}")
    if key not in metrics:
        raise KeyError(f"metric {name!r} not in eval output keys {sorted(metrics)}")
    return float(metrics[key])


def _apply_zscore(
    train_x: torch.Tensor,
    others: list[torch.Tensor | None],
    *,
    combine: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor | None], torch.Tensor, torch.Tensor]:
    fit = torch.cat(combine, dim=0) if len(combine) > 1 else combine[0]
    mean = fit.mean(dim=0)
    std = fit.std(dim=0, unbiased=True).clamp_min_(1e-6)
    train_x = train_x.sub(mean).div(std)
    out = []
    for tensor in others:
        out.append(None if tensor is None else tensor.sub(mean).div(std))
    return train_x, out, mean, std


def _apply_train_pca(
    train_x: torch.Tensor,
    others: list[torch.Tensor | None],
    *,
    n_components: int,
    seed: int,
    max_cells: int = PCA_MAX_CELLS,
) -> tuple[torch.Tensor, list[torch.Tensor | None], dict[str, Any] | None]:
    """Train-only PCA after the probe's standardisation. No whitening."""
    dim = int(train_x.shape[1])
    if dim <= int(n_components):
        return train_x, others, None
    rng = np.random.RandomState(int(seed))
    n = int(train_x.shape[0])
    if n > int(max_cells):
        pick = rng.choice(n, size=int(max_cells), replace=False)
        fit = train_x[pick].detach().float().cpu().numpy()
    else:
        fit = train_x.detach().float().cpu().numpy()
    k = min(int(n_components), fit.shape[0], fit.shape[1])
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="randomized", whiten=False, random_state=int(seed))
    pca.fit(fit)
    mean = torch.tensor(pca.mean_, dtype=torch.float32)
    components = torch.tensor(pca.components_, dtype=torch.float32)

    def _project(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        chunks = []
        for start in range(0, tensor.shape[0], PCA_PROJECT_BATCH):
            sl = tensor[start : start + PCA_PROJECT_BATCH].detach().float()
            z = (sl - mean) @ components.transpose(0, 1)
            chunks.append(z.to(dtype=tensor.dtype))
        return torch.cat(chunks, dim=0)

    payload = {
        "n_components": int(k),
        "input_dim": int(dim),
        "n_fit_cells": int(fit.shape[0]),
        "max_cells": int(max_cells),
        "whiten": False,
        "seed": int(seed),
        "mean": pca.mean_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return _project(train_x), [_project(t) for t in others], payload


def run_linear_probe(cfg) -> None:
    """Drop-in replacement for bytecode ``run_linear_probe`` (AdamW only)."""
    import training.offline_train as ot

    training = dict(cfg.training or {})
    data = dict(cfg.data or {})
    assert_no_lbfgs(training)
    if "max_epochs" not in training:
        raise ValueError("Linear probe requires training.max_epochs; LBFGS max_iter is removed.")
    lr, weight_decay = _require_adamw(training)

    seed = int(training.get("seed", 42))
    ot.set_seed(seed)
    device_name = training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    batch_size = int(training.get("batch_size", 4096))
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    max_epochs = int(training["max_epochs"])
    patience = int(training.get("early_stopping_patience", 0))
    min_delta = float(training.get("early_stopping_min_delta", 0.0))
    start_epoch = int(training.get("early_stopping_start_epoch", 0))
    evaluate_test = bool(training.get("evaluate_test", True))
    verbose = bool(training.get("verbose", False))
    combine_trainval = bool(training.get("combine_trainval", False))
    primary = str(training.get("primary_metric") or "macro_f1")
    primary_mode = str(training.get("primary_metric_mode") or "max")
    select_metric = str(training.get("checkpoint_select_metric") or f"valid_{primary}")
    if select_metric.startswith("test_"):
        select_metric = f"valid_{select_metric[len('test_'):]}"

    output_dirs = ot._prepare_output_dirs(cfg)
    train_log = output_dirs["logs"] / "train.log"
    ot._configure_torch_threads(cfg, train_log)
    ot._log(f"experiment_dir={cfg.output_dir}", train_log)

    train_split = str(data.get("train_split") or "train")
    valid_split = data.get("valid_split")
    test_split = str(data.get("test_split") or "test")
    train_dataset = ot._make_dataset(cfg, split=train_split)
    valid_dataset = ot._make_dataset(cfg, split=str(valid_split)) if valid_split else None
    train_x, train_y, train_rows = train_dataset.tensors(include_rows=True)
    if valid_dataset is not None:
        valid_x, valid_y, valid_rows = valid_dataset.tensors(include_rows=True)
    else:
        valid_x = valid_y = valid_rows = None

    test_x = test_y = test_rows = None
    test_dataset = None
    if evaluate_test:
        test_dataset = ot._make_dataset(cfg, split=test_split)
        test_x, test_y, test_rows = test_dataset.tensors(include_rows=True)

    if combine_trainval and valid_x is not None:
        train_x = torch.cat([train_x, valid_x], dim=0)
        train_y = torch.cat([train_y, valid_y], dim=0)
        train_rows = list(train_rows) + list(valid_rows)

    feature_norm_mean = feature_norm_std = None
    norm_name = str(training.get("feature_normalize") or data.get("feature_normalize") or "none")
    if norm_name == "zscore_train":
        extras = [valid_x, test_x]
        fit_parts = [train_x]
        train_x, extras, feature_norm_mean, feature_norm_std = _apply_zscore(
            train_x, extras, combine=fit_parts
        )
        valid_x, test_x = extras
        payload = {
            "feature_normalize": "zscore_train",
            "feature_norm_mean": feature_norm_mean.tolist(),
            "feature_norm_std": feature_norm_std.tolist(),
            "fit_partition": "train",
        }
        (cfg.output_dir / "feature_norm.json").write_text(json.dumps(payload), encoding="utf-8")
    elif norm_name not in {"none", "l2"}:
        raise ValueError(f"Unsupported feature_normalize={norm_name!r}")

    pca_dim = int(training.get("pca_dim") or data.get("pca_dim") or 0)
    pca_payload = None
    if pca_dim > 0:
        pca_max_cells = int(training.get("pca_max_cells") or data.get("pca_max_cells") or PCA_MAX_CELLS)
        extras = [valid_x, test_x]
        train_x, extras, pca_payload = _apply_train_pca(
            train_x, extras, n_components=pca_dim, seed=seed, max_cells=pca_max_cells
        )
        valid_x, test_x = extras
        if pca_payload is None:
            ot._log(f"PCA skipped: feature width already <= {pca_dim}", train_log)
        else:
            (cfg.output_dir / "pca.json").write_text(json.dumps(pca_payload), encoding="utf-8")
            ot._log(
                f"PCA({pca_payload['n_components']}) fit on {pca_payload['n_fit_cells']} "
                f"train cells ({pca_payload['input_dim']} -> {pca_payload['n_components']}, "
                f"whiten=false, seed={pca_payload['seed']})",
                train_log,
            )

    num_classes = int(train_dataset.num_classes)
    model = ot.HEAD_REGISTRY[cfg.head](int(train_x.shape[1]), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _make_cosine(optimizer, training, max_epochs)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    label_dict = getattr(train_dataset, "label_dict", None)

    ot._log(
        f"Linear probe (AdamW): max_epochs={max_epochs} batch_size={batch_size} "
        f"lr={lr} weight_decay={weight_decay} early_stopping_patience={patience} "
        f"skip_train_eval=1 select={select_metric}",
        train_log,
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf if primary_mode == "max" else math.inf
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0

    def _score(row: dict[str, Any]) -> float:
        if select_metric not in row or row[select_metric] in ("", None):
            fallback = f"valid_{primary}"
            if fallback not in row:
                raise KeyError(f"selection metric {select_metric!r} missing in {sorted(row)}")
            return float(row[fallback])
        return float(row[select_metric])

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item()) * int(batch_y.shape[0])
            n_samples += int(batch_y.shape[0])
        scheduler.step()
        epoch_train_loss = running_loss / max(n_samples, 1)

        # Train-set AUROC/F1 on ~2M cells dominates wall time and is never the
        # checkpoint metric. Keep the epoch mean CE from the training loop.
        train_metrics = {
            "loss": epoch_train_loss,
            "macro_f1": float("nan"),
            "bacc": float("nan"),
        }
        valid_metrics = {}
        if valid_x is not None:
            valid_metrics = ot._eval_linear_probe(
                model,
                valid_x,
                valid_y,
                valid_rows,
                num_classes=num_classes,
                device=device,
                batch_size=eval_batch_size,
                label_dict=label_dict,
                include_rows=False,
            )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_macro_f1": _metric_from_eval(train_metrics, "macro_f1"),
            "train_balanced_accuracy": _metric_from_eval(train_metrics, "bacc"),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer": "AdamW",
        }
        if valid_metrics:
            row.update(
                {
                    "valid_loss": float(valid_metrics.get("loss", float("nan"))),
                    "valid_macro_f1": _metric_from_eval(valid_metrics, "macro_f1"),
                    "valid_balanced_accuracy": _metric_from_eval(valid_metrics, "bacc"),
                    "valid_auroc": _metric_from_eval(valid_metrics, "auroc"),
                    "valid_auprc": _metric_from_eval(valid_metrics, "auprc"),
                }
            )
        history.append(row)

        score = _score(row)
        improved = score > (best_score + min_delta) if primary_mode == "max" else score < (best_score - min_delta)
        marker = ""
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
            marker = " *best*"
        elif epoch >= start_epoch:
            epochs_without_improve += 1
        msg = (
            f"epoch={epoch} train_loss={epoch_train_loss:.4f} "
            f"train_macro_f1={row['train_macro_f1']:.4f}"
        )
        if "valid_macro_f1" in row:
            msg += f" valid_macro_f1={row['valid_macro_f1']:.4f} valid_auroc={row.get('valid_auroc', float('nan')):.4f}"
        msg += marker
        ot._log(msg, train_log)
        if verbose:
            print(msg, flush=True)
        if patience > 0 and epoch >= start_epoch and epochs_without_improve >= patience:
            ot._log(
                f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best {select_metric}={best_score})",
                train_log,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ot._write_metrics(cfg.output_dir / "history.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "protocol": "adamw_cosine",
            "primary_metric": primary,
            "best_epoch": best_epoch,
            "feature_normalize": norm_name,
            "feature_norm_mean": None if feature_norm_mean is None else feature_norm_mean.tolist(),
            "feature_norm_std": None if feature_norm_std is None else feature_norm_std.tolist(),
        },
        cfg.output_dir / "best_model.pt",
    )

    if evaluate_test and test_x is not None:
        test_metrics = ot._eval_linear_probe(
            model,
            test_x,
            test_y,
            test_rows,
            num_classes=num_classes,
            device=device,
            batch_size=eval_batch_size,
            label_dict=label_dict,
            include_rows=True,
        )
        ot._write_linear_probe_test_outputs(cfg, test_metrics)
        ot._log(f"test_metrics={_jsonable(test_metrics)}", train_log)
        (cfg.output_dir / "test_metrics.json").write_text(
            json.dumps(_jsonable(test_metrics), indent=2), encoding="utf-8"
        )

    if valid_x is not None:
        valid_metrics = ot._eval_linear_probe(
            model,
            valid_x,
            valid_y,
            valid_rows,
            num_classes=num_classes,
            device=device,
            batch_size=eval_batch_size,
            label_dict=label_dict,
            include_rows=False,
        )
        valid_dir = cfg.output_dir / "validation_result"
        valid_dir.mkdir(parents=True, exist_ok=True)
        (valid_dir / "best_validation_metrics.json").write_text(
            json.dumps(_jsonable(valid_metrics), indent=2),
            encoding="utf-8",
        )
        if history:
            idx = best_epoch if best_epoch >= 0 else len(history) - 1
            ot._write_metrics(valid_dir / "best_validation.csv", [history[idx]])
            ot._write_metrics(valid_dir / "validation_metrics.csv", history)
