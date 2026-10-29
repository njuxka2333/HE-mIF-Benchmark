"""TLS no longer uses WSI sliding-window eval.

Encoder-route ``full_grid`` test scores pre-cropped PNG patches in
``online_segmentation_train`` instead. PathoCell / Lizard full-grid eval
walks every PFMB tile of every HDF in the split.
"""

from __future__ import annotations

from typing import Any

from data.loader import load_split_samples
from data.pathocell.sampler import HDFTileBatchSampler


def run_tls_sliding_window_eval(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "WSI sliding-window TLS eval is removed; use full_grid PNG scoring"
    )


def run_tls_sliding_window_test(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "WSI sliding-window TLS test is removed; use full_grid PNG scoring"
    )


def _pathocell_grid_eval(
    decoder,
    encoder,
    transform,
    exp,
    cfg,
    criterion,
    device,
    split,
    use_amp,
    log,
    *,
    write_outputs: bool = True,
    sampler: HDFTileBatchSampler | None = None,
):
    from training.online_segmentation_train import (
        materialize_online_valid_batches,
        resolve_num_classes,
        run_fixed_sampled_split_eval,
    )

    data = getattr(cfg, "data", None) or {}
    online = dict(data.get("online_sampling") or {}) if isinstance(data, dict) else {}
    batch_size = int(
        (getattr(cfg, "training", None) or {}).get("batch_size")
        or online.get("batch_size")
        or 8
    )
    owns_sampler = sampler is None
    if sampler is None:
        samples = load_split_samples(exp.dataset_config, split_filter=str(split))
        sampler = HDFTileBatchSampler(
            exp,
            samples,
            image_transform=transform,
            batch_size=max(1, batch_size),
            seed=int(online.get("seed") or 42) + (0 if write_outputs else 10000),
            max_cached_volumes=int(online.get("volume_cache_size") or 8),
            patch_workers=max(1, min(4, int(getattr(exp, "iterator_cpus", 1) or 1))),
            uniform_class_sampling=False,
        )
    try:
        sampler.begin_full_pass()
        n_patches = len(sampler)
        steps = max(1, (n_patches + sampler.batch_size - 1) // sampler.batch_size)
        batches = materialize_online_valid_batches(
            sampler, steps=steps, log=log, label=f"pathocell_grid_{split}"
        )
        return run_fixed_sampled_split_eval(
            decoder,
            encoder,
            criterion,
            cfg,
            exp,
            fixed_batches=batches,
            num_classes=resolve_num_classes(cfg, exp),
            device=device,
            use_amp=use_amp,
            split=str(split),
            log=log,
            write_outputs=write_outputs,
        )
    finally:
        if owns_sampler:
            sampler.close()


def run_pathocell_grid_eval(
    decoder,
    encoder,
    transform,
    exp,
    cfg,
    criterion,
    device,
    split=None,
    use_amp=True,
    log=None,
    write_outputs: bool = False,
    test_split=None,
    sampler: HDFTileBatchSampler | None = None,
    **_kwargs,
):
    split = split if split is not None else test_split
    if split is None:
        raise TypeError("run_pathocell_grid_eval requires split or test_split")
    return _pathocell_grid_eval(
        decoder,
        encoder,
        transform,
        exp,
        cfg,
        criterion,
        device,
        split,
        use_amp,
        log,
        write_outputs=write_outputs,
        sampler=sampler,
    )


def run_pathocell_grid_test(
    decoder,
    encoder,
    transform,
    exp,
    cfg,
    criterion,
    device,
    split=None,
    use_amp=True,
    log=None,
    write_outputs: bool = True,
    test_split=None,
    sampler: HDFTileBatchSampler | None = None,
    **_kwargs,
):
    split = split if split is not None else test_split
    if split is None:
        raise TypeError("run_pathocell_grid_test requires split or test_split")
    return _pathocell_grid_eval(
        decoder,
        encoder,
        transform,
        exp,
        cfg,
        criterion,
        device,
        split,
        use_amp,
        log,
        write_outputs=write_outputs,
        sampler=sampler,
    )
