"""Online TLS / PathoCell / Lizard training.

The loop is archived bytecode (``_orig_pyc/online_segmentation_train.pyc``).
Patches below force PNG full-grid eval and allow decoder_mode='cellseg'.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

from training._bytecode_loader import install_bytecode_shim

install_bytecode_shim(
    "online_segmentation_train",
    Path(__file__).resolve().parent,
    globals(),
    min_bytecode_size=2500,
)
_mod = sys.modules["_bytecode_online_segmentation_train"]

from data.prefetching_batch_source import PrefetchingBatchSource  # noqa: E402

_mod.PrefetchingBatchSource = PrefetchingBatchSource


def wrap_prefetch(source, prefetch_batches: int):
    if isinstance(source, PrefetchingBatchSource):
        return source
    return PrefetchingBatchSource(source, prefetch_batches=prefetch_batches)


_mod.wrap_prefetch = wrap_prefetch

# TLS train/eval reads the canonical 1024 PNG grid as PFMB tiles (no center crop).
# WSI OfficialOnlineSampler and WSI sliding-window test bytecode are not used.


def _tls_png_full_grid_eval(
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
    sampler,
    write_outputs: bool = True,
):
    """Score every PFMB tile of every canonical PNG (encoder-route full_grid)."""
    del transform  # PNG sampler already applied the encoder transform.
    begin = getattr(sampler, "begin_full_pass", None)
    if callable(begin):
        begin()
    n_patches = len(sampler)
    batch_size = max(1, int(getattr(sampler, "batch_size", 1)))
    steps = max(1, (n_patches + batch_size - 1) // batch_size)
    batches = materialize_online_valid_batches(
        sampler, steps=steps, log=log, label=f"tls_png_{split}"
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
        split=split,
        log=log,
        write_outputs=write_outputs,
    )


def _build_tls_sources(exp, cfg, transform, device, train_split, valid_split):
    """Train TLS from ``official_sampling.fixed_grid_root`` PNG patches only."""
    from data.hooknet_tls_sampler import FixedGridIndexSampler

    official = exp.official_sampling
    if official is None:
        raise ValueError("TLS training requires run.official_sampling in the experiment config")
    if official.fixed_grid_root is None:
        raise ValueError(
            "TLS training requires official_sampling.fixed_grid_root "
            "(pre-cropped PNG grid). WSI-while-train sampling is removed."
        )

    training = dict(getattr(cfg, "training", None) or {})
    uniform = bool(training.get("uniform_class_sampling", False))
    valid_workers = max(1, min(4, int(exp.iterator_cpus)))

    def _make(samples, split, seed_offset, workers, *, class_balanced: bool):
        return FixedGridIndexSampler(
            exp,
            samples,
            official,
            transform,
            split=split,
            device=device,
            backend=exp.backend,
            seed_offset=seed_offset,
            patch_workers=workers,
            uniform_class_sampling=class_balanced,
        )

    train_samples = load_split_samples(exp.dataset_config, split_filter=train_split)
    valid_samples = load_split_samples(exp.dataset_config, split_filter=valid_split)
    train_sampler = _make(train_samples, train_split, 0, exp.iterator_cpus, class_balanced=uniform)
    valid_sampler = _make(valid_samples, valid_split, 10000, valid_workers, class_balanced=False)

    test_split = str((getattr(cfg, "data", None) or {}).get("test_split") or "test")
    test_samples = load_split_samples(exp.dataset_config, split_filter=test_split)
    test_sampler = _make(
        test_samples,
        test_split,
        online_test_seed_offset(cfg),
        valid_workers,
        class_balanced=False,
    )

    def _valid_full_grid(**kwargs):
        return _tls_png_full_grid_eval(
            kwargs["decoder"],
            kwargs["encoder"],
            kwargs.get("transform"),
            kwargs["exp"],
            kwargs["cfg"],
            kwargs["criterion"],
            kwargs["device"],
            valid_split,
            kwargs["use_amp"],
            kwargs["log"],
            sampler=valid_sampler,
            write_outputs=False,
        )

    def _test_full_grid(
        decoder,
        encoder,
        transform,
        exp,
        cfg,
        criterion,
        device,
        test_split,
        use_amp,
        log,
    ):
        return _tls_png_full_grid_eval(
            decoder,
            encoder,
            transform,
            exp,
            cfg,
            criterion,
            device,
            test_split,
            use_amp,
            log,
            sampler=test_sampler,
            write_outputs=True,
        )

    valid_runner = _maybe_valid_runner(cfg, _valid_full_grid)
    return (
        wrap_prefetch(train_sampler, official.prefetch_batches),
        valid_sampler,
        valid_runner,
        _test_full_grid,
        test_sampler,
    )


_mod._build_tls_sources = _build_tls_sources

_orig_assert_pathocell_segmentation_head = _mod._assert_pathocell_segmentation_head


def _decoder_mode_from_cfg(cfg) -> str:
    head_config = getattr(cfg, "head_config", None)
    if head_config is None and isinstance(cfg, dict):
        head_config = cfg.get("head") or {}
    if not isinstance(head_config, dict):
        return ""
    return str(head_config.get("decoder_mode") or "").strip().lower()


def _assert_pathocell_segmentation_head(cfg):
    """Reported PathoCell/Lizard winners include decoder_mode='cellseg' as well as pfmb."""
    if _decoder_mode_from_cfg(cfg) == "cellseg":
        return None
    return _orig_assert_pathocell_segmentation_head(cfg)


def _assert_pathocell_benchmark_protocol(cfg):
    data = getattr(cfg, "data", None) or {}
    tiling = data.get("pathocell_tiling") if isinstance(data, dict) else None
    if tiling is not None and str(tiling).strip().lower() != "pfmb":
        raise ValueError("PathoCell benchmark protocol requires pathocell_tiling='pfmb'")
    return _assert_pathocell_segmentation_head(cfg)


_mod._assert_pathocell_segmentation_head = _assert_pathocell_segmentation_head
_mod._assert_pathocell_benchmark_protocol = _assert_pathocell_benchmark_protocol


def _online_valid_protocol(cfg):
    """Checkpoint selection runs full-grid inference on valid; test is last."""
    return "full_grid"


_orig_online_sampled_protocol = _mod.online_sampled_protocol


def online_sampled_protocol(cfg, scope="valid"):
    if str(scope).lower() == "valid":
        return "full_grid"
    return _orig_online_sampled_protocol(cfg, scope=scope)


_mod._online_valid_protocol = _online_valid_protocol
_mod.online_sampled_protocol = online_sampled_protocol
globals()["_online_valid_protocol"] = _online_valid_protocol
globals()["online_sampled_protocol"] = online_sampled_protocol
