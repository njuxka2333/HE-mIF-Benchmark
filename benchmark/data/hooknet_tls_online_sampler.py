"""Compatibility shims for archived online-training bytecode.

WSI ``OfficialOnlineSampler`` is gone. TLS reads pre-cropped PNG patches via
``FixedGridIndexSampler``. ``PrefetchingBatchSource`` lives in
``data.prefetching_batch_source``; this module only re-exports it because
``online_segmentation_train`` bytecode still imports that name here.
"""

from __future__ import annotations

from data.prefetching_batch_source import PrefetchingBatchSource

__all__ = ["OfficialOnlineSampler", "PrefetchingBatchSource"]


class OfficialOnlineSampler:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "WSI OfficialOnlineSampler has been removed. TLS training requires "
            "official_sampling.fixed_grid_root and FixedGridIndexSampler."
        )
