"""Background batch prefetch. Sampler-agnostic; not WSI-specific."""

from __future__ import annotations

import queue
import threading
from typing import Any


class PrefetchingBatchSource:
    """Build the next training batch on a background thread while the GPU trains."""

    def __init__(self, sampler: Any, *, prefetch_batches: int = 2) -> None:
        self.sampler = sampler
        self.prefetch_batches = max(1, int(prefetch_batches))
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="hooknet_prefetch",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.sampler.next_batch()
            except BaseException as exc:
                self._queue.put(exc)
                return
            self._queue.put(batch)

    def next_batch(self) -> Any:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def reset_epoch(self, epoch: int | None = None) -> None:
        reset = getattr(self.sampler, "reset_epoch", None)
        if callable(reset):
            reset(epoch)
        self._drain()

    def begin_full_pass(self) -> None:
        begin = getattr(self.sampler, "begin_full_pass", None)
        if callable(begin):
            begin()
        self._drain()

    def close(self) -> None:
        self._stop.set()
        self._drain()
        self._thread.join(timeout=5.0)

    def _drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
