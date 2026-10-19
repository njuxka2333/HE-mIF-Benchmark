"""Minimal distributed runtime helpers for online dense-prediction training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


def initialize_distributed(device_arg: str | None = None) -> tuple[DistributedContext, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = local_rank = 0
        world_size = 1

    if device_arg is not None:
        requested = torch.device(device_arg)
        if requested.type == "cuda" and requested.index is None:
            device = torch.device("cuda", local_rank if world_size > 1 else 0)
        else:
            device = requested
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(rank, local_rank, world_size, initialized_here), device


def destroy_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_bool(value: bool, *, device: torch.device, source: int = 0) -> bool:
    if not dist.is_initialized():
        return bool(value)
    payload = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.broadcast(payload, src=source)
    return bool(payload.item())
