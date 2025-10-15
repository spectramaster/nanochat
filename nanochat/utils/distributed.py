"""Distributed training helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist

from .logging import LOGGER


def is_ddp() -> bool:
    """Return whether the current process participates in DDP."""

    return int(os.environ.get("RANK", -1)) != -1


def get_dist_info() -> Tuple[bool, int, int, int]:
    """Return tuple describing distributed execution state."""

    if is_ddp():
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        if not all(var in os.environ for var in required):
            missing = ", ".join(var for var in required if var not in os.environ)
            raise EnvironmentError(f"Missing DDP environment variables: {missing}")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    return False, 0, 0, 1


def compute_init() -> Tuple[bool, int, int, int, torch.device]:
    """Initialize CUDA, RNG seeds and optionally the distributed process group."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is needed for a distributed run atm")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda")

    if ddp_rank == 0:
        LOGGER.info("Distributed world size: %s", ddp_world_size)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup() -> None:
    """Tear down the distributed process group if required."""

    if is_ddp():
        dist.destroy_process_group()


@dataclass(slots=True)
class DummyWandb:
    """Fallback implementation for the Weights & Biases logger."""

    def log(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """No-op logging method."""

    def finish(self) -> None:
        """No-op cleanup method."""
