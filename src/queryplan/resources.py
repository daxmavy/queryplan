"""Describing the GPUs a run may use. Resources are always given explicitly; nothing is discovered or backed off from."""

from __future__ import annotations

import os

from .planner import Slot


def gpus(indices: list[int], *, hw: str, mem_gb: float, per_slot: int = 1, host: str = "localhost") -> list[Slot]:
    """A fixed set of GPUs on one machine, grouped into slots of `per_slot` GPUs (one model per slot at a time).

    `hw` names the GPU type (e.g. "H100-80GB"); the cost model learns speeds and load times per type.
    """
    if per_slot < 1 or len(indices) % per_slot:
        raise ValueError(f"{len(indices)} GPUs cannot be split into slots of {per_slot}")
    return [Slot(id=f"gpu{'+'.join(map(str, indices[i:i + per_slot]))}", gpus=tuple(indices[i:i + per_slot]), hw=hw,
                 mem_gb=mem_gb, host=host) for i in range(0, len(indices), per_slot)]


def visible_gpus(*, hw: str, mem_gb: float, per_slot: int = 1) -> list[Slot]:
    """The GPUs this process may use, from CUDA_VISIBLE_DEVICES (set by Slurm for a job, or by any GPU launcher)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS")
    if not visible:
        raise RuntimeError("no GPUs visible: set CUDA_VISIBLE_DEVICES, or run inside a Slurm job that requested GPUs")
    return gpus([int(x) for x in visible.split(",") if x.strip()], hw=hw, mem_gb=mem_gb, per_slot=per_slot,
                host=os.environ.get("SLURMD_NODENAME", "localhost"))
