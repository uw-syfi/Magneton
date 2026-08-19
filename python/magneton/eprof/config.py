"""How a capture is configured.

What to record, whether to sample power, and whether to replay each operation
so that there is enough power to integrate.
"""

import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class EnergyConfig:
    """Configuration for energy profiling."""

    profile_energy: bool
    """Record energy data."""
    energy_profile_device: List[int]
    """The device IDs to profile energy on."""


@dataclasses.dataclass
class TracingConfig:
    """Configuration for tracing."""

    record_shapes: bool = True
    """Record tensor shapes passed to nn.Module."""
    record_stack: bool = False
    """Record Python call stack."""
    with_modules: bool = False
    """Record nn.Module calls."""
    with_memory: bool = False
    """Record memory data."""


@dataclasses.dataclass
class ReplayConfig:
    """Configuration for operator level replay."""

    replay: bool
    """Replay each operator."""
    replay_rounds: Union[int, str]
    """Number of rounds to replay each operator."""
    replay_cuda_graph: bool
    """Replay each operator with CUDA graph."""
    max_num_replay_ops: Optional[int] = None
    """Maximum number of operators to replay, or None for no limit.

    Optional because the replay plugin has always treated it that way -- it
    only applies the cap `if max_num_replay_ops is not None` -- while this
    dataclass required it, so every caller that did not pass one failed to
    construct a config at all."""
    replay_once_per_op: bool = True
    """If True, each op is replayed only on its first invocation and runs
    without replay on subsequent calls of the compiled model (useful for
    iterative workloads like training loops). If False, replay fires on
    every invocation — use this when the compiled model is only called
    once inside the profiled region."""
