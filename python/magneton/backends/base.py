"""What magneton needs from something that measures a run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from magneton.plugin import OpPlugin


@dataclass
class Cost:
    """What one annotated region of a run cost."""

    num_calls: int = 0
    num_kernels: int = 0
    cpu_time_ns: int = 0
    gpu_time_ns: int = 0
    gpu_energy_j: float = 0.0


@runtime_checkable
class CostBackend(Protocol):
    """Measures a run and reports per annotated region."""

    def plugins(self) -> Sequence[OpPlugin]:
        """Instrumentation to install alongside the dataflow recorder."""
        raise NotImplementedError

    def start(self) -> None:
        """Begin measuring. Called after the model is compiled and warmed up."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop measuring. The results must survive until `cost_by_annotation`."""
        raise NotImplementedError

    def cost_by_annotation(self, prefix: str) -> Mapping[str, Cost]:
        """What each annotated region cost, keyed by the scope's full name."""
        raise NotImplementedError
