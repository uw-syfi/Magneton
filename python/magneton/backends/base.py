"""What magneton needs from something that measures a run.

Recording a dataflow graph and matching two of them needs no measurement at
all: the graph and its tensors are enough to say that two implementations
compute the same thing, and where they stop agreeing. Measurement is what turns
that structural answer into a quantitative one -- these regions are equivalent,
*and this one cost eight times as much*.

So the measurement is a backend, and the interface is deliberately small. A
backend contributes whatever instrumentation it needs, brackets the measured
region, and afterwards hands back a cost per annotated scope. It never sees the
graph, the tensors, or the matcher.

Two implement it. `CudaEventTiming` needs nothing but torch and answers with
latency. `EnergyBackend` answers with latency *and energy*, by way of `eprof` --
which needs CUPTI, NVML and a build toolchain, and is a separate install that
magneton does not require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from magneton.plugin import OpPlugin


@dataclass
class Cost:
    """What one annotated region of a run cost.

    Deliberately not every backend's full story: a timing backend leaves
    `gpu_energy_j` at zero and a backend with no CPU-side view leaves
    `cpu_time_ns` there, and a comparison is expected to check rather than
    assume. `Run.has_energy` is how the report decides whether to print an
    energy column at all.
    """

    num_calls: int = 0
    num_kernels: int = 0
    cpu_time_ns: int = 0
    gpu_time_ns: int = 0
    gpu_energy_j: float = 0.0


@runtime_checkable
class CostBackend(Protocol):
    """Measures a run and reports per annotated region.

    The lifecycle is fixed by `Recorder`: `plugins()` is read once while the
    graph is being instrumented, `start()` and `stop()` bracket the measured
    region, and `cost_by_annotation` is called after it.
    """

    def plugins(self) -> Sequence[OpPlugin]:
        """Instrumentation to install alongside the dataflow recorder.

        A backend that measures energy from NVML needs each operation run
        enough times to catch more than one power sample, and that is a plugin.
        A backend that needs nothing returns an empty sequence.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Begin measuring. Called after the model is compiled and warmed up."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop measuring. The results must survive until `cost_by_annotation`."""
        raise NotImplementedError

    def cost_by_annotation(self, prefix: str) -> Mapping[str, Cost]:
        """What each annotated region cost, keyed by the scope's full name.

        `prefix` is what the annotator named its scopes with; anything not
        starting with it is not a region this caller asked about.
        """
        raise NotImplementedError
