"""The energy backend: eprof, behind magneton's cost interface.

This is the adapter that lets a comparison be about joules rather than just
milliseconds. It drives a profiler capture around the measured region and then
attributes what it collected to the scopes the graph transform annotated.

It compiles nothing, and neither does the profiler it drives. Whoever is
recording the dataflow has already installed the instrumentation and called
`torch.compile`; a second party doing its own would either fight with that or
silently profile a different graph. eprof profiles the region, and this
attributes what it collected to the scopes the recorder annotated.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from magneton.backends.base import Cost
from magneton.eprof import attribution
from magneton.eprof.profiler import ActivityType, Profiler
from magneton.eprof.replay import ReplayPlugin
from magneton.eprof.config import EnergyConfig, ReplayConfig, TracingConfig
from magneton.plugin import OpPlugin


class EnergyBackend:
    """Latency and energy per annotated region, measured by eprof.

    Args:
        devices: which GPUs to sample power from, by position in
            `CUDA_VISIBLE_DEVICES` rather than by NVML index
        tracing_config: what the capture should record
        replay_config: whether to re-run each operation for power samples
        device_index: restrict attribution to one GPU; None uses all found
    """

    def __init__(
        self,
        devices: Sequence[int] = (0,),
        tracing_config: Optional[TracingConfig] = None,
        replay_config: Optional[ReplayConfig] = None,
        device_index: Optional[int] = None,
        framework: str = "torch",
    ) -> None:
        if framework not in ("torch", "jax"):
            raise ValueError(f"framework must be 'torch' or 'jax', not {framework!r}")
        self._framework = framework
        self._devices = list(devices)
        self._device_index = device_index
        self._replay_config = replay_config
        self._tracing_config = tracing_config or TracingConfig(record_shapes=True)
        self._energy_config = EnergyConfig(
            profile_energy=True, energy_profile_device=list(devices)
        )
        self._profiler = None

    @property
    def framework(self) -> str:
        """Which framework this measures, so a recorder can check it matches.

        A backend measuring the wrong one does not fail; it reports every
        operation costing nothing.
        """
        return self._framework

    # --- the backend interface -----------------------------------------------

    def plugins(self) -> Sequence[OpPlugin]:
        """Replay, when it is configured. torch only -- there is no dispatcher
        to interpose on in JAX, and the recorder runs one primitive at a time
        regardless.

        Energy comes from integrating NVML power samples over the time an
        operation was running. NVML updates on the order of milliseconds, so an
        operation that finishes faster than that can fall between two samples
        and be credited nothing. Replaying it until the window is long enough
        is what makes the number exist.
        """
        if self._replay_config is None or not self._replay_config.replay:
            return []
        return [
            ReplayPlugin(
                replay_rounds=self._replay_config.replay_rounds,
                replay_cuda_graph=self._replay_config.replay_cuda_graph,
                max_num_replay_ops=self._replay_config.max_num_replay_ops,
                replay_once_per_op=self._replay_config.replay_once_per_op,
            )
        ]

    def start(self) -> None:
        # The profiler is built here rather than in __init__, and the timing
        # is load-bearing. Constructing one arms libkineto, and everything the
        # recorder does before the region opens -- compiling the model, and
        # running it once so the dataflow statistics are not measured -- would
        # then happen against an armed profiler. Starting the real capture
        # afterwards came back with no annotations at all for that first run,
        # so it reported a model that cost nothing.
        if self._framework == "jax":
            # JAX is watched through its own profiler's trace rather than
            # libkineto; the power sampler underneath is the same either way.
            self._profiler = Profiler(backend="jax", devices=self._devices)
        else:
            self._profiler = Profiler(
                activities=[ActivityType.CPU, ActivityType.CUDA],
                tracing_config=self._tracing_config,
                energy_config=self._energy_config,
            )
        self._profiler.__enter__()

    def stop(self) -> None:
        self._profiler.__exit__(None, None, None)

    def cost_by_annotation(self, prefix: str) -> Mapping[str, Cost]:
        """Read the capture out.

        Do this before another capture starts. libkineto is process-wide, and
        preparing a second trace reuses the buffers this result still points
        at, so the numbers quietly become the next run's.
        """
        records = attribution.attribute_by_annotation(
            self._profiler.trace, self._device_index, prefix
        )
        return {
            name: Cost(
                num_calls=r.num_calls,
                num_kernels=r.num_kernels,
                cpu_time_ns=r.cpu_time_ns,
                gpu_time_ns=r.gpu_time_ns,
                gpu_energy_j=r.gpu_energy_j,
            )
            for name, r in records.items()
        }

    # --- what eprof offers beyond the interface -------------------------------

    def export_chrome_trace(self, path: str) -> None:
        """The whole capture, for chrome://tracing or Perfetto."""
        self._profiler.export_chrome_trace(path)

    def per_op_table(self, device_index: Optional[int] = None):
        """Cost per aten operator, which needs no annotations at all.

        A different question from `cost_by_annotation`: that one answers what
        each part of the model cost, this one what each *kind of operation*
        cost across the whole run.
        """
        return self._profiler.per_op_table(device_index)

    @property
    def profiler(self):
        """The profiler itself, for anything not wrapped here."""
        return self._profiler

    @property
    def trace(self):
        """The raw capture."""
        return self._profiler.trace
