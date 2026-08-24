"""The energy backend: eprof, behind magneton's cost interface."""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence

from magneton.backends.base import Cost
from magneton.eprof import attribution
from magneton.eprof.profiler import ActivityType, Profiler
from magneton.eprof.replay import ReplayPlugin
from magneton.eprof.config import EnergyConfig, ReplayConfig, TracingConfig
from magneton.plugin import OpPlugin


class EnergyBackend:
    """Latency and energy per annotated region, measured by eprof."""

    def __init__(
        self,
        devices: Sequence[int] = (0,),
        tracing_config: Optional[TracingConfig] = None,
        replay_config: Optional[ReplayConfig] = None,
        device_index: Optional[int] = None,
        framework: str = "torch",
        allow_empty: bool = False,
    ) -> None:
        if framework not in ("torch", "jax"):
            raise ValueError(f"framework must be 'torch' or 'jax', not {framework!r}")
        self._framework = framework
        self._allow_empty = allow_empty
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
        """Which framework this measures, so a recorder can check it matches."""
        return self._framework

    # --- the backend interface -----------------------------------------------

    def plugins(self) -> Sequence[OpPlugin]:
        """Replay, when it is configured."""
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
        """Read the capture out."""
        records = attribution.attribute_by_annotation(
            self._profiler.trace, self._device_index, prefix
        )
        if not self._allow_empty and not any(r.num_kernels for r in records.values()):
            warnings.warn(
                f"the capture attributed no GPU kernel to any {prefix!r} region, "
                "so every operation will report costing nothing. This usually "
                "means the extension was built against a CUPTI that libkineto "
                "cannot drive; check that torch is within the supported range. "
                "Pass allow_empty=True to silence this.",
                RuntimeWarning,
                stacklevel=2,
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
        """Cost per aten operator, which needs no annotations at all."""
        return self._profiler.per_op_table(device_index)

    @property
    def profiler(self):
        """The profiler itself, for anything not wrapped here."""
        return self._profiler

    @property
    def trace(self):
        """The raw capture."""
        return self._profiler.trace
