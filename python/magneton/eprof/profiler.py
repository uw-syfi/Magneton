"""Profiling a region, whichever framework runs in it."""

from __future__ import annotations

import contextlib
import glob
import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from magneton_eprof import _ActivityType, _Profiler

from magneton.eprof import attribution
from magneton.eprof.attribution import PerOpRecord
from magneton.eprof.config import EnergyConfig, TracingConfig

ActivityType = _ActivityType

TORCH = "torch"
JAX = "jax"

_TORCH_HAS_CAPTURED = False


class Profiler:
    """A profiling run."""

    def __init__(
        self,
        activities: Optional[List[ActivityType]] = None,
        tracing_config: Optional[TracingConfig] = None,
        energy_config: Optional[EnergyConfig] = None,
        backend: str = TORCH,
        devices: Optional[Sequence[int]] = None,
        trace_dir: Optional[str] = None,
    ):
        if backend not in (TORCH, JAX):
            raise ValueError(f"backend must be {TORCH!r} or {JAX!r}, not {backend!r}")
        self.backend = backend

        if activities is None:
            activities = [ActivityType.CPU, ActivityType.CUDA]

        if backend == TORCH:
            self._impl: _Backend = _TorchBackend(
                activities, tracing_config, energy_config
            )
        else:
            if devices is None:
                devices = (
                    energy_config.energy_profile_device
                    if energy_config is not None else [0]
                )
            self._impl = _JaxBackend(devices, trace_dir)

    # --- the measured region -------------------------------------------------

    def __enter__(self) -> Tuple["Profiler", Optional[Callable]]:
        return self, self._impl.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._impl.stop()

    # --- what it collected ---------------------------------------------------

    def per_op_table(
        self, device_index: Optional[int] = None, by_kernel: bool = False
    ) -> List[PerOpRecord]:
        """Per-operation GPU time and energy, largest energy first."""
        if by_kernel:
            if self.backend != JAX:
                raise ValueError(
                    "by_kernel is a jax notion: the torch backend attributes "
                    "kernels to the aten operator that launched them, and has "
                    "no separate kernel-level grouping"
                )
            return self._impl.per_op_table(device_index, by_kernel=True)
        return self._impl.per_op_table(device_index)

    def per_hlo_op(self, device_index: Optional[int] = None) -> List["HloOpCost"]:
        """Each HLO operation, the kernels that ran it, and what they cost."""
        if self.backend != JAX:
            raise ValueError(
                "per_hlo_op is a jax notion; the torch backend attributes "
                "kernels to the aten operator that launched them, which "
                "per_op_table reports"
            )
        return self._impl.per_hlo_op(device_index)

    def kernel_costs(self, device_index: Optional[int] = None) -> List["LaunchCost"]:
        """Every kernel launch, with when it ran and what it cost."""
        return self._impl.kernel_costs(device_index)

    def export_per_op(self, path: str, device_index: Optional[int] = None) -> None:
        attribution.save_json(self.per_op_table(device_index), path)
        logging.info(f"Per-op latency/energy exported to {path}")

    def export_chrome_trace(self, path: str) -> None:
        self._impl.export_chrome_trace(path)

    @property
    def trace(self):
        """The backend's own result object, for anything this does not wrap."""
        return self._impl.trace


# --- backends ----------------------------------------------------------------


class _Backend:
    """What a backend has to provide. Everything else is shared."""

    trace: Any = None

    def start(self) -> Optional[Callable]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def per_op_table(self, device_index: Optional[int]) -> List[PerOpRecord]:
        raise NotImplementedError

    def export_chrome_trace(self, path: str) -> None:
        raise NotImplementedError


class _TorchBackend(_Backend):
    """torch, through libkineto and the RecordFunction callbacks."""

    def __init__(self, activities, tracing_config, energy_config):
        self.activities = set(activities)
        self.profiler = _Profiler(
            self.activities,
            tracing_config.record_shapes if tracing_config is not None else False,
            False,
            tracing_config.with_memory if tracing_config is not None else False,
            tracing_config.record_stack if tracing_config is not None else False,
            tracing_config.with_modules if tracing_config is not None else False,
            energy_config.profile_energy if energy_config is not None else False,
            energy_config.energy_profile_device if energy_config is not None else [],
        )

    def start(self) -> Optional[Callable]:
        global _TORCH_HAS_CAPTURED
        _TORCH_HAS_CAPTURED = True
        self.profiler.start(set())
        return None

    def stop(self) -> None:
        self.trace = self.profiler.stop()

    def per_op_table(self, device_index: Optional[int]) -> List[PerOpRecord]:
        return attribution.attribute(self.trace, device_index=device_index)

    def kernel_costs(self, device_index: Optional[int] = None) -> List["LaunchCost"]:
        """Every kernel launch, with when it ran and what it cost."""
        import magneton_eprof

        raw = self.trace.export_raw_nodes()
        launches = [
            d for d in raw
            if d["device_type"] == attribution._DEVICE_CUDA
            and d["dur_ns"] > 0
            and d["activity_type"] in attribution._GPU_WORK
        ]
        if not launches:
            return []
        labels = [str(i) for i in range(len(launches))]
        kernels = [
            (d["start_ns"], d["start_ns"] + d["dur_ns"], d["device_index"], i)
            for i, d in enumerate(launches)
        ]
        _, _, power = attribution._extract_from_raw(raw)
        rows = magneton_eprof.attribute(
            [(label, 0) for label in labels], kernels, power,
            dict(enumerate(labels)), device_index,
        )
        by_launch = {
            name: (gpu_ns, energy)
            for name, _c, _k, _cpu, gpu_ns, energy in rows
        }
        out = []
        for i, d in enumerate(launches):
            gpu_ns, energy = by_launch.get(str(i), (0, 0.0))
            out.append(LaunchCost(
                name=d["name"], start_ns=d["start_ns"],
                end_ns=d["start_ns"] + d["dur_ns"],
                gpu_time_ns=gpu_ns, gpu_energy_j=energy,
            ))
        return out

    def export_chrome_trace(self, path: str) -> None:
        self.trace.save(path)


class _JaxBackend(_Backend):
    """JAX, through its own profiler's trace and a standalone power sampler."""

    ANCHOR = "eprof_clock_anchor"

    def __init__(self, devices: Sequence[int], trace_dir: Optional[str]):
        self.devices = list(devices)
        self.keep_trace = trace_dir is not None
        self.directory = trace_dir or tempfile.mkdtemp(prefix="eprof-jax-")
        self.kernels: List[JaxKernel] = []
        self.power: List[Tuple[int, int, int]] = []
        self.skew_ns = 0
        self._stack: Optional[contextlib.ExitStack] = None
        self._anchor_wall_ns = 0

    def start(self) -> None:
        if _TORCH_HAS_CAPTURED:
            raise RuntimeError(
                "a torch capture has already run in this process, and JAX "
                "cannot be profiled after one.\n"
                "  CUPTI takes a single subscriber. libkineto claims it for "
                "the torch capture and does not\n"
                "  give it back, so JAX's profiler still runs and still writes "
                "a trace -- with no kernels in\n"
                "  it. The result would be a run that appears to have done no "
                "GPU work.\n"
                "\n"
                "  Profile every JAX region before the first torch one, or put "
                "the two in separate\n"
                "  processes and compare them with compare.Run.save and "
                "compare.Run.load."
            )
        import jax

        import magneton_eprof

        self._sampler = magneton_eprof._EnergySampler(self.devices)
        self._sampler.start()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(jax.profiler.trace(self.directory))
        with jax.profiler.TraceAnnotation(self.ANCHOR):
            self._anchor_wall_ns = time.time_ns()
        return None

    def stop(self) -> None:
        try:
            if self._stack is not None:
                self._stack.close()
                self._stack = None
        finally:
            self.power = self._sampler.stop()

        path = _find_trace(self.directory)
        if path is None:
            logging.warning("JAX wrote no trace; was the work on GPU?")
            return
        self.trace = path
        self.kernels, self.skew_ns = _kernels_from_trace(
            _load_trace(path), self._anchor_wall_ns, self.ANCHOR
        )

    def per_op_table(
        self, device_index: Optional[int], by_kernel: bool = False
    ) -> List[PerOpRecord]:
        """Cost per HLO operation, or per kernel with `by_kernel`."""
        import magneton_eprof

        if not self.kernels:
            return []

        def label(k: "JaxKernel") -> str:
            if by_kernel or not k.hlo_op:
                return k.name
            return k.hlo_op

        corr_to_op = {i: label(k) for i, k in enumerate(self.kernels)}
        kernels = [(k.start_ns, k.end_ns, k.device, i) for i, k in enumerate(self.kernels)]
        cpu_ops = [(label(k), 0) for k in self.kernels]

        rows = magneton_eprof.attribute(cpu_ops, kernels, self.power, corr_to_op, device_index)
        return [
            PerOpRecord(
                op_name=name, num_calls=calls, num_kernels=kernel_count,
                cpu_time_ns=cpu_ns, gpu_time_ns=gpu_ns, gpu_energy_j=energy,
            )
            for name, calls, kernel_count, cpu_ns, gpu_ns, energy in rows
        ]

    def per_hlo_op(self, device_index: Optional[int] = None) -> List["HloOpCost"]:
        """Each HLO operation, its kernels, and what each of them cost."""
        import magneton_eprof

        if not self.kernels:
            return []

        # A label per launch rather than per name, so `attribute` returns one
        # row for each and nothing is aggregated behind our back.
        labels = [str(i) for i in range(len(self.kernels))]
        corr_to_op = dict(enumerate(labels))
        launches = [
            (k.start_ns, k.end_ns, k.device, i) for i, k in enumerate(self.kernels)
        ]
        rows = magneton_eprof.attribute(
            [(label, 0) for label in labels], launches, self.power,
            corr_to_op, device_index,
        )
        by_launch = {
            name: (gpu_ns, energy)
            for name, _calls, _kernels, _cpu, gpu_ns, energy in rows
        }

        ops: Dict[str, HloOpCost] = {}
        for i, kernel in enumerate(self.kernels):
            gpu_ns, energy = by_launch.get(str(i), (0, 0.0))
            # A kernel with no hlo_op is one the trace did not attribute; it is
            # its own operation rather than being dropped.
            op = ops.setdefault(
                kernel.hlo_op or kernel.name,
                HloOpCost(name=kernel.hlo_op or kernel.name, module=kernel.hlo_module),
            )
            op.num_launches += 1
            op.gpu_time_ns += gpu_ns
            op.gpu_energy_j += energy

            for existing in op.kernels:
                if existing.name == kernel.name:
                    entry = existing
                    break
            else:
                entry = KernelCost(name=kernel.name)
                op.kernels.append(entry)
            entry.num_launches += 1
            entry.gpu_time_ns += gpu_ns
            entry.gpu_energy_j += energy

        for op in ops.values():
            op.kernels.sort(key=lambda k: -k.gpu_energy_j)
        return sorted(ops.values(), key=lambda o: -o.gpu_energy_j)

    def kernel_costs(self, device_index: Optional[int] = None) -> List["LaunchCost"]:
        """Every kernel launch with when it ran and what it cost."""
        import magneton_eprof

        if not self.kernels:
            return []
        labels = [str(i) for i in range(len(self.kernels))]
        launches = [
            (k.start_ns, k.end_ns, k.device, i) for i, k in enumerate(self.kernels)
        ]
        rows = magneton_eprof.attribute(
            [(label, 0) for label in labels], launches, self.power,
            dict(enumerate(labels)), device_index,
        )
        by_launch = {
            name: (gpu_ns, energy)
            for name, _c, _k, _cpu, gpu_ns, energy in rows
        }
        out = []
        for i, k in enumerate(self.kernels):
            gpu_ns, energy = by_launch.get(str(i), (0, 0.0))
            out.append(LaunchCost(
                name=k.name, hlo_op=k.hlo_op, start_ns=k.start_ns, end_ns=k.end_ns,
                gpu_time_ns=gpu_ns, gpu_energy_j=energy,
            ))
        return out

    def export_chrome_trace(self, path: str) -> None:
        """Copies out the trace JAX wrote. It is already a chrome trace."""
        if not self.trace:
            raise RuntimeError("no JAX trace was produced")
        if self.trace.endswith(".gz"):
            with gzip.open(self.trace, "rb") as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(self.trace, path)

    @property
    def total_gpu_time_ns(self) -> int:
        return sum(k.duration_ns for k in self.kernels)


# --- reading a JAX trace -----------------------------------------------------


@dataclass
class LaunchCost:
    """One kernel launch: when it ran, and what that cost."""

    name: str
    hlo_op: str = ""
    start_ns: int = 0
    end_ns: int = 0
    gpu_time_ns: int = 0
    gpu_energy_j: float = 0.0


@dataclass
class KernelCost:
    """One kernel of a compiled program, and what running it cost."""

    name: str
    num_launches: int = 0
    gpu_time_ns: int = 0
    gpu_energy_j: float = 0.0

    @property
    def gpu_time_us(self) -> float:
        return self.gpu_time_ns / 1e3


@dataclass
class HloOpCost:
    """One operation of a compiled program, and the kernels that ran it."""

    name: str
    module: str = ""
    num_launches: int = 0
    gpu_time_ns: int = 0
    gpu_energy_j: float = 0.0
    kernels: List[KernelCost] = field(default_factory=list)

    @property
    def gpu_time_us(self) -> float:
        return self.gpu_time_ns / 1e3

    @property
    def energy_mj(self) -> float:
        return self.gpu_energy_j * 1e3


@dataclass
class JaxKernel:
    """One GPU kernel, as the JAX trace describes it."""

    name: str
    start_ns: int
    end_ns: int
    device: int
    hlo_op: str = ""
    hlo_module: str = ""

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def _find_trace(directory: str) -> Optional[str]:
    """The trace JAX just wrote, wherever it put it."""
    for pattern in ("**/*.trace.json.gz", "**/*.trace.json", "**/*.json.gz"):
        found = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))
        if found:
            return found[-1]
    return None


def _load_trace(path: str) -> Dict[str, Any]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def _kernels_from_trace(
    trace: Dict[str, Any], anchor_wall_ns: int, anchor_name: str
) -> Tuple[List[JaxKernel], int]:
    """Reads the GPU kernels out, on the sampler's clock."""
    events = trace.get("traceEvents", [])

    gpu_pids: Dict[int, int] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            label = str(event.get("args", {}).get("name", ""))
            if "/device:GPU:" in label:
                gpu_pids[event["pid"]] = int(label.rsplit(":", 1)[1])

    anchor_ts_us = next(
        (e["ts"] for e in events if e.get("name") == anchor_name and "ts" in e), None
    )
    skew = anchor_wall_ns - int(anchor_ts_us * 1000) if anchor_ts_us is not None else 0

    kernels = []
    for event in events:
        if event.get("ph") != "X" or event.get("pid") not in gpu_pids:
            continue
        duration_us = event.get("dur")
        if not duration_us:
            continue
        arguments = event.get("args", {})
        start = int(event["ts"] * 1000) + skew
        kernels.append(
            JaxKernel(
                name=event.get("name", "?"),
                start_ns=start,
                end_ns=start + int(duration_us * 1000),
                device=gpu_pids[event["pid"]],
                hlo_op=str(arguments.get("hlo_op", "")),
                hlo_module=str(arguments.get("hlo_module", "")),
            )
        )

    kernels.sort(key=lambda k: k.start_ns)
    return kernels, skew
