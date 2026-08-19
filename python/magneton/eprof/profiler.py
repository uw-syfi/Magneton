"""Profiling a region, whichever framework runs in it.

    with eprof.Profiler(energy_config=...) as (prof, _):
        model(x)

    with eprof.Profiler(backend="jax", ...) as (prof, _):
        for _ in range(rounds):
            jax.block_until_ready(jitted(x))

    prof.per_op_table()

What is profiled is a block of code, not a module. Nothing here transforms a
graph or wraps a model: attribution works from what the kernels report about
themselves, and grouping them by anything finer than the operator that launched
them needs someone to have annotated the run -- which `attribute_by_annotation`
does, for whoever did the annotating. `magneton` is the usual such caller.

The two frameworks are watched by completely different means -- torch through
libkineto and the dispatcher's callbacks, JAX through its own profiler's trace
-- but what comes out is the same three things: which kernels ran and when,
what the GPU was drawing while they did, and what to charge each one to. The
attribution over those is shared, so what differs between backends is only how
they are collected, which is what `backend` selects.
"""

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

# CUPTI takes one subscriber per process. libkineto claims it when a torch
# capture starts and JAX's profiler cannot get it afterwards -- it still runs,
# still writes a trace, and the trace has no kernels in it. Nothing raises, so
# the only symptom is a run that appears to have done no GPU work. This records
# that a torch capture has happened so the jax backend can say so instead.
_TORCH_HAS_CAPTURED = False


class Profiler:
    """A profiling run.

    Args:
        activities: what to collect. Defaults to CPU and CUDA.
        backend: `"torch"` or `"jax"`.
        tracing_config, energy_config: see `eprof.config`.
        devices: which GPUs to sample power from, for the jax backend. The
            torch backend takes them from `energy_config` instead.
        trace_dir: where the jax backend leaves its trace. A temporary
            directory by default.
    """

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
        """Per-operation GPU time and energy, largest energy first.

        `by_kernel` is for the jax backend, where an operation can lower to
        several library kernels and it is sometimes the kernels you want.
        """
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
        """Each HLO operation, the kernels that ran it, and what they cost.

        jax only: an HLO operation is what XLA compiled, and the torch backend
        has no equivalent -- there, a kernel is already attributed to the aten
        operator that launched it, which `per_op_table` reports.
        """
        if self.backend != JAX:
            raise ValueError(
                "per_hlo_op is a jax notion; the torch backend attributes "
                "kernels to the aten operator that launched them, which "
                "per_op_table reports"
            )
        return self._impl.per_hlo_op(device_index)

    def kernel_costs(self, device_index: Optional[int] = None) -> List["LaunchCost"]:
        """Every kernel launch, with when it ran and what it cost.

        For grouping launches by something neither the trace nor the dispatcher
        can see -- which is what a recorder running one operation at a time
        needs, and the only attribution available for a framework whose kernels
        never pass an aten operator.
        """
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
        """Every kernel launch, with when it ran and what it cost.

        Below the aten operator, and for callers that group launches by
        something the dispatcher cannot see. TensorFlow is the case that needs
        it: CUPTI reports its kernels like any other, but they reach the GPU
        without passing an aten operator, so there is nothing for the ordinary
        attribution to charge them to.
        """
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
    """JAX, through its own profiler's trace and a standalone power sampler.

    Two things this has to get right that the torch backend does not.

    **The clocks.** A JAX trace is timestamped in microseconds from an origin of
    its own, near where tracing started; the sampler is in Unix nanoseconds.
    Integrating power over a kernel's window needs both on one clock, so a
    marker is emitted inside the trace at an instant whose wall-clock time is
    recorded, and that pair fixes the offset. Guessing it from when the trace
    was entered would be off by however long the profiler took to start --
    milliseconds, which is thousands of kernels.

    **The op names.** XLA names a generated kernel after the fused HLO op it
    came from, with the dots turned into underscores (`fusion.14` becomes
    `fusion_14`), which is what makes the names worth grouping by. A kernel that
    is a library call is named for the library's kernel instead. Each kernel's
    `hlo_op` and `hlo_module` are kept alongside for coarser grouping.
    """

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
        """Cost per HLO operation, or per kernel with `by_kernel`.

        Grouping by the HLO op is what makes this comparable to the torch
        backend's table: both answer "what did this operation cost". A kernel
        is below that. XLA names a generated kernel after the fused op it came
        from, so for those two the answer is the same string -- but an op that
        lowers to a library call does not. A convolution runs as three cuDNN
        kernels with names like `sm80_xmma_fprop_implicit_gemm_indexed...`,
        and charging them separately answers a question about cuDNN rather
        than about the model.
        """
        import magneton_eprof

        if not self.kernels:
            return []

        def label(k: "JaxKernel") -> str:
            if by_kernel or not k.hlo_op:
                return k.name
            return k.hlo_op

        # Each kernel gets a correlation id of its own. The trace's real
        # correlation ids identify the launch, and a command buffer launches
        # many kernels under one, which would collapse them into a single row.
        corr_to_op = {i: label(k) for i, k in enumerate(self.kernels)}
        kernels = [(k.start_ns, k.end_ns, k.device, i) for i, k in enumerate(self.kernels)]
        # cpu_time is zero throughout: this sees the GPU side only. JAX's host
        # work is in the trace, but it is XLA's dispatch rather than anything an
        # operation of the model owns.
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
        """Each HLO operation, its kernels, and what each of them cost.

        Attribution runs per kernel launch -- every launch gets a correlation
        id of its own, so nothing is merged before the energy is split -- and
        the launches are then gathered under the operation the trace says they
        belong to. That order matters: energy is board power shared between
        whatever was running concurrently, so it has to be worked out per
        launch and summed afterwards, not the other way round.

        Largest energy first, and the kernels within each operation likewise.
        """
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
        """Every kernel launch with when it ran and what it cost.

        Below `per_hlo_op`, and for callers that group launches by something
        the trace does not know about -- a recorder that ran the program one
        operation at a time knows which of its own operations each launch
        belongs to, by the window it fell in.

        Timestamps are Unix nanoseconds, the same clock `time.time_ns()`
        returns, because that is what the trace was anchored against.
        """
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
    """One operation of a compiled program, and the kernels that ran it.

    XLA names a kernel it generates after the fused operation it came from, so
    for those two the operation and the kernel are the same thing. An operation
    that lowers to a library call is not: a convolution runs as a cuDNN gemm
    and two layout conversions, none of which is named for anything in the
    program. The operation is the level the program was written at, and the
    kernels are how it was carried out.
    """

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
    """Reads the GPU kernels out, on the sampler's clock.

    The trace numbers its processes rather than naming them, so the GPU ones are
    found through the `process_name` metadata events.
    """
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
