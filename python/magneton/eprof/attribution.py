"""In-memory per-operator latency + energy attribution.

This consumes an ``magneton_eprof._ProfilerResult`` directly -- the in-memory IR: the
flat event list plus the ``experimental_event_tree`` -- and produces
per-operator latency and energy, without a chrome trace ever being written.

Energy semantics (per the project decision): a GPU kernel's energy is the
integral of *board* power over the kernel's execution window, with board power
split evenly across all kernels running concurrently at each instant. NVML
reports whole-board power, so concurrent kernels on different streams share it;
splitting by concurrency keeps per-kernel energies additive up to the board
total over any interval where the board is busy.

Each kernel's energy/latency is then attributed to the nearest enclosing
``aten::`` operator, recovered by walking the event tree
(aten op -> cudaLaunchKernel -> GPU kernel are linked by correlation id /
flow in the tree).
"""

from __future__ import annotations

from magneton.eprof import treediff
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

# c10::DeviceType integer values (see Event::DeviceType binding).
# libkineto activity types that are GPU work: a kernel, a copy, a fill. Not
# GPU_USER_ANNOTATION, which is a span drawn *over* those to label them -- it
# reports the same microseconds a second time, so counting it doubles both the
# GPU time and the energy of everything it covers.
_GPU_WORK = frozenset({
    3,  # GPU_MEMCPY
    4,  # GPU_MEMSET
    5,  # CONCURRENT_KERNEL
})

_DEVICE_CPU = 0
_DEVICE_CUDA = 1

_UNATTRIBUTED = "<unattributed>"


@dataclass
class PerOpRecord:
    """Aggregated latency + energy for one operator name."""

    op_name: str
    num_calls: int = 0          # number of aten-op invocations
    num_kernels: int = 0        # GPU kernels attributed to this op
    cpu_time_ns: int = 0        # summed CPU-side op duration
    gpu_time_ns: int = 0        # summed GPU kernel duration
    gpu_energy_j: float = 0.0   # summed concurrency-split kernel energy

    @property
    def gpu_time_us(self) -> float:
        return self.gpu_time_ns / 1e3

    @property
    def cpu_time_us(self) -> float:
        return self.cpu_time_ns / 1e3


@dataclass
class _Kernel:
    start_ns: int
    end_ns: int
    name: str
    correlation_id: int
    device_index: int


class PowerTimeline:
    """Piecewise-linear board-power curve from NVML samples for one device."""

    def __init__(self, samples: List[Tuple[int, float]]):
        # samples: (timestamp_ns, watts), assumed pre-sorted by time.
        self._ts = [t for t, _ in samples]
        self._w = [w for _, w in samples]

    def __bool__(self) -> bool:
        return len(self._ts) >= 2

    def power_at(self, t_ns: int) -> float:
        """Linearly interpolated power (W) at time ``t_ns`` (clamped at ends)."""
        ts = self._ts
        if not ts:
            return 0.0
        if t_ns <= ts[0]:
            return self._w[0]
        if t_ns >= ts[-1]:
            return self._w[-1]
        i = bisect_right(ts, t_ns) - 1
        t0, t1 = ts[i], ts[i + 1]
        w0, w1 = self._w[i], self._w[i + 1]
        if t1 == t0:
            return w0
        frac = (t_ns - t0) / (t1 - t0)
        return w0 + frac * (w1 - w0)

    def sample_points_in(self, lo_ns: int, hi_ns: int) -> List[int]:
        """Power-sample timestamps strictly inside (lo, hi)."""
        ts = self._ts
        out = []
        i = bisect_right(ts, lo_ns)
        while i < len(ts) and ts[i] < hi_ns:
            out.append(ts[i])
            i += 1
        return out


def _energy_concurrency_split(
    kernels: List[_Kernel], power: PowerTimeline
) -> Dict[int, float]:
    """Energy (J) per kernel index, board power split by concurrency.

    Sweep over the union of kernel boundaries and power-sample timestamps. On
    each segment both the active-kernel set and the (piecewise-linear) power
    curve are constant/linear, so the segment energy is a trapezoid; it is
    divided equally among the kernels active on that segment.
    """
    energy: Dict[int, float] = {i: 0.0 for i in range(len(kernels))}
    if not kernels or not power:
        return energy

    starts: Dict[int, List[int]] = {}
    ends: Dict[int, List[int]] = {}
    breakpoints = set()
    lo = min(k.start_ns for k in kernels)
    hi = max(k.end_ns for k in kernels)
    for idx, k in enumerate(kernels):
        starts.setdefault(k.start_ns, []).append(idx)
        ends.setdefault(k.end_ns, []).append(idx)
        breakpoints.add(k.start_ns)
        breakpoints.add(k.end_ns)
    for t in power.sample_points_in(lo, hi):
        breakpoints.add(t)

    points = sorted(breakpoints)
    active: set[int] = set()
    for seg_i in range(len(points) - 1):
        t0 = points[seg_i]
        t1 = points[seg_i + 1]
        # At t0: kernels ending exactly here leave, then kernels starting enter
        # ([start, end) semantics).
        for idx in ends.get(t0, []):
            active.discard(idx)
        for idx in starts.get(t0, []):
            active.add(idx)
        if not active or t1 <= t0:
            continue
        dt_s = (t1 - t0) / 1e9
        p_avg_w = (power.power_at(t0) + power.power_at(t1)) / 2.0
        seg_energy = p_avg_w * dt_s
        share = seg_energy / len(active)
        for idx in active:
            energy[idx] += share
    return energy


def _build_correlation_to_op(tree) -> Dict[int, str]:
    """Map correlation id -> nearest enclosing aten-op name, via the tree.

    aten op (TorchOp) -> cudaLaunchKernel (Kineto) -> GPU kernel (Kineto) are
    linked as parent/child with a shared correlation id, so a kernel's
    correlation id resolves to the innermost TorchOp ancestor.
    """
    out: Dict[int, str] = {}

    def visit(node, aten_ancestor: Optional[str]):
        # Accept both the C++ _ProfilerEvent ('_EventType.TorchOp') and the
        # Rust-materialized RawNode ('TorchOp') tag spellings.
        if str(node.tag).split(".")[-1] == "TorchOp":
            aten_ancestor = node.name
        cid = node.correlation_id
        if cid and aten_ancestor is not None and cid not in out:
            out[cid] = aten_ancestor
        for child in node.children:
            visit(child, aten_ancestor)

    for root in tree:
        visit(root, None)
    return out


# What an annotating transform is expected to have named its scopes, when the
# caller does not say. Whoever does the annotating owns the real string and
# should pass it; this default is only so that reading a trace produced by
# magneton's transform needs no ceremony.
DEFAULT_ANNOTATION_PREFIX = "forward_op_wrapper_"


def _by_annotation(tree, prefix: str) -> Tuple[Dict[int, str], List[Tuple[str, int]]]:
    """Attribute to annotation scopes instead of aten ops.

    Same walk as `_build_correlation_to_op`, tracking the enclosing annotation
    frame rather than the enclosing aten op. Returns the correlation map and
    the CPU durations, both keyed by frame name.
    """
    correlation_to_node: Dict[int, str] = {}
    cpu_times: List[Tuple[str, int]] = []

    def visit(node, wrapper: Optional[str]):
        name = str(node.name)
        if name.startswith(prefix):
            # A wrapper never nests inside another, but if the graph ever
            # changes that, the innermost is the one that did the work.
            wrapper = name
            cpu_times.append((name, max(0, node.duration_time_ns)))
        cid = node.correlation_id
        if cid and wrapper is not None and cid not in correlation_to_node:
            correlation_to_node[cid] = wrapper
        for child in node.children:
            visit(child, wrapper)

    for root in tree:
        visit(root, None)
    return correlation_to_node, cpu_times


def attribute_by_annotation(
    result,
    device_index: Optional[int] = None,
    annotation_prefix: str = DEFAULT_ANNOTATION_PREFIX,
) -> Dict[str, PerOpRecord]:
    """Latency and energy per annotation scope, keyed by the scope's name.

    `attribute` groups by aten op name, which answers "what did addmm cost".
    This groups by whatever scopes the caller annotated the run with, which
    answers "what did *this part of the model* cost".

    The profiler has no idea what those scopes mean. Something upstream opened
    a `record_function` around each region it wants costed and named them with
    a common prefix; every kernel launched inside one is charged to it. That
    the regions happen to be FX graph nodes is magneton's business, not this
    module's.

    Args:
        result: an ``magneton_eprof._ProfilerResult``
        device_index: restrict to one GPU; ``None`` uses all devices found
        annotation_prefix: what the annotator named its scopes

    Returns:
        scope name -> its record.
    """
    # The extension, imported here rather than at the top of the file because
    # magneton has to import without it -- see magneton/eprof/__init__.py.
    import magneton_eprof

    raw = result.export_raw_nodes()
    _, kernels, power = _extract_from_raw(raw)
    tree = _tree_from_raw_nodes(raw)
    correlation_to_node, cpu_ops = _by_annotation(tree, annotation_prefix)

    rows = magneton_eprof.attribute(cpu_ops, kernels, power, correlation_to_node, device_index)
    return {
        name: PerOpRecord(
            op_name=name, num_calls=calls, num_kernels=kernel_count,
            cpu_time_ns=cpu_ns, gpu_time_ns=gpu_ns, gpu_energy_j=energy,
        )
        for name, calls, kernel_count, cpu_ns, gpu_ns, energy in rows
    }


def _tree_from_raw_nodes(raw):
    """The materialized event tree, as both attributions need it."""
    import magneton_eprof

    tuples = [
        (d["id"], d["tag"], d["start_tid"], d["forward_tid"], d["start_ns"],
         d["start_ns"] + d["dur_ns"], d["flow_id"], d["flow_type"],
         d["flow_start"], d["linked_id"])
        for d in raw
    ]
    current_tid = next((d["start_tid"] for d in raw if d["tag"] == 0), 0)
    parents = magneton_eprof.materialize(tuples, current_tid)
    return treediff.tree_from_raw(
        [dict(d, parent_id=p) for d, p in zip(raw, parents)]
    )


def _corr_to_op_from_raw(raw) -> Dict[int, str]:
    """Build correlation->op by materializing the tree in Rust from the POD bridge.

    Depends only on the flat raw nodes (export_raw_nodes) + the Rust
    materialization -- there is no C++ tree/_Event structure anymore.
    """
    import magneton_eprof

    tuples = [
        (d["id"], d["tag"], d["start_tid"], d["forward_tid"], d["start_ns"],
         d["start_ns"] + d["dur_ns"], d["flow_id"], d["flow_type"],
         d["flow_start"], d["linked_id"])
        for d in raw
    ]
    # current_tid: orphan kineto activities are placed on the profiling thread,
    # which is where CPU (TorchOp) events run.
    current_tid = next((d["start_tid"] for d in raw if d["tag"] == 0), 0)
    parents = magneton_eprof.materialize(tuples, current_tid)
    roots = treediff.tree_from_raw(
        [dict(d, parent_id=p) for d, p in zip(raw, parents)]
    )
    return _build_correlation_to_op(roots)


def _extract_from_raw(raw):
    """Build the POD attribution inputs from the export_raw_nodes bridge.

    Both backends consume identical inputs; the device filter is applied inside
    the attribution. device_type is the c10::DeviceType int (CPU=0, CUDA=1).
    """
    cpu_ops = [
        (d["name"], max(0, d["dur_ns"]))
        for d in raw
        if d["device_type"] == _DEVICE_CPU and str_is_aten(d["name"])
    ]
    kernels = [
        (d["start_ns"], d["start_ns"] + d["dur_ns"], d["device_index"],
         d["correlation_id"])
        for d in raw
        if d["device_type"] == _DEVICE_CUDA
        and d["dur_ns"] > 0
        and d["activity_type"] in _GPU_WORK
    ]
    power = [
        (d["start_ns"], d["device_index"], d["power_usage"])
        for d in raw
        if d["power_usage"] >= 0
    ]
    return cpu_ops, kernels, power


def attribute(
    result, device_index: Optional[int] = None, backend: str = "auto"
) -> List[PerOpRecord]:
    """Attribute per-operator latency + energy from a ``_ProfilerResult``.

    Args:
        result: an ``magneton_eprof._ProfilerResult``.
        device_index: restrict to one GPU; ``None`` uses all devices found.
        backend: ``"rust"`` (the eprof extension), ``"python"`` (the
            reference implementation below), or ``"auto"`` (rust if available).

    Returns:
        ``PerOpRecord`` list, sorted by descending GPU energy.
    """
    raw = result.export_raw_nodes()
    cpu_ops, kernels, power = _extract_from_raw(raw)
    corr_to_op = _corr_to_op_from_raw(raw)

    if backend in ("auto", "rust"):
        try:
            import magneton_eprof
        except ImportError:
            if backend == "rust":
                raise
            magneton_eprof = None
        if magneton_eprof is not None:
            rows = magneton_eprof.attribute(cpu_ops, kernels, power, corr_to_op, device_index)
            return [
                PerOpRecord(
                    op_name=name, num_calls=nc, num_kernels=nk,
                    cpu_time_ns=cpu, gpu_time_ns=gpu, gpu_energy_j=energy,
                )
                for (name, nc, nk, cpu, gpu, energy) in rows
            ]

    return _attribute_python(cpu_ops, kernels, power, corr_to_op, device_index)


def _attribute_python(
    cpu_ops, kernels, power, corr_to_op, device_index: Optional[int] = None
) -> List[PerOpRecord]:
    """Pure-Python reference attribution over the POD bridge inputs (oracle).

    Mirrors eprof::attribution exactly so the dual-run test stays valid.
    Inputs: cpu_ops=[(name, dur_ns)], kernels=[(start, end, dev, corr)],
    power=[(t, dev, mw)], corr_to_op={corr: op_name}.
    """
    # 1. Board-power timelines, per device.
    power_by_dev: Dict[int, List[Tuple[int, float]]] = {}
    for (t, dev, mw) in power:
        power_by_dev.setdefault(dev, []).append((t, mw / 1000.0))
    for dev in power_by_dev:
        power_by_dev[dev].sort()
    timelines = {dev: PowerTimeline(s) for dev, s in power_by_dev.items()}

    # 2. GPU kernels (filtered by device).
    ks: List[_Kernel] = []
    for (start, end, dev, corr) in kernels:
        if device_index is not None and dev != device_index:
            continue
        if end <= start:
            continue
        ks.append(_Kernel(start_ns=start, end_ns=end, name="",
                          correlation_id=corr, device_index=dev))

    # 3. Energy per kernel, integrated per device with concurrency splitting.
    kernel_energy: Dict[int, float] = {}
    by_dev: Dict[int, List[int]] = {}
    for i, k in enumerate(ks):
        by_dev.setdefault(k.device_index, []).append(i)
    for dev, idxs in by_dev.items():
        sub = [ks[i] for i in idxs]
        sub_energy = _energy_concurrency_split(sub, timelines.get(dev, PowerTimeline([])))
        for local_i, global_i in enumerate(idxs):
            kernel_energy[global_i] = sub_energy[local_i]

    # 4. Aggregate (insertion order, stable sort -> matches the Rust backend).
    records: Dict[str, PerOpRecord] = {}
    order: List[str] = []

    def rec(name: str) -> PerOpRecord:
        if name not in records:
            order.append(name)
            records[name] = PerOpRecord(op_name=name)
        return records[name]

    for (name, dur) in cpu_ops:
        r = rec(name)
        r.num_calls += 1
        r.cpu_time_ns += max(0, dur)

    for i, k in enumerate(ks):
        op_name = corr_to_op.get(k.correlation_id, _UNATTRIBUTED)
        r = rec(op_name)
        r.num_kernels += 1
        r.gpu_time_ns += (k.end_ns - k.start_ns)
        r.gpu_energy_j += kernel_energy.get(i, 0.0)

    return sorted(
        (records[n] for n in order),
        key=lambda r: r.gpu_energy_j,
        reverse=True,
    )


def str_is_aten(name: str) -> bool:
    """Heuristic: treat dispatcher ops (aten::, c10d::, etc.) as operators."""
    return "::" in name


def records_to_dicts(records: List[PerOpRecord]) -> List[dict]:
    """Plain-dict view of records, suitable for JSON / DataFrame."""
    return [
        {
            "op_name": r.op_name,
            "num_calls": r.num_calls,
            "num_kernels": r.num_kernels,
            "cpu_time_ns": r.cpu_time_ns,
            "gpu_time_ns": r.gpu_time_ns,
            "gpu_energy_j": r.gpu_energy_j,
        }
        for r in records
    ]


def format_comparison(
    a: List[PerOpRecord],
    b: List[PerOpRecord],
    label_a: str = "a",
    label_b: str = "b",
    top: int = 12,
    per: int = 1,
) -> str:
    """Two runs side by side, the operators that differ most listed first.

    This is what localizes a change to an operator when the two runs are the
    same program done two ways: whatever the difference between them is, it
    has to show up in the rows where the energy moved. Operators present on
    only one side are included, with zero on the other -- an operator that the
    fix removed is exactly the thing being looked for.
    """
    ax = {r.op_name: r for r in a}
    bx = {r.op_name: r for r in b}
    rows = []
    for name in set(ax) | set(bx):
        ea = ax[name].gpu_energy_j * 1e3 / per if name in ax else 0.0
        eb = bx[name].gpu_energy_j * 1e3 / per if name in bx else 0.0
        ga = ax[name].gpu_time_us / per if name in ax else 0.0
        gb = bx[name].gpu_time_us / per if name in bx else 0.0
        rows.append((name, ga, ea, gb, eb, eb - ea))
    rows.sort(key=lambda r: -abs(r[5]))

    width = 34 + 22 + 22 + 12
    lines = [
        f"{'operator':<34}{label_a[:20] + ' us/mJ':>22}"
        f"{label_b[:20] + ' us/mJ':>22}{'delta mJ':>12}",
        "-" * width,
    ]
    for name, ga, ea, gb, eb, delta in rows[:top]:
        lines.append(
            f"{name[:33]:<34}{ga:>11.1f} /{ea:>9.2f}{gb:>11.1f} /{eb:>9.2f}"
            f"{delta:>+12.2f}"
        )
    if len(rows) > top:
        lines.append(f"{f'... and {len(rows) - top} more operators':<34}")
    lines.append("-" * width)
    ta = sum(r.gpu_energy_j for r in a) * 1e3 / per
    tb = sum(r.gpu_energy_j for r in b) * 1e3 / per
    lines.append(f"{'total':<34}{'':>11}  {ta:>9.2f}{'':>11}  {tb:>9.2f}{tb - ta:>+12.2f}")
    return "\n".join(lines)


def save_json(records: List[PerOpRecord], path: str) -> None:
    """Write per-op records to a JSON file (the PerOp sink)."""

    with open(path, "w") as f:
        json.dump(records_to_dicts(records), f, indent=2)


def format_table(records: List[PerOpRecord], top: int = 20, per: int = 1) -> str:
    """Render per-op records as a fixed-width table, largest energy first.

    `per` divides every figure by that many repetitions, which is how to get
    per-call numbers out of a region that ran the workload in a loop. The loop
    is usually there because the power sampler needs something to integrate
    over, not because the caller wanted a total.
    """
    unit = "" if per == 1 else " per call"
    lines = [
        f"{'operator':<34}{'calls':>7}{'kernels':>9}"
        f"{'cpu_us':>12}{'gpu_us':>12}{'energy_mJ':>12}{unit}",
        "-" * 86,
    ]
    total_e = sum(r.gpu_energy_j for r in records)
    for r in records[:top]:
        lines.append(
            f"{r.op_name[:33]:<34}{r.num_calls // per:>7}{r.num_kernels // per:>9}"
            f"{r.cpu_time_us / per:>12.1f}{r.gpu_time_us / per:>12.1f}"
            f"{r.gpu_energy_j * 1e3 / per:>12.4f}"
        )
    if len(records) > top:
        lines.append(f"{f'... and {len(records) - top} more operators':<34}")
    lines.append("-" * 86)
    lines.append(
        f"{'total':<34}{'':<16}{'':>24}{total_e * 1e3 / per:>12.4f}"
    )
    return "\n".join(lines)
