"""Tests for in-memory per-operator latency + energy attribution.

These validate the modular attribution path (eprof.attribution) that replaces
the chrome-trace round-trip. The key correctness property is energy
conservation: with board power split by concurrency, the total energy
attributed across all kernels must equal the integral of board power over the
union of kernel-active intervals.
"""

import pytest
import torch

import magneton_eprof as C
from magneton.eprof import attribution as A


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="attribution test needs a CUDA device"
)


def _profile_workload():
    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        True, False, False, False, False, True, [0],
    )
    prof.start(set())
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    for _ in range(30):
        c = a @ b
        d = torch.relu(c)
        _ = d + a
    torch.cuda.synchronize()
    return prof.stop()


def _busy_window_energy(result, device_index=0):
    """Oracle: integral of board power over the union of kernel intervals.

    Sourced from the POD bridge (export_raw_nodes), like attribution itself.
    """
    raw = result.export_raw_nodes()
    power = sorted(
        (d["start_ns"], d["power_usage"] / 1000.0)
        for d in raw
        if d["power_usage"] >= 0 and d["device_index"] == device_index
    )
    tl = A.PowerTimeline(power)
    intervals = sorted(
        (d["start_ns"], d["start_ns"] + d["dur_ns"])
        for d in raw
        if d["device_type"] == 1 and d["dur_ns"] > 0
        and d["device_index"] == device_index
    )
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    def integ(s, e):
        pts = [s] + tl.sample_points_in(s, e) + [e]
        tot = 0.0
        for i in range(len(pts) - 1):
            t0, t1 = pts[i], pts[i + 1]
            tot += (tl.power_at(t0) + tl.power_at(t1)) / 2 * (t1 - t0) / 1e9
        return tot

    return sum(integ(s, e) for s, e in merged)


def test_energy_conservation():
    """Total attributed energy == board energy over the busy window."""
    result = _profile_workload()
    records = A.attribute(result, device_index=0)
    attributed = sum(r.gpu_energy_j for r in records)
    oracle = _busy_window_energy(result, device_index=0)
    assert oracle > 0.0
    assert attributed == pytest.approx(oracle, rel=1e-6)


def test_latency_and_attribution():
    """Matmul should dominate energy and have nonzero GPU time, attributed to aten::mm."""
    result = _profile_workload()
    records = A.attribute(result, device_index=0)
    by_name = {r.op_name: r for r in records}

    assert "aten::mm" in by_name
    mm = by_name["aten::mm"]
    assert mm.num_kernels >= 30
    assert mm.gpu_time_ns > 0
    assert mm.gpu_energy_j > 0
    # matmul should be the largest GPU energy consumer here
    assert records[0].op_name == "aten::mm"
    # every GPU-time op should be a real dispatcher op, not unattributed
    assert "<unattributed>" not in by_name or by_name["<unattributed>"].gpu_time_ns == 0


def test_rust_backend_matches_python():
    """The Rust core (magneton_eprof) must produce identical records to the Python reference."""
    pytest.importorskip("magneton_eprof")
    result = _profile_workload()
    py = A.attribute(result, device_index=0, backend="python")
    rs = A.attribute(result, device_index=0, backend="rust")
    assert [r.op_name for r in py] == [r.op_name for r in rs]
    for p, r in zip(py, rs):
        assert (p.num_calls, p.num_kernels, p.cpu_time_ns, p.gpu_time_ns) == (
            r.num_calls, r.num_kernels, r.cpu_time_ns, r.gpu_time_ns
        )
        assert r.gpu_energy_j == pytest.approx(p.gpu_energy_j, abs=1e-9)


def test_power_timeline_interpolation():
    tl = A.PowerTimeline([(0, 100.0), (10, 200.0)])
    assert tl.power_at(-5) == 100.0      # clamp low
    assert tl.power_at(15) == 200.0      # clamp high
    assert tl.power_at(5) == pytest.approx(150.0)  # midpoint


def test_annotation_spans_are_not_counted_as_gpu_work():
    """GPU_USER_ANNOTATION overlays the kernels it labels.

    Kineto draws it over the same microseconds the kernel already reported, so
    counting it as work doubles both the GPU time and the energy of everything
    it covers. Attribution over a real trace was ~2x too high until this was
    excluded, and the tell was that the time and energy ratios disagreed.
    """
    from magneton.eprof.attribution import _extract_from_raw

    def node(activity_type, start, dur, **kw):
        base = dict(
            name="aten::mm", device_type=1, device_index=0, dur_ns=dur,
            start_ns=start, correlation_id=7, activity_type=activity_type,
            power_usage=-1,
        )
        base.update(kw)
        return base

    raw = [
        node(5, 1000, 500),   # CONCURRENT_KERNEL: real work
        node(2, 1000, 500),   # GPU_USER_ANNOTATION: the same time, again
        node(3, 2000, 100),   # GPU_MEMCPY: real work
    ]
    _, kernels, _ = _extract_from_raw(raw)

    assert len(kernels) == 2, f"the annotation was counted: {kernels}"
    assert sum(end - start for start, end, _, _ in kernels) == 600


# --- the annotation contract -------------------------------------------------


def test_the_annotator_and_the_reader_agree_on_the_prefix():
    """The only thing connecting a cost back to a graph node is a string.

    `transform` opens a `record_function` per wrapped operation; `attribution`
    finds those scopes again in the trace by matching a prefix. Nothing else
    ties the two together, and nothing in either module would fail loudly if
    one of them changed its mind -- the attribution would simply come back
    empty, which reads like a run that did no GPU work.
    """
    from magneton import transform

    assert A.DEFAULT_ANNOTATION_PREFIX == transform.ANNOTATION_PREFIX


def test_every_wrapped_op_is_annotated_with_that_prefix():
    """Checking the constants agree is not enough: the transform has to
    actually emit names that start with the constant it exports."""
    from magneton import transform
    from magneton.plugin import OpPlugin

    class Silent(OpPlugin):
        def before_execute(self, op_id, op_name, args, kwargs):
            return {}

        def after_execute(self, op_id, op_name, output, context):
            return output

        def wrap_execute(self, op_callable, context):
            return op_callable()

    class Tiny(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 1)

    gm = torch.fx.symbolic_trace(Tiny())
    wrapped = transform.pluggable_pass(gm, [Silent()])

    names = [
        f"forward_{m.op_name}"
        for m in wrapped.children()
        if isinstance(m, transform.OpPluggableWrapper)
    ]
    assert names, "the pass wrapped nothing, so the check proves nothing"
    assert all(n.startswith(transform.ANNOTATION_PREFIX) for n in names), names


def test_a_plugin_need_not_inherit_anything():
    """The protocol is structural, which is what lets a plugin come from
    outside magneton -- a profiler supplying its own replay plugin should not
    have to import magneton to inherit a base class from it."""
    from magneton import transform
    from magneton.plugin import OpPlugin
    from magneton.plugin import PluginManager

    seen = []

    class Foreign:  # no base class, no import of OpPlugin
        def before_execute(self, op_id, op_name, args, kwargs):
            seen.append(op_name)
            return {}

        def after_execute(self, op_id, op_name, output, context):
            return output

        def wrap_execute(self, op_callable, context):
            return op_callable()

    plugin = Foreign()
    assert isinstance(plugin, OpPlugin), "the protocol should accept it"

    # It declares no priority, so the manager has to supply one rather than
    # raising when it sorts.
    assert PluginManager([plugin]).plugins == [plugin]

    class Tiny(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 1)

    gm = transform.pluggable_pass(torch.fx.symbolic_trace(Tiny()), [plugin])
    gm(torch.ones(4))
    assert seen, "the foreign plugin never ran"
