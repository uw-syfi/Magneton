"""Recording a run, and the backend that measures it."""

import pytest

torch = pytest.importorskip("torch")

import magneton  # noqa: E402
from magneton import compare  # noqa: E402
from magneton.backends.base import Cost, CostBackend  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the recorder compiles for CUDA"
)


class Two(torch.nn.Module):
    """Two layers, so a recording has more than one node to talk about."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 64)
        self.b = torch.nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(self.b(torch.relu(self.a(x))))


def _inputs(device):
    return torch.randn(8, 64, device=device)


# --- recording alone ---------------------------------------------------------


@needs_cuda
def test_recording_needs_no_backend():
    """Structure without cost."""
    x = _inputs("cuda")
    with magneton.record(Two().cuda(), ([x], {})) as (rec, compiled):
        compiled(x)

    run = rec.run("plain")
    assert len(run.dag.nodes) > 0
    assert run.per_node == {}, "nothing measured it, so there is nothing to report"
    assert not run.has_energy


@needs_cuda
def test_two_recordings_of_the_same_model_match():
    x = _inputs("cuda")
    model = Two().cuda()

    runs = []
    for label in ("first", "second"):
        with magneton.record(model, ([x], {}), clone_outputs=True) as (rec, compiled):
            compiled(x)
        runs.append(rec.run(label))

    report = compare.compare(*runs)
    assert report.regions, "identical models should agree somewhere"


@needs_cuda
def test_the_recorder_reports_what_it_cannot_do():
    x = _inputs("cuda")
    with magneton.record(Two().cuda(), ([x], {}), record_dataflow=False) as (rec, m):
        m(x)
    with pytest.raises(RuntimeError, match="nothing was recorded"):
        rec.run("empty")


# --- the backend seam --------------------------------------------------------


class Fake:
    """A backend that measures nothing, to show the interface is the interface."""

    def __init__(self):
        self.started = self.stopped = False

    def plugins(self):
        return []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def cost_by_annotation(self, prefix):
        return {f"{prefix}0_input": Cost(num_calls=1, gpu_time_ns=1234)}


@needs_cuda
def test_a_backend_from_outside_is_driven_correctly():
    backend = Fake()
    assert isinstance(backend, CostBackend)

    x = _inputs("cuda")
    with magneton.record(Two().cuda(), ([x], {}), backend=backend) as (rec, compiled):
        assert backend.started, "measuring should begin before the region runs"
        assert not backend.stopped
        compiled(x)
    assert backend.stopped

    costs = rec.costs()
    assert costs and next(iter(costs.values())).gpu_time_ns == 1234


@needs_cuda
def test_the_warm_up_happens_before_the_backend_starts():
    """The dataflow recorder computes tensor statistics on."""
    order = []

    class Watcher(Fake):
        def start(self):
            order.append("start")

        def cost_by_annotation(self, prefix):
            return {}

    class Noisy(Two):
        def forward(self, x):
            order.append("forward")
            return super().forward(x)

    x = _inputs("cuda")
    with magneton.record(Noisy().cuda(), ([x], {}), backend=Watcher()) as (rec, m):
        pass

    assert order.index("forward") < order.index("start"), order


# --- what a backend contributes ----------------------------------------------


@needs_cuda
def test_a_backends_plugins_are_installed_alongside_the_recorder():
    seen = []

    class Counting(Fake):
        def plugins(self):
            class P:
                def before_execute(self, op_id, op_name, args, kwargs):
                    seen.append(op_name)
                    return {}

                def after_execute(self, op_id, op_name, output, context):
                    return output

                def wrap_execute(self, op_callable, context):
                    return op_callable()

            return [P()]

    x = _inputs("cuda")
    backend = Counting()
    with magneton.record(Two().cuda(), ([x], {}), backend=backend) as (rec, compiled):
        compiled(x)

    assert seen, "the backend's plugin never ran"
    # The dataflow recorder is still installed too, not displaced by it.
    assert len(rec.dataflow_dag.nodes) > 0


@needs_cuda
def test_cuda_event_timing_reports_latency_and_no_energy():
    x = _inputs("cuda")
    with magneton.record(
        Two().cuda(), ([x], {}), backend=magneton.CudaEventTiming()
    ) as (rec, compiled):
        for _ in range(5):
            compiled(x)

    run = rec.run("timed")
    assert run.per_node, "the timing backend reported nothing"
    assert all(c.gpu_time_ns > 0 for c in run.per_node.values())
    assert not run.has_energy, "CUDA events cannot see power"
    assert all(c.num_calls == 5 for c in run.per_node.values())


# --- what the graph transform must not lose ----------------------------------


@needs_cuda
def test_a_get_attr_node_survives_the_pass():
    """`get_attr` is not an operation, so the."""
    from magneton import transform

    class HasBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gain", torch.full((4,), 3.0))

        def forward(self, x):
            return x * self.gain

    gm = torch.fx.symbolic_trace(HasBuffer())
    assert any(n.op == "get_attr" for n in gm.graph.nodes), "no get_attr to lose"

    wrapped = transform.pluggable_pass(gm, [_Silent()])
    assert any(n.op == "get_attr" for n in wrapped.graph.nodes), "the pass dropped it"

    x = torch.ones(4)
    assert torch.equal(wrapped(x), torch.full((4,), 3.0))


def test_an_unknown_node_kind_is_refused_rather_than_dropped():
    from magneton import transform

    class Odd:
        op = "something_new"
        name = "n0"
        args = ()
        kwargs = {}

    class FakeGraph:
        nodes = [Odd()]

    class FakeGM:
        graph = FakeGraph()

    with pytest.raises(RuntimeError, match="does not know what to do"):
        transform.pluggable_pass(FakeGM(), [_Silent()])


class _Silent:
    """A plugin that observes and changes nothing."""

    def before_execute(self, op_id, op_name, args, kwargs):
        return {}

    def after_execute(self, op_id, op_name, output, context):
        return output

    def wrap_execute(self, op_callable, context):
        return op_callable()
