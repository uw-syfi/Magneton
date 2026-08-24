"""Recording a JAX computation into the same DAG the torch recorder fills."""

import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

import magneton  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the handoff is a device pointer"
)


def test_a_jaxpr_becomes_a_dataflow_graph():
    x = jnp.ones((8, 8)) * 0.5

    def f(a):
        return jnp.sum(jnp.tanh(a @ a))

    with magneton.record(
        f, ([x], {}), framework="jax") as (rec, run_it):
        run_it(x)
    run = rec.run("jax")

    assert [n.target for n in run.dag.nodes.values()] == [
        "dot_general", "tanh", "reduce_sum",
    ]


def test_the_graph_has_edges():
    """The test that fails if tensor identity is lost in conversion."""
    x = jnp.ones((8, 8)) * 0.5

    def f(a):
        return jnp.sum(jnp.tanh(a @ a))

    with magneton.record(
        f, ([x], {}), framework="jax") as (rec, run_it):
        run_it(x)
    dag = rec.run("jax").dag

    assert dag.build_edges(), "the recorded graph has no edges"


@needs_cuda
def test_the_handoff_does_not_leave_the_device():
    """A round trip through numpy would work and would cost two copies."""
    from magneton.recorders.jaxpr import _TensorCache

    x = jnp.ones((4, 4)) * 3
    tensor = _TensorCache().convert(x)
    assert tensor.is_cuda, "the tensor came back on the host"
    assert float(tensor.sum()) == pytest.approx(48.0)


def test_the_same_array_converts_to_the_same_tensor():
    from magneton.recorders.jaxpr import _TensorCache

    cache = _TensorCache()
    x = jnp.ones((4, 4))
    assert cache.convert(x) is cache.convert(x)


@needs_cuda
def test_a_jax_graph_matches_an_equivalent_torch_graph():
    """The whole point: two frameworks, one comparison, matched on values."""
    from magneton import compare

    values = np.random.RandomState(0).randn(32, 32).astype("float32") * 0.1

    def f(a):
        return jnp.sum(jnp.tanh(a @ a))

    class M(torch.nn.Module):
        def forward(self, a):
            return torch.sum(torch.tanh(a @ a))

    with magneton.record(f, ([jnp.asarray(values)], {}), framework="jax") as (rec, run_it):
        run_it(jnp.asarray(values))
    jax_run = rec.run("jax")

    x = torch.from_numpy(values).cuda()
    with magneton.record(M().cuda(), ([x], {}), clone_outputs=True) as (rec, compiled):
        compiled(x)
    torch_run = rec.run("torch")

    report = compare.compare(jax_run, torch_run)
    assert report.regions, "the two graphs compute the same thing and did not match"


def test_profiling_jax_after_torch_says_so_rather_than_returning_nothing():
    """CUPTI has one subscriber, and libkineto does not give it back."""
    from magneton.eprof import profiler as P

    was = P._TORCH_HAS_CAPTURED
    try:
        P._TORCH_HAS_CAPTURED = True
        backend = P._JaxBackend(devices=[0], trace_dir=None)
        with pytest.raises(RuntimeError, match="cannot be profiled after"):
            backend.start()
    finally:
        P._TORCH_HAS_CAPTURED = was


def test_a_backend_measuring_the_wrong_framework_is_refused():
    """The failure this prevents is silent, which is why it is checked."""
    from magneton.eprof import EnergyBackend

    with pytest.raises(ValueError, match="needs a .jax. capture"):
        magneton.record(
            lambda a: a, ([jnp.ones((2, 2))], {}),
            framework="jax", backend=EnergyBackend(devices=[0]),
        )


def test_plugins_are_refused_rather_than_ignored():
    """There is no FX transform in a jaxpr to install one into."""
    with pytest.raises(ValueError, match="plugins need"):
        magneton.record(
            lambda a: a, ([jnp.ones((2, 2))], {}),
            framework="jax", plugins=[object()],
        )


def test_an_unknown_framework_is_named():
    with pytest.raises(ValueError, match="framework must be"):
        magneton.record(lambda a: a, ([1], {}), framework="mxnet")
