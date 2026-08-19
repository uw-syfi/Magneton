"""Recording a JAX program by walking the jaxpr.

Three things this has to get right, and each was a bug before it was a comment.

**Tensors have to be the same object across nodes.** The DAG identifies a
tensor by `id()`, and joins one node's output to the next node's input on that
identity. A dlpack handoff makes a new torch tensor every time it is called, so
converting per node would produce a graph with no edges at all. Every JAX array
is converted once and the result is cached, keyed by the array's identity, with
the array itself held alongside so the identity cannot be recycled underneath
the cache.

**The handoff stays on the device.** `jax.dlpack` to `torch.utils.dlpack` moves
no data; the torch tensor points at the same GPU memory. A round trip through
numpy would work and would cost a copy each way, so it is only the fallback for
arrays dlpack refuses.

**A primitive has to finish before the next one starts.** JAX dispatches
asynchronously, so without blocking, the window an operation appears to occupy
says when it was launched rather than when it ran, and the kernels would be
attributed to whichever operation happened to be executing. Each primitive is
therefore waited on. That is also what makes this a recording rather than a
measurement of the fused program -- see `JaxRecorder.run`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy
import torch

from magneton.backends.base import Cost, CostBackend
from magneton.dataflow import NodeExecution
from magneton.recorder import JAX, BaseRecorder

# What each primitive's scope is called in the trace.
ANNOTATION_PREFIX = "jax_op_"

class _TensorCache:
    """One torch view per JAX array, so node identity survives conversion."""

    def __init__(self) -> None:
        self._by_id: Dict[int, Tuple[Any, Any]] = {}

    def convert(self, value: Any):
        # jax is not a dependency of magneton and this module must be
        # importable without it -- which is the only reason this import is not
        # at the top of the file.
        try:
            import jax
        except ImportError:  # pragma: no cover - only used from a jax program
            return None

        if not isinstance(value, jax.Array):
            return None
        key = id(value)
        hit = self._by_id.get(key)
        if hit is not None:
            return hit[1]

        try:
            tensor = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(value))
        except Exception:
            # Anything dlpack will not take -- a weak-typed scalar, an unusual
            # dtype -- goes the slow way rather than being dropped, because a
            # missing tensor is a missing edge.

            tensor = torch.from_numpy(numpy.asarray(value))
        # The array is kept as well as the tensor: id() is only unique while
        # the object it names is alive.
        self._by_id[key] = (value, tensor)
        return tensor


class JaxRecorder(BaseRecorder):
    """One recorded execution of a JAX function."""

    framework = JAX
    backend_frameworks = (JAX,)
    annotation_prefix = ANNOTATION_PREFIX

    def __init__(
        self,
        target: Callable,
        example_inputs: Optional[Tuple[Sequence[Any], Dict[str, Any]]] = None,
        *,
        backend: Optional[CostBackend] = None,
        record_dataflow: bool = True,
        plugins: Sequence[Any] = (),
    ) -> None:
        if plugins:
            raise ValueError(
                "plugins need the FX transform to be installed into, and a "
                "jaxpr has none. Whatever the plugin was for has to be part of "
                "the recorder here."
            )
        super().__init__(
            target, example_inputs, backend=backend, record_dataflow=record_dataflow
        )
        self._cache = _TensorCache()
        # (node_id, start_ns, end_ns) per primitive, on the wall clock the
        # trace is anchored against.
        self._windows: List[Tuple[int, int, int]] = []
        self._jaxpr = None
        self._instrument()

    def _instrument(self) -> None:
        # jax is not a dependency of magneton; this module is only imported
        # when a caller names framework="jax".
        import jax

        if self.target is None:
            return
        args, _ = self.example_inputs or ((), {})
        self._jaxpr = jax.make_jaxpr(self.target)(*args)
        self.instrumented = self._run_recording
        if args:
            # Once before measuring, exactly as the torch recorder does.
            self._run_recording(*args, record=False)

    def _run_recording(self, *args, record: bool = True):
        import jax

        # These moved to jax.extend.core at different times, and the old
        # spellings warn. Take each from wherever this jax keeps it.
        try:
            from jax.extend.core import Literal
        except ImportError:  # pragma: no cover - older jax
            Literal = jax.core.Literal
        try:
            from jax.extend.core import DropVar
        except ImportError:
            DropVar = jax.core.DropVar

        closed = self._jaxpr
        jaxpr, consts = closed.jaxpr, closed.consts
        env: Dict[Any, Any] = {}

        def read(var):
            if isinstance(var, Literal):
                return var.val
            return env[var]

        def write(var, val):
            env[var] = val

        for var, val in zip(jaxpr.constvars, consts):
            write(var, val)
        for var, val in zip(jaxpr.invars, args):
            write(var, val)

        for index, eqn in enumerate(jaxpr.eqns):
            invals = [read(v) for v in eqn.invars]
            name = f"{ANNOTATION_PREFIX}{index}_{eqn.primitive.name}"

            started = time.time_ns()
            with jax.profiler.TraceAnnotation(name):
                outvals = eqn.primitive.bind(*invals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                # Waited on so the window below is when this ran, not when it
                # was queued. Nothing downstream can tell those apart.
                jax.block_until_ready(outvals)
            ended = time.time_ns()

            for var, val in zip(eqn.outvars, outvals):
                if not isinstance(var, DropVar):
                    write(var, val)

            if record and self.dataflow_dag is not None:
                self._record(index, name, eqn, invals, outvals)
                self._windows.append((index, started, ended))

        return [read(v) for v in jaxpr.outvars]

    def _record(self, index, name, eqn, invals, outvals) -> None:
        inputs = [t for t in (self._cache.convert(v) for v in invals) if t is not None]
        outputs = [t for t in (self._cache.convert(v) for v in outvals) if t is not None]
        self.dataflow_dag.add_node(
            NodeExecution(
                node_id=index,
                node_name=name,
                op_type="call_function",
                target=str(eqn.primitive.name),
            ),
            inputs,
            outputs,
        )

    def costs(self) -> Dict[str, Cost]:
        """What each recorded operation cost, keyed by its scope name.

        Kernels are assigned to the operation whose window they started in.
        The trace cannot do this itself: it knows an HLO op, and running the
        program one primitive at a time means several operations can compile
        to the same one.
        """
        if self.backend is None or not self._windows:
            return {}
        profiler = getattr(self.backend, "profiler", None)
        if profiler is None:
            return {}

        launches = profiler.kernel_costs()
        windows = sorted(self._windows, key=lambda w: w[1])
        out: Dict[str, Cost] = {}
        for launch in launches:
            for index, start, end in windows:
                if start <= launch.start_ns <= end:
                    node = self.dataflow_dag.nodes.get(index)
                    key = f"{node.node_name}" if node else str(index)
                    cost = out.setdefault(key, Cost())
                    cost.num_calls = 1
                    cost.num_kernels += 1
                    cost.gpu_time_ns += launch.gpu_time_ns
                    cost.gpu_energy_j += launch.gpu_energy_j
                    break
        return out

    def run(self, label: str):
        """This recording as a `compare.Run`.

        The costs are of the program run one operation at a time, which is not
        what the fused program costs -- XLA fuses, and instrumenting every
        primitive is what stops it. The same is true of the torch recorder,
        which runs each graph node on its own, so the two sides of a
        cross-framework comparison are unfused in the same way and remain
        comparable to each other. They are not comparable to a production run's
        totals; take those from an unmodified pass.
        """
        return super().run(label)


    def __enter__(self):
        self._windows.clear()
        return super().__enter__()
