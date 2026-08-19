"""Recording what a computation did, whichever framework ran it.

A recording is three things done to a program: walk it operation by operation,
mark each one so a profiler can find it again, and keep the tensors that went in
and came out. What differs between frameworks is only the walk -- an FX graph, a
jaxpr, a TensorFlow graph -- so that is all a recorder here supplies. The
lifecycle, the costing and the conversion to a `compare.Run` are the same
whichever produced the graph, which is the point: the matcher compares tensors
by value and never learns which framework it is looking at.

    with magneton.record(model, ([x], {})) as (rec, run_it):
        run_it(x)
    run = rec.run("a label")

torch is the default and needs no argument. The others are named explicitly:

    magneton.record(fn, ([x], {}), framework="jax")
    magneton.record(fn, ([x], {}), framework="tf")

Deliberately named rather than detected. Detecting would mean importing jax or
tensorflow to decide what a torch program is, and a wrong guess is silent: the
annotations never match and every operation reports costing nothing. Naming it
costs one argument in the two cases that are not the default, and a machine
with neither installed never touches either.

Compiling and warming up happen at construction, and measuring only begins on
entry. That ordering is not incidental. The dataflow recorder computes a mean
and standard deviation for every tensor it sees, on that node's first
execution, and a run that pays for it inside the measured region is not
comparable to one that does not -- on one comparison that cost was 77% of the
total and made the faster system look eight times slower.

It also means `instrumented` exists before the `with`, which matters when the
model has to be installed somewhere first. vLLM holds its own reference to the
module it runs, so the instrumented one has to be put there and exercised once
before the region opens.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from magneton.backends.base import Cost, CostBackend
from magneton.compare import Run
from magneton.dataflow import DataflowDAG

TORCH = "torch"
JAX = "jax"
TF = "tf"


class BaseRecorder:
    """What every recorder does once its framework has been walked.

    Subclasses supply `framework`, `annotation_prefix`, `_instrument`, and
    `costs`; everything below is shared.
    """

    framework = TORCH
    annotation_prefix = ""
    # Which capture can measure this. Usually the matching one, but the two are
    # different questions: `framework` is whose graph is walked, and this is
    # whose profiler can see the kernels that come out of it.
    backend_frameworks: Tuple[str, ...] = (TORCH,)

    def __init__(
        self,
        target: Any,
        example_inputs: Optional[Tuple[Sequence[Any], Dict[str, Any]]] = None,
        *,
        backend: Optional[CostBackend] = None,
        record_dataflow: bool = True,
    ) -> None:
        self.target = target
        self.example_inputs = example_inputs
        self.backend = backend
        self.instrumented: Optional[Callable] = None
        self.dataflow_dag: Optional[DataflowDAG] = (
            DataflowDAG() if record_dataflow else None
        )
        self._check_backend()

    def _check_backend(self) -> None:
        """The backend has to be a capture that can see this program's kernels.

        Not always the matching one. libkineto reads CUPTI, and CUPTI reports
        every kernel in the process whoever launched it, so a torch capture
        measures a TensorFlow program perfectly well -- the recorder attributes
        by time window rather than by operator, which is what makes that
        usable. JAX is the exception, because its profiler is a different
        capture entirely.

        The failure this prevents is silent. A backend that cannot see the
        program profiles nothing it recognises and every operation comes back
        costing zero, which is indistinguishable from a real measurement of a
        program that did no work.
        """
        if self.backend is None:
            return
        theirs = getattr(self.backend, "framework", TORCH)
        if theirs not in self.backend_frameworks:
            wanted = " or ".join(repr(f) for f in self.backend_frameworks)
            raise ValueError(
                f"this records a {self.framework} program, which needs a "
                f"{wanted} capture; this backend measures {theirs!r}. Nothing "
                f"would fail: it would see none of the kernels and every "
                f"operation would report costing nothing."
            )

    # --- what a subclass provides --------------------------------------------

    def _instrument(self) -> None:
        raise NotImplementedError

    def costs(self) -> Dict[str, Cost]:
        raise NotImplementedError

    # --- the measured region -------------------------------------------------

    def __enter__(self) -> Tuple["BaseRecorder", Optional[Callable]]:
        if self.backend is not None:
            self.backend.start()
        return self, self.instrumented

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.backend is not None:
            self.backend.stop()

    # --- what it recorded ----------------------------------------------------

    def export_dataflow(self, json_path: str, tensor_path: Optional[str] = None) -> None:
        """Write the recorded graph out, and optionally the tensors with it.

        Without `tensor_path` this is the structure alone -- readable, and
        enough to see what ran, but not enough to match against another
        recording, which compares tensor values.
        """
        if self.dataflow_dag is None:
            raise RuntimeError("nothing was recorded: record_dataflow was off")
        self.dataflow_dag.save(json_path, tensor_path)

    def run(self, label: str) -> Run:
        """This recording as a `compare.Run`, ready to match against another."""
        if self.dataflow_dag is None:
            raise RuntimeError(
                f"{label}: nothing was recorded. Construct the recorder with "
                "record_dataflow=True to compare this run against another."
            )
        self.dataflow_dag.build_edges()
        return Run(label=label, dag=self.dataflow_dag, per_node=self.costs())


def record(
    target: Any,
    example_inputs: Optional[Tuple[Sequence[Any], Dict[str, Any]]] = None,
    *,
    framework: str = TORCH,
    backend: Optional[CostBackend] = None,
    record_dataflow: bool = True,
    **kwargs: Any,
) -> BaseRecorder:
    """Record one execution of `target`. See the module docstring.

    `framework` selects the walk: "torch" (the default), "jax", or "tf". The
    module for the one named is the only one imported.
    """
    if framework == TORCH:
        from magneton.recorders.fx import TorchRecorder as R
    elif framework == JAX:
        from magneton.recorders.jaxpr import JaxRecorder as R
    elif framework == TF:
        from magneton.recorders.tfgraph import TfRecorder as R
    else:
        raise ValueError(
            f"framework must be one of {TORCH!r}, {JAX!r}, {TF!r}, "
            f"not {framework!r}"
        )
    return R(
        target,
        example_inputs,
        backend=backend,
        record_dataflow=record_dataflow,
        **kwargs,
    )
