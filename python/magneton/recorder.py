"""Recording what a computation did, whichever framework ran it."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from magneton.backends.base import Cost, CostBackend
from magneton.compare import Run
from magneton.dataflow import DataflowDAG

TORCH = "torch"
JAX = "jax"
TF = "tf"


class BaseRecorder:
    """What every recorder does once its framework has been walked."""

    framework = TORCH
    annotation_prefix = ""
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
        """The backend has to be a capture that can see this program's kernels."""
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
        """Write the recorded graph out, and optionally the tensors with it."""
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
    """Record one execution of `target`. See the module docstring."""
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
