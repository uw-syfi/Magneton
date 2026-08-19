"""Recording a torch program by walking the FX graph dynamo produces.

Instrumentation is `magneton.transform.pluggable_pass`, installed as the
backend `torch.compile` hands its graph to. Each node becomes a submodule call
that opens a `record_function` scope and runs the plugins, one of which records
the dataflow.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from magneton import transform
from magneton.backends.base import Cost, CostBackend
from magneton.config import DataflowConfig
from magneton.dataflow import DataflowPlugin
from magneton.plugin import OpPlugin
from magneton.recorder import TORCH, BaseRecorder


class TorchRecorder(BaseRecorder):
    """One recorded execution of a torch model.

    Args:
        target: the module to instrument
        example_inputs: `(args, kwargs)` to compile and warm up with, before
            measurement starts. Strongly recommended -- without it the first
            call inside the measured region pays for compilation and for every
            tensor statistic the recorder computes.
        backend: what measures the run, or None to record structure only
        clone_outputs: keep a copy of each operation's output rather than a
            reference. Needed when the model writes through its own tensors,
            which would otherwise leave the recording holding the final value
            of a buffer rather than what that operation produced.
        plugins: further instrumentation to install
        dataflow_config: the same two settings as one object, when it is more
            convenient to pass around. Overrides them if given.
    """

    framework = TORCH
    annotation_prefix = transform.ANNOTATION_PREFIX

    def __init__(
        self,
        target: Any,
        example_inputs: Optional[Tuple[Sequence[Any], Dict[str, Any]]] = None,
        *,
        backend: Optional[CostBackend] = None,
        clone_outputs: bool = False,
        plugins: Sequence[OpPlugin] = (),
        record_dataflow: bool = True,
        dataflow_config: Optional[DataflowConfig] = None,
    ) -> None:
        if dataflow_config is not None:
            record_dataflow = dataflow_config.record_dataflow
            clone_outputs = dataflow_config.clone_outputs
        super().__init__(
            target, example_inputs, backend=backend, record_dataflow=record_dataflow
        )
        # The model, under its historical name: enough callers reach for
        # `.compiled_model` that renaming it would be churn for its own sake.
        self.model = target

        installed: List[OpPlugin] = []
        if self.dataflow_dag is not None:
            installed.append(
                DataflowPlugin(dag=self.dataflow_dag, clone_outputs=clone_outputs)
            )
        if backend is not None:
            installed.extend(backend.plugins())
        installed.extend(plugins)
        self._plugins = installed
        self._instrument()

    @property
    def compiled_model(self) -> Optional[Callable]:
        return self.instrumented

    def _instrument(self) -> None:
        """Compile the model, and run it once if given something to run.

        Both belong outside the measured region: compilation is not the
        workload, and the dataflow recorder's per-tensor statistics are
        computed on a node's first execution.
        """
        if self.target is None:
            return

        def compile_backend(gm, *args, **kwargs):
            if self._plugins:
                return transform.pluggable_pass(gm, self._plugins, *args, **kwargs)
            return gm

        self.instrumented = torch.compile(
            self.target, fullgraph=True, backend=compile_backend
        )
        if self.example_inputs is not None:
            args, kwargs = self.example_inputs
            self.instrumented(*args, **kwargs)

    def costs(self) -> Dict[str, Cost]:
        """What each graph node cost, keyed by its annotation scope.

        Empty without a backend. Read this before another run starts: a backend
        is free to hold onto process-wide state that the next capture will
        reuse, and eprof's does.
        """
        if self.backend is None:
            return {}
        return dict(self.backend.cost_by_annotation(self.annotation_prefix))
