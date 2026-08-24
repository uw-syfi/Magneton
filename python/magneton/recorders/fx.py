"""Recording a torch program by walking the FX graph dynamo produces."""

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
    """One recorded execution of a torch model."""

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
        """Compile the model, and run it once if given something to run."""
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
        """What each graph node cost, keyed by its annotation scope."""
        if self.backend is None:
            return {}
        return dict(self.backend.cost_by_annotation(self.annotation_prefix))
