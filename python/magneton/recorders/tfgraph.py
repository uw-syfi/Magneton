"""Recording a TensorFlow program by walking the graph a tf.function builds."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from magneton.backends.base import Cost, CostBackend
from magneton.dataflow import NodeExecution
from magneton.recorder import TF, BaseRecorder

ANNOTATION_PREFIX = "tf_op_"


class _TensorCache:
    """One torch view per TF tensor, so node identity survives conversion."""

    def __init__(self) -> None:
        self._by_id: Dict[int, Tuple[Any, Any]] = {}

    def convert(self, value: Any):
        import tensorflow as tf

        if not isinstance(value, tf.Tensor):
            return None
        key = id(value)
        hit = self._by_id.get(key)
        if hit is not None:
            return hit[1]
        try:
            tensor = torch.utils.dlpack.from_dlpack(
                tf.experimental.dlpack.to_dlpack(value)
            )
        except Exception:
            # bool and a few other dtypes dlpack will not carry; a copy is
            # better than a dropped tensor, which would be a dropped edge.
            tensor = torch.as_tensor(value.numpy())
        self._by_id[key] = (value, tensor)
        return tensor


class TfRecorder(BaseRecorder):
    """One recorded execution of a TensorFlow function."""

    framework = TF
    backend_frameworks = ("torch",)
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
                "TensorFlow graph has none."
            )
        super().__init__(
            target, example_inputs, backend=backend, record_dataflow=record_dataflow
        )
        self._cache = _TensorCache()
        self._windows: List[Tuple[int, int, int]] = []
        self._graph = None
        self._instrument()

    def _instrument(self) -> None:
        import tensorflow as tf

        if self.target is None:
            return
        args, _ = self.example_inputs or ((), {})
        specs = [tf.TensorSpec(a.shape, a.dtype) for a in args]
        self._graph = tf.function(self.target).get_concrete_function(*specs).graph
        self.instrumented = self._run_recording
        if args:
            self._run_recording(*args, record=False)

    def _run_recording(self, *args, record: bool = True):
        import tensorflow as tf

        env: Dict[str, Any] = {}
        placeholders = [o for o in self._graph.get_operations()
                        if o.type == "Placeholder"]
        for op, value in zip(placeholders, args):
            env[op.outputs[0].name] = value

        index = 0
        outputs: List[Any] = []
        for op in self._graph.get_operations():
            if op.type == "Placeholder":
                continue
            if op.type == "Const":
                # Its value is an attribute, not something to execute.
                env[op.outputs[0].name] = tf.constant(
                    tf.make_ndarray(op.get_attr("value"))
                )
                continue

            raw = getattr(tf.raw_ops, op.type, None)
            if raw is None:
                raise RuntimeError(
                    f"no tf.raw_ops entry for {op.type!r}, so this graph cannot "
                    f"be run one operation at a time. The recorder needs one "
                    f"per operation it meets."
                )
            kwargs = self._arguments(op, raw, env)

            started = time.time_ns()
            result = raw(**kwargs)
            values = list(result) if isinstance(result, (list, tuple)) else [result]
            # TF is synchronous per op in eager mode, so the window below is
            # already when this ran rather than when it was queued.
            ended = time.time_ns()

            for tensor, value in zip(op.outputs, values):
                env[tensor.name] = value

            if record and self.dataflow_dag is not None:
                name = f"{ANNOTATION_PREFIX}{index}_{op.type}"
                self._record(index, name, op, [env[t.name] for t in op.inputs], values)
                self._windows.append((index, started, ended))
            index += 1
            outputs = values

        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def _arguments(op, raw, env) -> Dict[str, Any]:
        """What to hand `tf.raw_ops.<Type>` for this graph operation."""
        params = [n for n in inspect.signature(raw).parameters if n != "name"]
        kwargs = {n: env[t.name] for n, t in zip(params, op.inputs)}
        from_inputs = {a.type_attr for a in op.op_def.input_arg if a.type_attr}
        for key in op.node_def.attr:
            if key.startswith("_") or key in from_inputs:
                continue
            kwargs[key] = op.get_attr(key)
        return kwargs

    def _record(self, index, name, op, invals, outvals) -> None:
        inputs = [t for t in (self._cache.convert(v) for v in invals) if t is not None]
        outputs = [t for t in (self._cache.convert(v) for v in outvals) if t is not None]
        self.dataflow_dag.add_node(
            NodeExecution(
                node_id=index,
                node_name=name,
                op_type="call_function",
                target=op.type,
            ),
            inputs,
            outputs,
        )

    def __enter__(self):
        self._windows.clear()
        return super().__enter__()

    def costs(self) -> Dict[str, Cost]:
        """What each recorded operation cost, by the window it ran in."""
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
                    key = node.node_name if node else str(index)
                    cost = out.setdefault(key, Cost())
                    cost.num_calls = 1
                    cost.num_kernels += 1
                    cost.gpu_time_ns += launch.gpu_time_ns
                    cost.gpu_energy_j += launch.gpu_energy_j
                    break
        return out
