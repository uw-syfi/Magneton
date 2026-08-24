"""Latency from CUDA events, and nothing else."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from magneton.backends.base import Cost
from magneton.plugin import OpPlugin
import torch


class CudaEventTiming:
    """Latency per operation, from CUDA events and nothing else."""

    def __init__(self) -> None:
        self._pairs: Dict[str, list] = {}
        self._running = False

    # --- the backend interface -----------------------------------------------

    def plugins(self) -> Sequence[OpPlugin]:
        return [_EventTimer(self)]

    def start(self) -> None:
        self._pairs.clear()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def cost_by_annotation(self, prefix: str) -> Mapping[str, Cost]:

        torch.cuda.synchronize()
        out: Dict[str, Cost] = {}
        for op_name, pairs in self._pairs.items():
            name = f"forward_{op_name}"
            if not name.startswith(prefix):
                continue
            # elapsed_time is milliseconds; everything else here is nanoseconds.
            total_ns = int(sum(s.elapsed_time(e) for s, e in pairs) * 1e6)
            out[name] = Cost(
                num_calls=len(pairs),
                num_kernels=0,  # this backend cannot see individual kernels
                cpu_time_ns=0,
                gpu_time_ns=total_ns,
                gpu_energy_j=0.0,
            )
        return out

    # --- what the plugin reports back ----------------------------------------

    def _record(self, op_name: str, start, end) -> None:
        self._pairs.setdefault(op_name, []).append((start, end))

    @property
    def measuring(self) -> bool:
        return self._running


class _EventTimer(OpPlugin):
    """Brackets one operation with a CUDA event pair."""

    def __init__(self, sink: CudaEventTiming) -> None:
        self._sink = sink

    @property
    def priority(self) -> int:
        return 20

    def before_execute(self, op_id: int, op_name: str, args: tuple, kwargs: dict) -> dict:
        # wrap_execute is handed only this plugin's own context, not the
        # operation, so the name it will be filed under has to be put here.
        return {"op_name": op_name}

    def after_execute(self, op_id: int, op_name: str, output: Any, context: dict) -> Any:
        return output

    def wrap_execute(self, op_callable: Callable, context: dict) -> Any:

        if not self._sink.measuring or not torch.cuda.is_available():
            return op_callable()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = op_callable()
        end.record()
        self._sink._record(context["op_name"], start, end)
        return out
