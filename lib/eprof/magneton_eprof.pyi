# Type stub for the eprof extension module.
#
# Types only. Every description lives on the Rust item it describes, in
# src/lib.rs, and PyO3 publishes those as __doc__ -- so `help(eprof._Profiler)`
# already works and duplicating any of it here would only give it somewhere to
# go stale. That is also what PYI021 asks for.
#
# Hand-written, because PyO3 does not emit a stub. tests/python/test_stub.py
# compares it against the real module so it cannot quietly fall behind.

from collections.abc import Iterable, Sequence
from typing import Any

class _ActivityType:
    CPU: _ActivityType
    XPU: _ActivityType
    CUDA: _ActivityType
    MTIA: _ActivityType
    PrivateUse1: _ActivityType

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

class _EnergySampler:
    def __init__(self, device_ids: Sequence[int] | None = None) -> None: ...
    def start(self) -> None: ...
    # (unix_time_ns, device_index, power_mw)
    def stop(self) -> list[tuple[int, int, int]]: ...

class _TreeNode:
    name: str
    tag: str
    start_time_ns: int
    duration_time_ns: int
    correlation_id: int
    children: list[_TreeNode]

class _ProfilerResult:
    def trace_start_ns(self) -> int: ...
    def save(self, path: str) -> None: ...
    def experimental_event_tree(self) -> list[_TreeNode]: ...
    def export_raw_nodes(self) -> list[dict[str, Any]]: ...

class _Profiler:
    def __init__(
        self,
        activities: Iterable[_ActivityType],
        record_shapes: bool,
        with_flops: bool,
        profile_memory: bool,
        with_stack: bool,
        with_modules: bool,
        profile_energy: bool,
        device_ids: Sequence[int],
    ) -> None: ...
    # Accepted and ignored; torch's profiler passes it.
    def start(self, _scopes: object = None) -> None: ...
    def stop(self) -> _ProfilerResult: ...
    def toggle_config(
        self, enable: bool, activities: Sequence[_ActivityType]
    ) -> None: ...

# (name, cpu_duration_ns) -> (start_ns, end_ns, device, correlation_id)
# -> (time_ns, device, power_mw) -> {correlation_id: op name}
# -> [(name, count, cpu_ns, gpu_ns, calls, energy_mj)]
def attribute(
    cpu_ops: Sequence[tuple[str, int]],
    kernels: Sequence[tuple[int, int, int, int]],
    power: Sequence[tuple[int, int, int]],
    corr_to_op: dict[int, str],
    device_index: int | None = None,
) -> list[tuple[str, int, int, int, int, float]]: ...

# (id, tag, start_tid, forward_tid, start_ns, end_ns,
#  flow_id, flow_type, flow_start, linked_id) -> parent id per node
def materialize(
    nodes: Sequence[tuple[int, int, int, int, int, int, int, int, int, int]],
    current_tid: int = 0,
) -> list[int]: ...

# (id, orig_parent_id, tag, flow_id, flow_type, flow_start, linked_id)
# -> parent id per node
def reassign_kineto_parents(
    nodes: Sequence[tuple[int, int, int, int, int, int, int]],
) -> list[int]: ...
