"""The paths the goldens do not reach.

The golden workloads run with profile_memory off, on one thread, and never
toggle. Each of those is a code path with its own C++/Rust boundary, and at
least one of them (the allocation reports) has already been broken once by a
change the goldens happened to catch for the wrong reason -- they noticed
events appearing that should not have, and would not have noticed events
failing to appear.
"""

import collections
import pytest
import torch

pytest.importorskip("magneton_eprof")
import magneton_eprof as C  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

ACTIVITIES = None
TAG_ALLOCATION = 2
TAG_TORCH_OP = 0


def _profiler(*, record_shapes=False, profile_memory=False, with_stack=False):
    return C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        record_shapes,
        False,  # with_flops
        profile_memory,
        with_stack,
        False,  # with_modules
        False,  # profile_energy
        [],
    )


def test_profile_memory_records_allocations():
    """With profile_memory on, allocations reach the trace."""
    prof = _profiler(profile_memory=True)
    prof.start(set())
    xs = [torch.randn(256, 256, device="cuda") for _ in range(8)]
    del xs
    torch.cuda.synchronize()
    raw = prof.stop().export_raw_nodes()

    allocations = [d for d in raw if d["tag"] == TAG_ALLOCATION]
    assert allocations, "profile_memory=True recorded no allocation events"
    # An allocation names the device it happened on, unlike an op.
    assert any(d["device_type"] == 1 for d in allocations), (
        "no allocation was attributed to the GPU"
    )


def test_profile_memory_off_records_none():
    """And with it off, none do -- the guard runs in two places."""
    prof = _profiler(profile_memory=False)
    prof.start(set())
    xs = [torch.randn(256, 256, device="cuda") for _ in range(8)]
    del xs
    torch.cuda.synchronize()
    raw = prof.stop().export_raw_nodes()

    assert not [d for d in raw if d["tag"] == TAG_ALLOCATION]


def test_backward_ops_land_on_the_autograd_threads_subqueue():
    """The real multi-thread path.

    Collection registers an at::addThreadLocalCallback, so a thread the user
    spawns is deliberately not instrumented. What does cross threads is
    autograd: the engine propagates at::ThreadLocalState to its device threads,
    which is why an op carries a forward thread id at all. Each of those
    threads takes its own subqueue, through the thread-local cache.
    """
    prof = _profiler(record_shapes=True)
    prof.start(set())
    a = torch.randn(64, 64, device="cuda", requires_grad=True)
    ((a @ a).relu().sum()).backward()
    torch.cuda.synchronize()
    raw = prof.stop().export_raw_nodes()

    ops = [d for d in raw if d["tag"] == TAG_TORCH_OP]
    assert ops, "no ops captured"

    backward = [d for d in ops if d["forward_tid"]]
    assert backward, "no backward op recorded a forward thread id"
    tids = {d["start_tid"] for d in ops}
    assert len(tids) >= 2, (
        f"forward and backward both ran on tid(s) {tids}; expected the autograd "
        "engine to use its own"
    )

    # Correlation ids come from one global counter, so two subqueues filling
    # concurrently must never hand out the same one.
    corrs = [d["correlation_id"] for d in ops if d["correlation_id"]]
    dupes = [c for c, n in collections.Counter(corrs).items() if n > 1]
    assert not dupes, f"correlation ids reused across threads: {dupes[:5]}"


def test_toggling_cpu_collection_stops_and_resumes_it():
    prof = _profiler()
    prof.start(set())
    a = torch.randn(32, 32, device="cuda")
    (a @ a).relu()

    prof.toggle_config(False, [C._ActivityType.CPU])
    for _ in range(5):
        (a @ a).relu()
    prof.toggle_config(True, [C._ActivityType.CPU])
    (a @ a).relu()

    torch.cuda.synchronize()
    raw = prof.stop().export_raw_nodes()
    assert [d for d in raw if d["tag"] == TAG_TORCH_OP], (
        "toggling collection off and on again lost every op"
    )
