"""Profiling a model that uses torch.autograd.Function.

This used to abort in C++ with std::bad_alloc from
torch::autograd::Function::apply, and the cause is worth keeping written down.

The profiler puts its per-thread state into c10::ThreadLocalDebugInfo under
DebugInfoKind::PROFILER_STATE, because that is the slot the CUDA allocator
reads when it reports an allocation. But that slot is torch's own: torch reads
it back through ProfilerStateBase::get(), which static_casts whatever it finds.
The state derived only from c10::MemoryReportingInfoBase -- enough for the
allocator, and the wrong type for that getter.

Nothing noticed until something called it. Ordinary operators never do;
applying an autograd.Function does, and read the object through a vtable that
was not its own.

The fix is to be the type the slot promises. See lib/eprof-torch/include/state.h.

The other way that fix could have gone is quietly: giving the state a slot of
its own would have stopped the crash and stopped the allocator ever calling us.
test_uncovered_paths.py::test_profile_memory_records_allocations is what would
have caught that, and it is why it matters here.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("magneton_eprof")

import magneton  # noqa: E402
from magneton import eprof  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the capture profiles CUDA work"
)


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * x

    @staticmethod
    def backward(ctx, grad):
        return grad


@needs_cuda
def test_an_autograd_function_outside_the_region_is_fine():
    """The boundary: the same call, not profiled."""
    x = torch.randn(8, 8, device="cuda")
    # Constructed but never entered, so no capture is running.
    magneton.record(None, backend=eprof.EnergyBackend(devices=[0]),
                    record_dataflow=False)
    assert torch.allclose(Square.apply(x), x * x)


@needs_cuda
def test_an_autograd_function_can_be_profiled():
    x = torch.randn(8, 8, device="cuda")
    backend = eprof.EnergyBackend(devices=[0])
    with magneton.record(None, backend=backend, record_dataflow=False) as (rec, _):
        out = Square.apply(x)
    torch.cuda.synchronize()
    assert torch.allclose(out, x * x)
    assert rec.backend.per_op_table(), "the region recorded nothing"
