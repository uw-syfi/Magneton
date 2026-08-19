"""Gathering a JAX run's kernels under the operations that ran them.

XLA names a kernel it generates after the fused operation it came from, so for
those the operation and the kernel are the same string. An operation that
lowers to a library call is not: a convolution runs as a cuDNN gemm and two
layout conversions, none of which is named for anything in the program. The
operation is the level the program was written at.

These build the kernel list by hand rather than running JAX, so they check the
gathering and the arithmetic rather than the trace parser.
"""

import pytest

pytest.importorskip("magneton_eprof")

from magneton.eprof.profiler import JaxKernel, _JaxBackend  # noqa: E402

MS = 1_000_000


def backend(kernels, watts=100.0, span_ns=10 * MS):
    """A JAX backend holding `kernels`, with flat board power over the run."""
    impl = _JaxBackend(devices=[0], trace_dir=None)
    impl.kernels = kernels
    # Two readings is the minimum that can be integrated between.
    impl.power = [(0, 0, int(watts * 1000)), (span_ns, 0, int(watts * 1000))]
    return impl


def kernel(name, start_ms, end_ms, hlo_op="", module="m"):
    return JaxKernel(
        name=name, start_ns=start_ms * MS, end_ns=end_ms * MS,
        device=0, hlo_op=hlo_op, hlo_module=module,
    )


def test_an_operations_kernels_are_gathered_under_it():
    ops = backend([
        kernel("sm80_xmma_fprop", 0, 2, hlo_op="cudnn-conv.1.0"),
        kernel("nchwToNhwcKernel", 2, 3, hlo_op="cudnn-conv.1.0"),
        kernel("nchwToNhwcKernel", 3, 4, hlo_op="cudnn-conv.1.0"),
        kernel("loop_add_fusion", 4, 5, hlo_op="loop_add_fusion"),
    ]).per_hlo_op(None)

    assert [o.name for o in ops] == ["cudnn-conv.1.0", "loop_add_fusion"]

    conv = ops[0]
    assert conv.num_launches == 3, "three launches belong to the convolution"
    assert {k.name for k in conv.kernels} == {"sm80_xmma_fprop", "nchwToNhwcKernel"}
    # The two layout conversions share a name and so share a row.
    layout = next(k for k in conv.kernels if k.name == "nchwToNhwcKernel")
    assert layout.num_launches == 2


def test_the_kernels_account_for_the_operation():
    ops = backend([
        kernel("gemm", 0, 3, hlo_op="conv.1"),
        kernel("layout", 3, 4, hlo_op="conv.1"),
    ]).per_hlo_op(None)

    op = ops[0]
    assert sum(k.gpu_time_ns for k in op.kernels) == op.gpu_time_ns
    assert sum(k.gpu_energy_j for k in op.kernels) == pytest.approx(op.gpu_energy_j)
    assert op.gpu_time_ns == 4 * MS


def test_energy_is_split_per_launch_before_it_is_summed():
    """Two kernels running at once share the board between them.

    This is why the gathering happens after attribution rather than before: an
    operation's energy is the sum of what each of its launches was charged,
    and what a launch is charged depends on what else was running.
    """
    alone = backend([kernel("k", 0, 2, hlo_op="op")]).per_hlo_op(None)
    together = backend([
        kernel("k", 0, 2, hlo_op="op"),
        kernel("other", 0, 2, hlo_op="elsewhere"),
    ]).per_hlo_op(None)

    solo = alone[0].gpu_energy_j
    shared = next(o for o in together if o.name == "op").gpu_energy_j
    assert shared == pytest.approx(solo / 2, rel=1e-6), (
        "a kernel sharing the board with one other should be charged half"
    )


def test_a_kernel_the_trace_did_not_attribute_is_its_own_operation():
    """Better to report it under its kernel name than to drop it."""
    ops = backend([kernel("mystery_kernel", 0, 1, hlo_op="")]).per_hlo_op(None)
    assert [o.name for o in ops] == ["mystery_kernel"]
    assert ops[0].gpu_energy_j > 0


def test_operations_come_back_largest_energy_first():
    ops = backend([
        kernel("small", 0, 1, hlo_op="small_op"),
        kernel("big", 1, 9, hlo_op="big_op"),
    ]).per_hlo_op(None)
    assert [o.name for o in ops] == ["big_op", "small_op"]


def test_the_torch_backend_has_no_hlo_operations():
    from magneton.eprof import Profiler

    prof = Profiler()
    with pytest.raises(ValueError, match="jax notion"):
        prof.per_hlo_op()
