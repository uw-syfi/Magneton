"""Tests for the R3 event-tree diff harness (eprof.treediff).

The harness must be trustworthy *before* a Rust reconstruction exists, so most
of these validate the harness mechanics with synthetic duck-typed nodes (no GPU
needed). The last test confirms a real C++ tree canonicalizes self-consistently
and that perturbations are caught.
"""


import pytest

from magneton.eprof import treediff


class FakeNode:
    """Minimal duck-typed node matching what canonicalize() reads."""

    def __init__(self, name, tag, start_ns, dur_ns, corr, children=None):
        self.name = name
        self.tag = tag
        self.start_time_ns = start_ns
        self.duration_time_ns = dur_ns
        self.correlation_id = corr
        self.children = children or []


def _sample_tree(child_order="ab"):
    a = FakeNode("aten::mm", "_EventType.TorchOp", 100, 50, 1)
    k = FakeNode("gemm", "_EventType.Kineto", 110, 30, 1)
    a.children = [k]
    b = FakeNode("aten::add", "_EventType.TorchOp", 200, 20, 2)
    roots = [a, b] if child_order == "ab" else [b, a]
    return roots


def test_canonicalize_order_insensitive():
    """Different sibling/root order -> identical canonical form."""
    assert treediff.trees_equal(_sample_tree("ab"), _sample_tree("ba"))


def test_diff_empty_for_identical():
    assert treediff.diff(
        treediff.canonicalize(_sample_tree()), treediff.canonicalize(_sample_tree())
    ) == []


def test_diff_catches_field_change():
    ref = treediff.canonicalize(_sample_tree())
    cand_tree = _sample_tree()
    cand_tree[0].children[0].duration_time_ns = 999  # perturb a kernel duration
    cand = treediff.canonicalize(cand_tree)
    msgs = treediff.diff(ref, cand)
    assert msgs and any("999" in m for m in msgs)


def test_diff_catches_missing_node():
    ref = treediff.canonicalize(_sample_tree())
    cand_tree = _sample_tree()
    cand_tree[0].children = []  # drop the kernel
    cand = treediff.canonicalize(cand_tree)
    msgs = treediff.diff(ref, cand)
    assert any("node count differs" in m for m in msgs)


def test_summarize():
    s = treediff.summarize(_sample_tree())
    assert s["total"] == 3
    assert s["by_tag"]["TorchOp"] == 2
    assert s["by_tag"]["Kineto"] == 1
    assert s["max_depth"] == 1


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for a real capture")
def test_real_cpp_tree_self_consistent():
    import torch

    import magneton_eprof as C

    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        True, False, False, False, False, True, [0],
    )
    prof.start(set())
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    res = prof.stop()
    roots = res.experimental_event_tree()

    ref = treediff.canonicalize(roots)
    assert len(ref) > 0
    # Canonicalization is deterministic: same roots -> identical twice.
    assert treediff.diff(ref, treediff.canonicalize(roots)) == []
    # And the summary is sane (has TorchOp + Kineto nodes).
    s = treediff.summarize(roots)
    assert s["by_tag"].get("TorchOp", 0) > 0
    assert s["by_tag"].get("Kineto", 0) > 0


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for a real capture")
def test_raw_pod_export_roundtrips_to_cpp_tree():
    """The flat POD bridge (export_raw_nodes) must rebuild the same tree.

    NOTE: this was originally an independence check -- export_raw_nodes encoded
    nesting as id/parent_id while export_tree walked the C++ children_ pointers,
    so agreeing meant the encoding was right. The shared_ptr tree no longer
    survives collection (ProfilerResult holds pre-order FlatNodes), so both
    paths now derive from the same array and this can no longer catch an
    encoding bug. It still exercises the schema and the diff harness; the
    golden oracle in test_golden.py is what guards the tree shape.
    """
    import torch

    import magneton_eprof as C

    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        True, False, False, False, False, True, [0],
    )
    prof.start(set())
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    for _ in range(15):
        _ = a @ b
    torch.cuda.synchronize()
    res = prof.stop()

    raw = res.export_raw_nodes()
    rebuilt = treediff.tree_from_raw(raw)
    cpp = res.experimental_event_tree()
    assert treediff.diff(treediff.canonicalize(cpp), treediff.canonicalize(rebuilt)) == []

    # linked_id, where present, must point at a real node (unambiguous target).
    ids = {d["id"] for d in raw}
    for d in raw:
        if d["linked_id"] >= 0:
            assert d["linked_id"] in ids


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for a real capture")
def test_rust_flow_linked_merge_matches_cpp():
    """The Rust kineto flow/linked merge (R3) reproduces C++ parents exactly.

    This covers the setParents stage: every Kineto node whose parent comes from
    a flow-start match or a linked activity must get the same parent in Rust as
    in C++. The remaining containment stage (build_tree) for orphan nodes is a
    separate port; here we assert the flow/linked subset is exact.
    """
    magneton_eprof = pytest.importorskip("magneton_eprof")
    import torch

    import magneton_eprof as C

    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        True, False, False, False, False, True, [0],
    )
    prof.start(set())
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    for _ in range(20):
        c = a @ b
        d = torch.relu(c)
        _ = d + a
    torch.cuda.synchronize()
    res = prof.stop()

    raw = res.export_raw_nodes()
    tuples = [
        (d["id"], d["parent_id"], d["tag"], d["flow_id"], d["flow_type"],
         d["flow_start"], d["linked_id"])
        for d in raw
    ]
    new_parents = magneton_eprof.reassign_kineto_parents(tuples)

    KINETO = 6
    K_ASYNC = 2
    flow_starts = {
        d["flow_id"]
        for d in raw
        if d["tag"] == KINETO and d["flow_type"] == K_ASYNC and d["flow_start"] == 1
    }
    matched = 0
    for d, parent in zip(raw, new_parents):
        if d["tag"] != KINETO:
            continue
        has_flow = (
            d["flow_type"] == K_ASYNC
            and d["flow_start"] == 0
            and d["flow_id"] in flow_starts
        )
        has_link = d["linked_id"] >= 0
        if has_flow or has_link:
            matched += 1
            assert parent == d["parent_id"], (
                f"{d['name']}: rust parent {parent} != C++ {d['parent_id']}"
            )
    assert matched > 0


def _full_materialize_matches(res) -> list:
    """Run the full Rust materialization from scratch and diff vs the C++ tree."""
    import magneton_eprof

    from magneton.eprof import treediff

    raw = res.export_raw_nodes()
    tuples = [
        (d["id"], d["tag"], d["start_tid"], d["forward_tid"], d["start_ns"],
         d["start_ns"] + d["dur_ns"], d["flow_id"], d["flow_type"],
         d["flow_start"], d["linked_id"])
        for d in raw
    ]
    current_tid = next((d["start_tid"] for d in raw if d["tag"] == 0), 0)
    parents = magneton_eprof.materialize(tuples, current_tid)
    rust_tree = treediff.tree_from_raw(
        [dict(d, parent_id=p) for d, p in zip(raw, parents)]
    )
    cpp_tree = res.experimental_event_tree()
    return treediff.diff(
        treediff.canonicalize(cpp_tree), treediff.canonicalize(rust_tree)
    )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for a real capture")
def test_rust_full_materialize_matches_cpp():
    """Rust materialize() (flow/linked + containment) rebuilds the exact C++ tree.

    Ignores the C++ parents entirely and reconstructs the whole tree from the
    raw POD, then diffs node-for-node. Exercised across several workloads,
    including a backward pass (multi-thread / fwd_tid path).
    """
    pytest.importorskip("magneton_eprof")
    import torch

    import magneton_eprof as C

    def run(fn):
        prof = C._Profiler(
            {C._ActivityType.CPU, C._ActivityType.CUDA},
            True, False, False, False, False, True, [0],
        )
        prof.start(set())
        fn()
        torch.cuda.synchronize()
        return prof.stop()

    def matmuls():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        for _ in range(20):
            c = a @ b
            d = torch.relu(c)
            _ = d + a

    def with_backward():
        m = torch.nn.Linear(512, 512).cuda()
        x = torch.randn(64, 512, device="cuda", requires_grad=True)
        for _ in range(10):
            y = m(x).relu().sum()
            y.backward()

    for workload in (matmuls, with_backward):
        res = run(workload)
        diff = _full_materialize_matches(res)
        assert diff == [], f"{workload.__name__}: {diff[:6]}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for a real capture")
def test_rust_materialize_recomputes_kineto_tids():
    """Rust materialize must recompute kineto tids itself (runtime condition).

    At runtime, kineto nodes have no tid until setKinetoTID runs. Simulate that
    by zeroing kineto start_tids before materializing; the Rust port must still
    rebuild the exact C++ tree, proving its setKinetoTID is correct (the export
    path otherwise hands it already-correct tids).
    """
    pytest.importorskip("magneton_eprof")
    import magneton_eprof
    import torch

    import magneton_eprof as C
    from magneton.eprof import treediff

    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        True, False, False, False, False, True, [0],
    )
    prof.start(set())
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    for _ in range(15):
        _ = torch.relu(a @ b)
    torch.cuda.synchronize()
    res = prof.stop()

    raw = res.export_raw_nodes()
    current_tid = next((d["start_tid"] for d in raw if d["tag"] == 0), 0)
    KINETO = 6
    # Zero kineto tids to force recomputation (runtime has them unset).
    tuples = [
        (d["id"], d["tag"], (0 if d["tag"] == KINETO else d["start_tid"]),
         d["forward_tid"], d["start_ns"], d["start_ns"] + d["dur_ns"],
         d["flow_id"], d["flow_type"], d["flow_start"], d["linked_id"])
        for d in raw
    ]
    parents = magneton_eprof.materialize(tuples, current_tid)
    rust_tree = treediff.tree_from_raw(
        [dict(d, parent_id=p) for d, p in zip(raw, parents)]
    )
    cpp_tree = res.experimental_event_tree()
    diff = treediff.diff(
        treediff.canonicalize(cpp_tree), treediff.canonicalize(rust_tree)
    )
    assert diff == [], diff[:6]
