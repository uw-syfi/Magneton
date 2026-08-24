"""The topology-aware subgraph matcher."""

import pytest

torch = pytest.importorskip("torch")

from magneton.matching.matcher import (  # noqa: E402
    MatchConfig,
    find_cut_points,
    match_graphs,
    recursive_match,
)
from magneton.matching.dominator import DominatorTree  # noqa: E402
from magneton.dataflow import DataflowDAG, NodeExecution  # noqa: E402


def dag(spec) -> DataflowDAG:
    """Builds a DAG from `(node_id, [input tensor ids], [output tensor ids])`."""
    d = DataflowDAG()
    pool = {}
    for node_id, inputs, outputs in spec:
        for tid in list(inputs) + list(outputs):
            if tid not in pool:
                t = torch.full((2, 2), float(tid))
                pool[tid] = t
                d.tensors[tid] = t
                d.tensor_metadata[tid] = {
                    "shape": [2, 2],
                    "dtype": "torch.float32",
                    "device": "cpu",
                    "mean": float(tid),
                    "std": 0.0,
                }
        d.nodes[node_id] = NodeExecution(
            node_id=node_id,
            node_name=f"op_{node_id}",
            op_type="call_function",
            target=f"aten.op_{node_id}",
            input_tensor_ids=list(inputs),
            output_tensor_ids=list(outputs),
        )
    return d


# --- the spine ---------------------------------------------------------------


def test_a_straight_line_is_its_own_dominator_path():
    d = dag([(0, [], [10]), (1, [10], [11]), (2, [11], [12])])
    assert DominatorTree(d).dominator_path == [0, 1, 2]


def test_a_branch_is_not_on_the_dominator_path():
    # 0 -> {1, 2} -> 3. Neither branch is on every path; 0 and 3 are.
    d = dag([(0, [], [10]), (1, [10], [11]), (2, [10], [12]), (3, [11, 12], [13])])
    path = DominatorTree(d).dominator_path
    assert path == [0, 3], f"a branch cannot dominate the sink: {path}"


def test_the_spine_does_not_depend_on_node_numbering():
    """A subgraph keeps the ids the full."""
    d = dag([(7, [], [10]), (8, [10], [11]), (9, [11], [12])])
    assert DominatorTree(d).dominator_path == [7, 8, 9]


# --- cut points --------------------------------------------------------------


def test_cut_points_advance_along_both_paths():
    # Equivalences that would go backwards in path2 if taken greedily.
    cuts = find_cut_points([1, 2, 3], [11, 12, 13], [(1, 12), (2, 11), (3, 13)])
    assert cuts == [(1, 12), (3, 13)], (
        "taking (2, 11) after (1, 12) would run the second graph backwards"
    )


def test_cut_points_prefer_the_earliest_partner():
    cuts = find_cut_points([1, 2], [11, 12, 13], [(1, 11), (1, 13), (2, 12)])
    assert cuts == [(1, 11), (2, 12)]


# --- the recursion -----------------------------------------------------------


def two_chains():
    """Two chains computing the same thing, agreeing at three points."""
    g1 = dag([(0, [10], [11]), (1, [11], [15]), (2, [15], [16]),
              (3, [16], [20]), (4, [20], [25])])
    g2 = dag([(0, [10], [15]), (1, [15], [20]), (2, [20], [25])])
    return g1, g2


# The node pairs of two_chains whose outputs agree.
TWO_CHAIN_EQUIV = [(1, 0), (3, 1), (4, 2)]


def test_a_pair_with_one_cut_is_already_minimal():
    g1 = dag([(0, [10], [11]), (1, [11], [20])])
    g2 = dag([(0, [10], [20])])
    equiv = [(1, 0)]  # both produce tensor 20
    matches = recursive_match(g1, g2, equiv, MatchConfig(), 0)
    assert len(matches) == 1
    assert matches[0].graph1_nodes == {0, 1}
    assert matches[0].graph2_nodes == {0}


def test_two_cuts_bound_a_single_region():
    """Two agreements leave one gap, so one match -- not one per cut."""
    g1 = dag([(0, [10], [11]), (1, [11], [15]), (2, [15], [16]), (3, [16], [20])])
    g2 = dag([(0, [10], [15]), (1, [15], [20])])
    matches = recursive_match(g1, g2, [(1, 0), (3, 1)], MatchConfig(), 0)
    assert len(matches) == 1
    assert matches[0].graph1_nodes == {1, 2, 3}
    assert matches[0].graph2_nodes == {0, 1}


def test_three_cuts_divide_the_pair_in_two():
    g1, g2 = two_chains()
    matches = recursive_match(g1, g2, TWO_CHAIN_EQUIV, MatchConfig(), 0)

    assert len(matches) == 2, f"n cuts bound n-1 regions, got {len(matches)}"
    first, second = sorted(matches, key=lambda m: min(m.graph1_nodes))
    assert first.graph1_nodes == {1, 2, 3} and first.graph2_nodes == {0, 1}
    assert second.graph1_nodes == {3, 4} and second.graph2_nodes == {1, 2}


def test_the_regions_are_smaller_than_what_they_came_from():
    """The point of dividing: each reported pair is a piece, not the whole."""
    g1, g2 = two_chains()
    matches = recursive_match(g1, g2, TWO_CHAIN_EQUIV, MatchConfig(), 0)
    assert all(len(m.graph1_nodes) < len(g1.nodes) for m in matches)
    assert all(len(m.graph2_nodes) < len(g2.nodes) for m in matches)


def test_no_equivalence_means_no_match():
    g1 = dag([(0, [10], [11]), (1, [11], [12])])
    g2 = dag([(0, [30], [31]), (1, [31], [32])])
    assert recursive_match(g1, g2, [], MatchConfig(), 0) == []


def test_recursion_is_bounded_when_the_cuts_stop_dividing():
    """Two cuts that bracket the whole graph divide nothing."""
    g1 = dag([(0, [10], [11]), (1, [11], [20])])
    g2 = dag([(0, [10], [21]), (1, [21], [20])])
    equiv = [(0, 0), (1, 1)]
    matches = recursive_match(g1, g2, equiv, MatchConfig(max_recursion_depth=8), 0)
    assert matches, "should terminate with a result, not an empty one"
    assert all(m.graph1_nodes for m in matches)


def test_match_graphs_runs_end_to_end():
    g1, g2 = two_chains()
    matches = match_graphs(g1, g2, MatchConfig(check_stats=False))
    assert matches
    for m in matches:
        assert m.graph1_nodes and m.graph2_nodes
        assert m.graph1_nodes <= set(g1.nodes)
        assert m.graph2_nodes <= set(g2.nodes)


def test_min_subgraph_size_filters_the_result():
    g1, g2 = two_chains()
    big = match_graphs(g1, g2, MatchConfig(min_subgraph_size=1))
    huge = match_graphs(g1, g2, MatchConfig(min_subgraph_size=99))
    assert big and not huge


# --- what counts as the same tensor ------------------------------------------


def spec(shape=(2, 2)):
    return {"shape": list(shape), "dtype": "?", "device": "cpu"}


def test_complex_tensors_are_compared_by_magnitude():
    """They used to compare equal to anything of the same size."""
    from magneton.matching.tensor import tensors_equivalent

    a = torch.tensor([1 + 1j, 2 + 2j])
    same = torch.tensor([1 + 1j, 2 + 2j])
    # A phase rotation: different numbers, same magnitudes, and eigenvectors
    # are only defined up to one.
    rotated = a * torch.exp(torch.tensor(1j))
    different = torch.tensor([90 + 90j, 90 + 90j])

    assert tensors_equivalent(spec(), spec(), a, same)
    assert tensors_equivalent(spec(), spec(), a, rotated)
    assert not tensors_equivalent(spec(), spec(), a, different)


def test_a_complex_tensor_is_not_a_float_tensor():
    from magneton.matching.tensor import tensors_equivalent

    complex_values = torch.tensor([1 + 0j, 2 + 0j])
    real_values = torch.tensor([1.0, 2.0])
    assert not tensors_equivalent(spec(), spec(), complex_values, real_values)
    assert not tensors_equivalent(spec(), spec(), real_values, complex_values)


def test_the_configured_tolerance_is_the_one_used():
    """MatchConfig.stat_tolerance reached nothing: the comparison always ran at"""
    from magneton.matching.tensor import find_equivalent_node_pairs

    a = dag([(0, [10], [11])])
    b = dag([(0, [10], [12])])
    # Tensor 11 has mean 11, tensor 12 has mean 12: about 4% apart.
    assert find_equivalent_node_pairs(a, b, stat_tolerance=1e-3) == []
    assert find_equivalent_node_pairs(a, b, stat_tolerance=0.5) == [(0, 0)]


def test_an_unused_entry_does_not_hide_the_graph():
    """The lowest node id is an entry only by coincidence."""
    d = dag([(0, [], [10]), (1, [], [11]), (2, [], [12]), (3, [11, 12], [13])])
    # Node 0 produces a tensor nothing consumes, so the walk starts at 1 -- the
    # lowest entry that actually reaches the sink -- and 1 dominates 3.
    assert DominatorTree(d).dominator_path == [1, 3]

    other = dag([(0, [], [20]), (1, [], [11]), (2, [], [12]), (3, [11, 12], [13])])
    matches = recursive_match(d, other, [(3, 3)], MatchConfig(), 0)
    assert len(matches) == 1


def test_integer_tensors_of_the_same_size_but_different_shape():
    """Equivalence is by element count, so these."""
    from magneton.matching.tensor import tensors_equivalent

    a = torch.arange(1024, dtype=torch.int64).reshape(512, 2)
    same = torch.arange(1024, dtype=torch.int64).reshape(2, 512)
    different = torch.zeros(1024, dtype=torch.int64).reshape(2, 512)

    assert tensors_equivalent(spec(), spec(), a, same)
    assert not tensors_equivalent(spec(), spec(), a, different)
