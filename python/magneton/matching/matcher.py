"""
Recursive subgraph matching algorithm.

Implements topology-aware divide-and-conquer matching using dominator trees.
"""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple
import torch
from magneton.dataflow import DataflowDAG
from .dominator import DominatorTree
from .tensor import find_equivalent_node_pairs
from .subgraph import SubgraphMatch


@dataclasses.dataclass
class MatchConfig:
    """Configuration for recursive subgraph matching."""
    
    check_stats: bool = True
    """Whether to compare tensor statistics (mean, std)"""
    
    stat_tolerance: float = 1e-3
    """How far two tensors' mean and standard deviation may differ and still
    count as the same value, relative to their size.

    1e-3 suits comparisons that differ only by floating-point association --
    a fused op against the same arithmetic done separately. Comparisons that
    change the arithmetic need more room: TF32 keeps about 10 bits of mantissa,
    so the same matmul in TF32 and FP32 differs by roughly 1e-3 and sits right
    on this boundary."""
    
    max_recursion_depth: int = 100
    """Maximum recursion depth to prevent infinite loops"""
    
    min_subgraph_size: int = 1
    """Minimum number of nodes in a subgraph to report"""
    
    verbose: bool = False
    """Print debug information during matching"""


def match_graphs(
    dag1: DataflowDAG,
    dag2: DataflowDAG,
    config: Optional[MatchConfig] = None
) -> List[SubgraphMatch]:
    """
    Find all minimal equivalent subgraph pairs between two DAGs.
    
    Uses a topology-aware divide-and-conquer approach based on dominator trees
    to achieve O(N²) complexity.
    
    Args:
        dag1: First dataflow DAG
        dag2: Second dataflow DAG
        config: Matching configuration
    
    Returns:
        List of SubgraphMatch objects representing minimal equivalent subgraphs
    
    Example:
        >>> from magneton.matching import match_graphs, MatchConfig
        >>> config = MatchConfig(check_stats=True, stat_tolerance=1e-3)
        >>> matches = match_graphs(dag1, dag2, config)
        >>> print(f"Found {len(matches)} equivalent subgraph pairs")
    """
    if config is None:
        config = MatchConfig()
    
    if config.verbose:
        print(f"Starting graph matching: {len(dag1.nodes)} nodes vs {len(dag2.nodes)} nodes")
    
    # Phase 1: Find equivalent tensors
    equiv_tensors = find_equivalent_node_pairs(
        dag1, dag2, stat_tolerance=config.stat_tolerance
    )
    
    if not equiv_tensors:
        if config.verbose:
            print("No equivalent tensors found")
        return []
    
    if config.verbose:
        print(f"Found {len(equiv_tensors)} equivalent tensor pairs")
    
    # Phase 2: Recursive matching
    matches = recursive_match(dag1, dag2, equiv_tensors, config, depth=0)

    # Filter by minimum size
    matches = [m for m in matches if len(m.graph1_nodes) >= config.min_subgraph_size]

    if config.verbose:
        print(f"Found {len(matches)} minimal equivalent subgraphs")

    return matches


def recursive_match(
    dag1: DataflowDAG,
    dag2: DataflowDAG,
    equiv_tensors: List[Tuple[int, int]],
    config: MatchConfig,
    depth: int = 0,
    bounds1: Tuple[Optional[Set[int]], Optional[Set[int]]] = (None, None),
    bounds2: Tuple[Optional[Set[int]], Optional[Set[int]]] = (None, None),
) -> List[SubgraphMatch]:
    """
    Split two equivalent graphs into the smallest equivalent pieces they share.

    Topology-aware divide and conquer. The dominator path of a graph -- the
    chain of nodes every path from source to sink must pass through -- is a
    spine that no amount of branching can route around. Where two graphs have
    equivalent tensors at spine nodes, those points are cuts that hold for the
    whole graph, so the regions between successive cuts can only correspond to
    each other, and each pair can be matched independently.

    That is what makes this O(N^2) rather than the subgraph isomorphism problem:
    the cuts are found by comparing tensors, and the recursion never has to
    search for a correspondence between regions.

        M <- {}
        T1, T2 <- DominatorTree(G1), DominatorTree(G2)
        P1, P2 <- dominator paths, source to sink
        E  <- {(t1, t2) in P1 x P2 | out(t1) equivalent to out(t2)}
        if |E| = 1: return G1, G2            -- nothing left to divide by
        for each consecutive (e_k, e_k+1) in E:
            G1k <- G1[{v | t1_k dominates v, v dominates t1_k+1}]
            G2k <- G2[{v | t2_k dominates v, v dominates t2_k+1}]
            M <- M + RecursiveMatch(G1k, G2k)
        return M

    Args:
        dag1, dag2: the two graphs, known to be equivalent as a whole
        equiv_tensors: node pairs whose outputs hold equivalent tensors
        config: matching configuration
        depth: recursion depth, against `config.max_recursion_depth`

    Returns:
        The minimal equivalent subgraph pairs found beneath this pair.
    """
    if not dag1.nodes or not dag2.nodes:
        return []

    indent = "  " * depth
    if depth >= config.max_recursion_depth:
        if config.verbose:
            print(f"{indent}depth limit at {len(dag1.nodes)}/{len(dag2.nodes)} nodes")
        return [_whole_graph_match(dag1, dag2, equiv_tensors)]

    # The ends of a region are the cuts it was carved between, so say so
    # rather than letting the tree guess from ids that mean nothing here.
    tree1 = DominatorTree(dag1, sources=bounds1[0], sinks=bounds1[1])
    tree2 = DominatorTree(dag2, sources=bounds2[0], sinks=bounds2[1])
    path1, path2 = tree1.dominator_path, tree2.dominator_path

    cuts = find_cut_points(path1, path2, equiv_tensors)
    if config.verbose:
        print(
            f"{indent}{len(dag1.nodes)}/{len(dag2.nodes)} nodes, "
            f"spine {len(path1)}/{len(path2)}, {len(cuts)} cut points"
        )

    # No cut point at all: nothing here is known to correspond, so there is no
    # match to report. Distinct from one cut point, which is a match.
    if not cuts:
        return []

    # One cut point: the spine offers nowhere to divide, so this pair is
    # already minimal. This is the base case the recursion descends toward.
    if len(cuts) == 1:
        return [_whole_graph_match(dag1, dag2, equiv_tensors)]

    matches: List[SubgraphMatch] = []
    for (start1, start2), (end1, end2) in zip(cuts, cuts[1:]):
        sub1 = extract_subgraph_between_cuts(dag1, tree1, start1, end1)
        sub2 = extract_subgraph_between_cuts(dag2, tree2, start2, end2)
        if not sub1.nodes or not sub2.nodes:
            continue

        # A region that is the whole graph again would recurse forever. It
        # means the cuts did not actually divide anything, so stop and take the
        # pair as minimal.
        if len(sub1.nodes) >= len(dag1.nodes) and len(sub2.nodes) >= len(dag2.nodes):
            matches.append(_whole_graph_match(sub1, sub2, equiv_tensors))
            continue

        matches.extend(
            recursive_match(
                sub1, sub2, equiv_tensors, config, depth + 1,
                bounds1=({start1}, {end1}), bounds2=({start2}, {end2}),
            )
        )

    return matches


def _whole_graph_match(
    dag1: DataflowDAG,
    dag2: DataflowDAG,
    equiv_tensors: List[Tuple[int, int]],
) -> SubgraphMatch:
    """The pair taken whole: a region that the spine cannot divide further.

    Its input and output edges are the tensors crossing its boundary, paired up
    through the node equivalences that justified the match in the first place.
    """
    nodes1, nodes2 = set(dag1.nodes), set(dag2.nodes)
    equiv_of = {}
    for n1, n2 in equiv_tensors:
        if n1 in nodes1 and n2 in nodes2:
            equiv_of.setdefault(n1, n2)

    def boundary(dag, inward: bool):
        """Nodes whose inputs come from outside (inward), or whose outputs
        leave (outward)."""
        produced = {t for n in dag.nodes.values() for t in n.output_tensor_ids}
        consumed = {t for n in dag.nodes.values() for t in n.input_tensor_ids}
        out = []
        for node in dag.nodes.values():
            tensors = node.input_tensor_ids if inward else node.output_tensor_ids
            against = produced if inward else consumed
            for index, tid in enumerate(tensors):
                if tid not in against:
                    out.append((node.node_id, index))
        return out

    def paired(edges1, edges2):
        by_node2 = {}
        for node_id, index in edges2:
            by_node2.setdefault(node_id, []).append(index)
        pairs = []
        for node_id, index in edges1:
            partner = equiv_of.get(node_id)
            if partner is not None and by_node2.get(partner):
                pairs.append(((node_id, index), (partner, by_node2[partner][0])))
        return pairs

    return SubgraphMatch(
        graph1_nodes=nodes1,
        graph2_nodes=nodes2,
        input_edges=paired(boundary(dag1, True), boundary(dag2, True)),
        output_edges=paired(boundary(dag1, False), boundary(dag2, False)),
    )


def find_cut_points(
    path1: List[int],
    path2: List[int],
    equiv_tensors: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Find equivalent pairs along dominator paths that serve as cut points.
    
    A cut point is a pair of nodes (n1, n2) where:
    - n1 is on path1, n2 is on path2
    - Their output tensors are equivalent
    
    Args:
        path1, path2: Dominator paths for dag1 and dag2
        equiv_tensors: Set of equivalent tensor pairs
    
    Returns:
        Cut point pairs in path order, increasing along both paths.

    The pairs must advance in *both* paths together. A cut that went forward in
    one and backward in the other would describe a region running one way
    through the first graph and the other way through the second, and the
    subgraphs between successive cuts would overlap rather than partition. So a
    candidate is taken only when it lies beyond the last one on both sides,
    which is what makes the regions a partition and the recursion finite.
    """
    equiv_set = set(equiv_tensors)
    position2 = {node: i for i, node in enumerate(path2)}

    cut_points = []
    taken2 = -1
    for n1 in path1:
        candidates = [
            n2 for n2 in path2
            if position2[n2] > taken2 and (n1, n2) in equiv_set
        ]
        if candidates:
            # The earliest still-available partner, so the regions stay as
            # small as they can be and the division is as fine as possible.
            n2 = min(candidates, key=lambda n: position2[n])
            cut_points.append((n1, n2))
            taken2 = position2[n2]

    return cut_points


def get_reachable_nodes(dag: DataflowDAG, start_node: int) -> Set[int]:
    """
    Get all nodes reachable from start_node via forward edges.
    
    Skips edges through torch.nn.Parameter.
    
    Args:
        dag: The dataflow DAG
        start_node: Starting node ID
    
    Returns:
        Set of node IDs reachable from start_node (including start_node itself)
    """
    # Build forward adjacency list (skip Parameter edges)
    forward_adj: Dict[int, List[int]] = {node_id: [] for node_id in dag.nodes}
    
    # Build tensor_id -> producer_node_id mapping
    tensor_producers: Dict[int, int] = {}
    for node in dag.nodes.values():
        for tensor_id in node.output_tensor_ids:
            if tensor_id not in tensor_producers:
                tensor_producers[tensor_id] = node.node_id
    
    # Build forward edges (skip Parameters)
    for node in dag.nodes.values():
        for tensor_id in node.input_tensor_ids:
            # Skip Parameter edges
            if tensor_id in dag.tensors and isinstance(dag.tensors[tensor_id], torch.nn.Parameter):
                continue
            if tensor_id in tensor_producers:
                producer_id = tensor_producers[tensor_id]
                if node.node_id not in forward_adj[producer_id]:
                    forward_adj[producer_id].append(node.node_id)
    
    # BFS from start_node
    reachable = set()
    if start_node not in dag.nodes:
        return reachable
    
    queue = [start_node]
    reachable.add(start_node)
    
    while queue:
        current = queue.pop(0)
        for successor in forward_adj.get(current, []):
            if successor not in reachable:
                reachable.add(successor)
                queue.append(successor)
    
    return reachable


def get_nodes_reaching(dag: DataflowDAG, end_node: int) -> Set[int]:
    """
    Get all nodes that can reach end_node via forward edges.
    
    Skips edges through torch.nn.Parameter.
    
    Args:
        dag: The dataflow DAG
        end_node: Target node ID
    
    Returns:
        Set of node IDs that can reach end_node (including end_node itself)
    """
    # Build backward adjacency list (skip Parameter edges)
    backward_adj: Dict[int, List[int]] = {node_id: [] for node_id in dag.nodes}
    
    # Build tensor_id -> producer_node_id mapping
    tensor_producers: Dict[int, int] = {}
    for node in dag.nodes.values():
        for tensor_id in node.output_tensor_ids:
            if tensor_id not in tensor_producers:
                tensor_producers[tensor_id] = node.node_id
    
    # Build backward edges (skip Parameters)
    for node in dag.nodes.values():
        for tensor_id in node.input_tensor_ids:
            # Skip Parameter edges
            if tensor_id in dag.tensors and isinstance(dag.tensors[tensor_id], torch.nn.Parameter):
                continue
            if tensor_id in tensor_producers:
                producer_id = tensor_producers[tensor_id]
                if producer_id not in backward_adj[node.node_id]:
                    backward_adj[node.node_id].append(producer_id)
    
    # BFS backward from end_node
    can_reach = set()
    if end_node not in dag.nodes:
        return can_reach
    
    queue = [end_node]
    can_reach.add(end_node)
    
    while queue:
        current = queue.pop(0)
        for predecessor in backward_adj.get(current, []):
            if predecessor not in can_reach:
                can_reach.add(predecessor)
                queue.append(predecessor)
    
    return can_reach


def extract_subgraph_between_cuts(
    dag: DataflowDAG,
    dom_tree: DominatorTree,
    start_node: int,
    end_node: int
) -> DataflowDAG:
    """
    Extract subgraph between two cut points.
    
    The subgraph includes all nodes v where:
    - v is reachable from start_node
    - v can reach end_node
    
    Args:
        dag: The full dataflow DAG
        dom_tree: The dominator tree (kept for compatibility, not used)
        start_node: Starting cut point (inclusive)
        end_node: Ending cut point (inclusive)
    
    Returns:
        A new DataflowDAG containing only the subgraph nodes
    """
    # Get nodes reachable from start
    reachable_from_start = get_reachable_nodes(dag, start_node)
    
    # Get nodes that can reach end
    can_reach_end = get_nodes_reaching(dag, end_node)
    
    # Intersection: nodes on paths from start to end
    subgraph_nodes = reachable_from_start & can_reach_end
    
    # Create new DAG with only these nodes
    sub_dag = DataflowDAG()
    
    for node_id in sorted(subgraph_nodes):
        if node_id in dag.nodes:
            node = dag.nodes[node_id]
            # Copy node to new DAG
            sub_dag.nodes[node_id] = node
            
            # Copy relevant tensors
            for tid in node.input_tensor_ids + node.output_tensor_ids:
                if tid in dag.tensors:
                    sub_dag.tensors[tid] = dag.tensors[tid]
                if tid in dag.tensor_metadata:
                    sub_dag.tensor_metadata[tid] = dag.tensor_metadata[tid]
    
    return sub_dag


def create_subgraph_match(
    dag1: DataflowDAG,
    dag2: DataflowDAG,
    equiv_tensors: List[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> Optional[SubgraphMatch]:
    """
    Create a SubgraphMatch from two subgraphs and their equivalent tensors.
    
    Args:
        dag1, dag2: The subgraphs
        equiv_tensors: Equivalent tensor pairs
    
    Returns:
        SubgraphMatch object, or None if invalid
    """
    # Get all node IDs in subgraphs
    nodes1 = set(dag1.nodes.keys())
    nodes2 = set(dag2.nodes.keys())
    
    if not nodes1 or not nodes2:
        return None
    
    # Find input edges: tensors produced outside, consumed inside
    input_edges = []
    
    # Find output edges: tensors produced inside
    output_edges = []
    
    # Collect all edges from equiv_tensors that involve nodes in the subgraphs
    for (n1, idx1), (n2, idx2) in equiv_tensors:
        # Both nodes in subgraphs -> potential output edge
        if n1 in nodes1 and n2 in nodes2:
            output_edges.append(((n1, idx1), (n2, idx2)))
        # One node in, one out -> potential input edge
        elif (n1 not in nodes1 and n2 in nodes2) or (n1 in nodes1 and n2 not in nodes2):
            # This is tricky - for now, skip
            pass
    
    # If no output edges, try to find any equivalent tensors at boundaries
    if not output_edges:
        # Just use any equiv tensors involving these nodes
        for (n1, idx1), (n2, idx2) in equiv_tensors:
            if n1 in nodes1 or n2 in nodes2:
                output_edges.append(((n1, idx1), (n2, idx2)))
    
    # For input edges, find tensors consumed by subgraph nodes
    # that are produced outside (or are inputs to the full graph)
    for node_id in nodes1:
        node = dag1.nodes[node_id]
        for tid in node.input_tensor_ids:
            # Find which node produced this tensor
            producer = None
            for other_id in dag1.nodes:
                if tid in dag1.nodes[other_id].output_tensor_ids:
                    producer = other_id
                    break
            
            # If producer is outside subgraph, this is an input edge
            if producer is not None and producer not in nodes1:
                # Find corresponding tensor in dag2
                for (n1, idx1), (n2, idx2) in equiv_tensors:
                    if n1 == producer:
                        input_edges.append(((n1, idx1), (n2, idx2)))
    
    return SubgraphMatch(
        graph1_nodes=nodes1,
        graph2_nodes=nodes2,
        input_edges=input_edges,
        output_edges=output_edges
    )


def filter_equiv_tensors(
    equiv_tensors: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    dag1: DataflowDAG,
    dag2: DataflowDAG
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Filter equivalent tensors to only those relevant to the subgraphs.
    
    Args:
        equiv_tensors: Full set of equivalent tensor pairs
        dag1, dag2: Subgraphs
    
    Returns:
        Filtered list of equivalent tensors
    """
    nodes1 = set(dag1.nodes.keys())
    nodes2 = set(dag2.nodes.keys())
    
    filtered = []
    for (n1, idx1), (n2, idx2) in equiv_tensors:
        # Include if both nodes are in the subgraphs
        if n1 in nodes1 and n2 in nodes2:
            filtered.append(((n1, idx1), (n2, idx2)))
    
    return filtered

