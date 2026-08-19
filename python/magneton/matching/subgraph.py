"""
Subgraph match result representation.

Minimal classes to represent matching results and export functions.
"""

import dataclasses
import json
from typing import TYPE_CHECKING, List, Set, Tuple, Union

if TYPE_CHECKING:
    from magneton.dataflow import DataflowDAG


@dataclasses.dataclass
class SubgraphMatch:
    """
    Represents a pair of equivalent subgraphs.
    
    Two subgraphs (V1, E1) and (V2, E2) are equivalent if:
    - Their input edges have equivalent tensor values
    - Their output edges have equivalent tensor values
    
    This is the primary result type returned by the matching algorithm.
    """
    
    graph1_nodes: Set[int]
    """Set of node IDs from first graph (V1)"""
    
    graph2_nodes: Set[int]
    """Set of node IDs from second graph (V2)"""
    
    input_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    """
    List of equivalent input edges (edges entering the subgraph).
    Each pair: ((src_node_id1, output_idx1), (src_node_id2, output_idx2))
    These are edges from outside the subgraph into the subgraph.
    """
    
    output_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    """
    List of equivalent output edges (edges leaving the subgraph).
    Each pair: ((src_node_id1, output_idx1), (src_node_id2, output_idx2))
    These are edges from inside the subgraph to outside (or to sinks).
    """
    
    def size(self) -> Tuple[int, int]:
        """
        Get the size of both subgraphs.
        
        Returns:
            Tuple of (graph1_size, graph2_size)
        
        Example:
            >>> match = SubgraphMatch(...)
            >>> size1, size2 = match.size()
            >>> print(f"Subgraph 1: {size1} nodes, Subgraph 2: {size2} nodes")
        """
        return len(self.graph1_nodes), len(self.graph2_nodes)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        size1, size2 = self.size()
        return (
            f"SubgraphMatch("
            f"graph1={size1} nodes, "
            f"graph2={size2} nodes, "
            f"input_edges={len(self.input_edges)}, "
            f"output_edges={len(self.output_edges)})"
        )


def export_matches(
    matches: List[SubgraphMatch],
    dag1: Union['DataflowDAG', str],
    dag2: Union['DataflowDAG', str],
    output_path: str
):
    """
    Export matching results to JSON file.
    
    Output format:
    {
        "total_matches": int,
        "matches": [
            {
                "graph1_nodes": [node_ids],
                "graph2_nodes": [node_ids],
                "input_edges": [{"graph1": {...}, "graph2": {...}}],
                "output_edges": [{"graph1": {...}, "graph2": {...}}]
            },
            ...
        ]
    }
    
    Args:
        matches: List of SubgraphMatch objects
        dag1: First DataflowDAG or path to JSON (not used, kept for compatibility)
        dag2: Second DataflowDAG or path to JSON (not used, kept for compatibility)
        output_path: Path to save results
    
    Example:
        >>> export_matches(matches, dag1, dag2, "matches.json")
    """
    # Build output structure
    output = {
        "total_matches": len(matches),
        "matches": [
            {
                "graph1_nodes": sorted(list(match.graph1_nodes)),
                "graph2_nodes": sorted(list(match.graph2_nodes)),
                "graph1_size": len(match.graph1_nodes),
                "graph2_size": len(match.graph2_nodes),
                "input_edges": [
                    {
                        "graph1": {"node_id": n1, "output_idx": i1},
                        "graph2": {"node_id": n2, "output_idx": i2}
                    }
                    for (n1, i1), (n2, i2) in match.input_edges
                ],
                "output_edges": [
                    {
                        "graph1": {"node_id": n1, "output_idx": i1},
                        "graph2": {"node_id": n2, "output_idx": i2}
                    }
                    for (n1, i1), (n2, i2) in match.output_edges
                ],
            }
            for match in matches
        ]
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def print_matches(matches: List[SubgraphMatch], verbose: bool = False):
    """
    Print matching results to console.
    
    Args:
        matches: List of SubgraphMatch objects
        verbose: If True, print details of each match
    
    Example:
        >>> print_matches(matches, verbose=True)
    """
    print("=" * 60)
    print(f"Found {len(matches)} equivalent subgraph pairs")
    print("=" * 60)
    
    if verbose and matches:
        for i, match in enumerate(matches, 1):
            size1, size2 = match.size()
            print(f"\nMatch {i}:")
            print(f"  Graph 1: {size1} nodes {sorted(match.graph1_nodes)}")
            print(f"  Graph 2: {size2} nodes {sorted(match.graph2_nodes)}")
            print(f"  Input edges: {len(match.input_edges)}")
            print(f"  Output edges: {len(match.output_edges)}")
            
            if verbose and match.input_edges:
                print("  Input edge pairs:")
                for (n1, i1), (n2, i2) in match.input_edges:
                    print(f"    ({n1}, {i1}) <-> ({n2}, {i2})")
            
            if verbose and match.output_edges:
                print("  Output edge pairs:")
                for (n1, i1), (n2, i2) in match.output_edges:
                    print(f"    ({n1}, {i1}) <-> ({n2}, {i2})")

