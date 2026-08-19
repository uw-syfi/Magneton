"""
Dominator tree construction for dataflow DAGs.

Implements the Cooper-Harvey-Kennedy (Simple Fast Dominance) algorithm.
"""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from magneton.dataflow import DataflowDAG
from magneton.utils.fx_utils import OpType


def build_adjacency_lists(
    dag: DataflowDAG,
    sources: Set[int],
    sinks: Set[int],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build forward and backward adjacency lists from a DAG.

    Edges are derived from tensor IDs: if a node consumes a tensor produced by another node,
    there's an edge between them. Edges through torch.nn.Parameter do not count.

    Only includes nodes that are:
    1. Reachable from node 0 (entry point), AND
    2. Can reach the last node (exit point)

    Args:
        dag: DataflowDAG object

    Returns:
        Tuple of (forward_adj, backward_adj) where:
        - forward_adj[node_id] = list of successor node IDs
        - backward_adj[node_id] = list of predecessor node IDs

    Example:
        >>> forward, backward = build_adjacency_lists(dag)
        >>> successors = forward[node_id]
        >>> predecessors = backward[node_id]
    """
    # Build full adjacency lists first
    forward_adj: Dict[int, Set[int]] = {node_id: set() for node_id in dag.nodes}
    backward_adj: Dict[int, Set[int]] = {node_id: set() for node_id in dag.nodes}

    tensor_producers: Dict[int, int] = {}
    for node in dag.nodes.values():
        if node.op_type != OpType.INPUT.value:
            for tensor_id in node.input_tensor_ids:
                # A tensor produced outside this DAG is an input to it, not a
                # dangling reference. That is the normal case for a subgraph,
                # which is what the matcher recurses into.
                producer_id = tensor_producers.get(tensor_id)
                if producer_id is None:
                    continue
                backward_adj[node.node_id].add(producer_id)
                forward_adj[producer_id].add(node.node_id)
        for tensor_id in node.output_tensor_ids:
            tensor_producers[tensor_id] = node.node_id

    # BFS from node 0 to find all reachable nodes
    reachable_from_start = set(sources)
    queue = list(sources)
    while queue:
        current = queue.pop(0)
        for successor in forward_adj[current].difference(reachable_from_start):
            reachable_from_start.add(successor)
            queue.append(successor)

    # BFS backward from last node to find all nodes that can reach it
    can_reach_end = set(sinks)
    queue = list(sinks)
    while queue:
        current = queue.pop(0)
        for predecessor in backward_adj[current].difference(can_reach_end):
            can_reach_end.add(predecessor)
            queue.append(predecessor)

    # Keep only nodes that satisfy both conditions
    valid_nodes = reachable_from_start & can_reach_end

    # Filter adjacency lists to only include valid nodes
    filtered_forward: Dict[int, List[int]] = {}
    filtered_backward: Dict[int, List[int]] = {}

    for node_id in valid_nodes:
        filtered_forward[node_id] = list(forward_adj[node_id].intersection(valid_nodes))
        filtered_backward[node_id] = list(
            backward_adj[node_id].intersection(valid_nodes)
        )

    return filtered_forward, filtered_backward


@dataclasses.dataclass
class DominatorTree:
    """
    Dominator tree for a dataflow DAG.

    Uses the Cooper-Harvey-Kennedy algorithm to compute dominators in O(N²) worst case,
    but typically O(N) on real graphs.

    Attributes:
        dag: The dataflow DAG
        idom: Immediate dominator for each node (None for entry)
        children: Children in the dominator tree
    """

    dag: DataflowDAG
    idom: Dict[int, Optional[int]]
    children: Dict[int, List[int]]
    _postorder_num: Dict[int, int]
    _dom_set_cache: Dict[int, Set[int]] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        dag: DataflowDAG,
        sources: Optional[Set[int]] = None,
        sinks: Optional[Set[int]] = None,
    ):
        """
        Build dominator tree using Cooper-Harvey-Kennedy algorithm.

        Args:
            dag: The dataflow DAG to analyze
            sources: where the graph starts. Defaults to its lowest node id,
                which for a traced graph is the input placeholder.
            sinks: where it ends. Defaults to its highest node id.

        The defaults are right for a whole graph and wrong for a piece of one.
        A recorded graph has many nodes whose outputs nothing consumes -- dead
        ends, and values kept but never used again -- so deriving the sinks
        from the tensor flow finds hundreds of them, and then nothing dominates
        the exit and the dominator path is empty. A subgraph knows its own ends,
        because they are the cut points it was carved between, so the matcher
        passes them in.
        """
        self.dag = dag
        self._sources = sources
        self._sinks = sinks
        self.idom = {}
        self.children = {}
        self._postorder_num = {}
        self._dom_set_cache = {}
        self._dominator_paths = []
        self._build()

    def _build(self):
        """
        Main build method.

        Steps:
        1. Add virtual entry node (if multiple entries)
        2. Compute reverse postorder
        3. Build predecessor lists
        4. Compute immediate dominators iteratively
        5. Build tree structure
        """
        sources = set(self._sources) if self._sources else self._entry_nodes()
        sinks = set(self._sinks) if self._sinks else {max(self.dag.nodes)}

        # Step 1: Handle multiple entry nodes
        adj_list, _ = build_adjacency_lists(self.dag, sources, sinks)
        sources &= adj_list.keys()
        sinks &= adj_list.keys()

        # Build virtual entry and exit nodes. They give the graph a single
        # source and a single sink, so that the dominator relation is defined
        # even when it has several of either.
        virtual_entry = -1
        virtual_exit = max(self.dag.nodes.keys()) + 1
        adj_list[virtual_entry] = list(sources)
        adj_list[virtual_exit] = []
        for sink in sinks:
            adj_list[sink].append(virtual_exit)

        # Build predecessor lists (reverse edges)
        pred_list = {}
        for node in adj_list:
            for child in adj_list.get(node, []):
                if child not in pred_list:
                    pred_list[child] = []
                pred_list[child].append(node)

        # Step 2: Compute reverse postorder
        rpo = self._compute_reverse_postorder(virtual_entry, adj_list)

        # Store postorder numbers for _intersect
        for i, node in enumerate(rpo):
            self._postorder_num[node] = i

        # Step 3: Compute immediate dominators
        self.idom = self._compute_dominators(virtual_entry, rpo, adj_list, pred_list)

        # Step 4: Build tree structure
        self._build_tree_structure()

        # Step 5: Compute dominator paths
        self._dominator_paths = self._compute_dominator_paths(
            virtual_entry, virtual_exit
        )

    def _entry_nodes(self) -> Set[int]:
        """Where to start the walk, when the caller did not say.

        The lowest node id that can actually reach the sink. Both halves of that
        matter, and each was learned from a graph the other rule broke:

        - not simply the lowest id. A graph with several independent inputs
          numbers them 0, 1, 2, and the operation consuming them may take 1 and
          2 and never 0. Starting at 0 reaches nothing and every node is then
          filtered out as unreachable.
        - not every node without a producer either. In a recorded graph every
          weight is one, so that is most of the graph; with that many entries
          almost nothing dominates the exit and the dominator path collapses to
          nothing. Following one entry's forward cone is what gives a spine
          worth cutting on, and the lowest-numbered one is the model's input
          rather than its parameters.
        """
        produced_here = {
            tid for node in self.dag.nodes.values() for tid in node.output_tensor_ids
        }
        producers = {
            tid: node.node_id
            for node in self.dag.nodes.values()
            for tid in node.output_tensor_ids
        }
        consumers: Dict[int, List[int]] = {}
        for node in self.dag.nodes.values():
            for tid in node.input_tensor_ids:
                producer = producers.get(tid)
                if producer is not None:
                    consumers.setdefault(producer, []).append(node.node_id)

        sink = max(self.dag.nodes)
        # Everything that can reach the sink, walking the edges backwards.
        reaches_sink = {sink}
        frontier = [sink]
        backwards: Dict[int, List[int]] = {}
        for producer, consuming in consumers.items():
            for consumer in consuming:
                backwards.setdefault(consumer, []).append(producer)
        while frontier:
            node_id = frontier.pop()
            for predecessor in backwards.get(node_id, []):
                if predecessor not in reaches_sink:
                    reaches_sink.add(predecessor)
                    frontier.append(predecessor)

        entries = sorted(
            node.node_id
            for node in self.dag.nodes.values()
            if not any(tid in produced_here for tid in node.input_tensor_ids)
            and node.node_id in reaches_sink
        )
        return {entries[0]} if entries else {min(self.dag.nodes)}

    @property
    def dominator_path(self) -> List[int]:
        """The source-to-sink chain of the dominator tree: every node that every
        path through the graph must pass through, in order.

        This is P in the matching algorithm. The virtual entry and exit are not
        in it -- they are scaffolding for the tree, not nodes of the graph.
        """
        real = set(self.dag.nodes)
        return [n for n in self._dominator_paths if n in real]

    def _compute_reverse_postorder(
        self, entry: int, adj_list: Dict[int, List[int]]
    ) -> List[int]:
        """
        Compute reverse postorder traversal.

        Reverse postorder ensures that dominators are processed before
        dominated nodes, leading to fast convergence.

        Args:
            entry: Entry node
            adj_list: Adjacency list

        Returns:
            List of node IDs in reverse postorder
        """
        # Iterative, not recursive: a real dataflow graph is a few thousand ops
        # deep and mostly a chain, so a recursive walk exceeds Python's stack
        # long before it runs out of graph.
        visited = set()
        postorder = []
        # Each frame is a node and how many of its children have been dealt
        # with; a node is appended once its children are all behind it.
        stack = [(entry, 0)]
        visited.add(entry)

        while stack:
            node, index = stack.pop()
            children = adj_list.get(node, [])
            if index < len(children):
                stack.append((node, index + 1))
                child = children[index]
                if child not in visited:
                    visited.add(child)
                    stack.append((child, 0))
            else:
                postorder.append(node)

        # Reverse to get reverse postorder
        return list(reversed(postorder))

    def _compute_dominators(
        self,
        entry: int,
        rpo: List[int],
        adj_list: Dict[int, List[int]],
        pred_list: Dict[int, List[int]],
    ) -> Dict[int, Optional[int]]:
        """
        Compute immediate dominators using iterative dataflow.

        Algorithm:
        1. Initialize: idom[entry] = entry, all others = undefined
        2. Iterate until no changes:
           - For each node b (in reverse postorder):
             - idom[b] = intersection of idom[pred] for all predecessors

        Args:
            entry: Entry node
            rpo: Nodes in reverse postorder
            adj_list: Forward adjacency list
            pred_list: Predecessor list (reverse edges)

        Returns:
            Immediate dominator mapping
        """
        # Initialize
        idom = {}
        idom[entry] = entry  # Entry dominates itself

        # Iterate until convergence
        changed = True
        while changed:
            changed = False

            # Process nodes in reverse postorder (skip entry)
            for node in rpo[1:]:
                # Find predecessors that have been processed
                processed_preds = [p for p in pred_list.get(node, []) if p in idom]

                if not processed_preds:
                    continue

                # Start with first processed predecessor
                new_idom = processed_preds[0]

                # Intersect with other predecessors
                for pred in processed_preds[1:]:
                    new_idom = self._intersect(pred, new_idom, idom)

                # Update if changed
                if node not in idom or idom[node] != new_idom:
                    idom[node] = new_idom
                    changed = True

        # Convert entry's idom to None (standard representation)
        idom[entry] = None

        return idom

    def _intersect(self, node1: int, node2: int, idom: Dict[int, Optional[int]]) -> int:
        """
        Find the lowest common ancestor of two nodes in the dominator tree.

        This is the intersection of their dominator sets.

        Args:
            node1, node2: Nodes to intersect
            idom: Current immediate dominator mapping

        Returns:
            The lowest common ancestor (intersection point)
        """
        finger1 = node1
        finger2 = node2

        # Walk up the dominator tree until paths meet
        while finger1 != finger2:
            while self._postorder_num[finger1] > self._postorder_num[finger2]: # type: ignore
                finger1: int = idom[finger1] # type: ignore
                # If we reached the entry (which dominates itself), stop
                if idom[finger1] == finger1:
                    return finger1
            while self._postorder_num[finger2] > self._postorder_num[finger1]: # type: ignore
                finger2: int = idom[finger2] # type: ignore
                # If we reached the entry (which dominates itself), stop
                if idom[finger2] == finger2:
                    return finger2
        return finger1

    def _build_tree_structure(self):
        """
        Build dominator tree structure from immediate dominators.

        Populates self.children.
        """
        self.children = {}

        for node, dom in self.idom.items():
            if dom is not None:
                if dom not in self.children:
                    self.children[dom] = []
                self.children[dom].append(node)

    def dominates(self, node_a: int, node_b: int) -> bool:
        """
        Check if node_a dominates node_b.

        A dominates B if A is an ancestor of B in the dominator tree.

        Args:
            node_a: Potential dominator
            node_b: Node to check

        Returns:
            True if node_a dominates node_b

        Example:
            >>> dom_tree = DominatorTree(dag)
            >>> if dom_tree.dominates(5, 10):
            ...     print("Node 5 dominates node 10")
        """
        if node_a == node_b:
            return True

        # Walk up from B to root
        current = node_b
        while current is not None:
            if current == node_a:
                return True
            current = self.idom.get(current)

        return False

    def get_dominators(self, node: int) -> Set[int]:
        """
        Get all dominators of a node.

        Returns the set of all nodes that dominate the given node.

        Args:
            node: Node to query

        Returns:
            Set of all dominators (including the node itself)

        Example:
            >>> dom_tree = DominatorTree(dag)
            >>> dominators = dom_tree.get_dominators(10)
            >>> print(f"Dominators of 10: {dominators}")
        """
        if node in self._dom_set_cache:
            return self._dom_set_cache[node]

        dominators = {node}
        current = node

        while current is not None:
            dominators.add(current)
            current = self.idom.get(current)

        self._dom_set_cache[node] = dominators
        return dominators

    def get_dominated_region(self, node: int) -> Set[int]:
        """
        Get all nodes dominated by the given node.

        Returns the set of all nodes in the subtree rooted at node.

        Args:
            node: Root of the region

        Returns:
            Set of all dominated nodes (including the node itself)

        Example:
            >>> dom_tree = DominatorTree(dag)
            >>> region = dom_tree.get_dominated_region(5)
            >>> print(f"Nodes dominated by 5: {region}")
        """
        region = set()

        def dfs(n: int):
            region.add(n)
            for child in self.children.get(n, []):
                dfs(child)

        dfs(node)
        return region

    def _compute_dominator_paths(
        self, virtual_entry: int, virtual_exit: int
    ) -> List[int]:
        """
        Extract the dominator path from source to sink.

        The dominator path is a sequence of nodes ⟨t₁, t₂, ..., tₙ⟩ where:
        - t₁ is the source (entry node)
        - tₙ is the sink (a node that dominates no other nodes, or has no successors)
        - tᵢ dominates tᵢ₊₁ for all i

        Args:
            dag: The dataflow DAG
            dom_tree: The dominator tree for the DAG

        Returns:
            List of node IDs forming the dominator path
        """
        # Walk from sink to source along dominator tree.
        #
        # A node can be absent from idom rather than merely have None there: if
        # every sink was filtered out as unreachable, nothing points at the
        # virtual exit, so it never appears in the reverse postorder and never
        # gets an immediate dominator. That is a graph with no path from source
        # to sink, which has no dominator path -- not an error.
        current = virtual_exit
        path = [current]
        visited = set()

        while current not in visited:
            visited.add(current)
            current = self.idom.get(current)
            if current is None:
                break
            path.append(current)

        # Reverse to get source-to-sink order
        path.reverse()

        return path
