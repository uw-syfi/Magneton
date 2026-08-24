"""Dominator tree construction for dataflow DAGs."""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from magneton.dataflow import DataflowDAG
from magneton.utils.fx_utils import OpType


def build_adjacency_lists(
    dag: DataflowDAG,
    sources: Set[int],
    sinks: Set[int],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Build forward and backward adjacency lists from a DAG."""
    # Build full adjacency lists first
    forward_adj: Dict[int, Set[int]] = {node_id: set() for node_id in dag.nodes}
    backward_adj: Dict[int, Set[int]] = {node_id: set() for node_id in dag.nodes}

    tensor_producers: Dict[int, int] = {}
    for node in dag.nodes.values():
        if node.op_type != OpType.INPUT.value:
            for tensor_id in node.input_tensor_ids:
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
    """Dominator tree for a dataflow DAG."""

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
        """Build dominator tree using Cooper-Harvey-Kennedy algorithm."""
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
        """Main build method."""
        sources = set(self._sources) if self._sources else self._entry_nodes()
        sinks = set(self._sinks) if self._sinks else {max(self.dag.nodes)}

        # Step 1: Handle multiple entry nodes
        adj_list, _ = build_adjacency_lists(self.dag, sources, sinks)
        sources &= adj_list.keys()
        sinks &= adj_list.keys()

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
        """Where to start the walk, when the caller did not say."""
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
        """The source-to-sink chain of the dominator tree."""
        real = set(self.dag.nodes)
        return [n for n in self._dominator_paths if n in real]

    def _compute_reverse_postorder(
        self, entry: int, adj_list: Dict[int, List[int]]
    ) -> List[int]:
        """Compute reverse postorder traversal."""
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
        """Compute immediate dominators using iterative dataflow."""
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
        """Find the lowest common ancestor of two nodes in the dominator tree."""
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
        """Build dominator tree structure from immediate dominators."""
        self.children = {}

        for node, dom in self.idom.items():
            if dom is not None:
                if dom not in self.children:
                    self.children[dom] = []
                self.children[dom].append(node)

    def dominates(self, node_a: int, node_b: int) -> bool:
        """Check if node_a dominates node_b."""
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
        """Get all dominators of a node."""
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
        """Get all nodes dominated by the given node."""
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
        """Extract the dominator path from source to sink."""
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
