"""
Dataflow recording plugin.

This plugin captures input and output tensors for each operation
and builds a dataflow DAG (Directed Acyclic Graph).
"""

import dataclasses
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from magneton.utils.tensor_utils import extract_tensors

from magneton.plugin import OpPlugin
import os


logger = logging.getLogger(__name__)



@dataclasses.dataclass
class NodeExecution:
    """Records one execution of an FX node."""
    
    node_id: int
    """Sequential operation ID"""
    
    node_name: str
    """Operation name (e.g., "add_1")"""
    
    op_type: str
    """Operation type: "call_function", "call_method", or "call_module" """
    
    target: str
    """Operation target (e.g., "torch.ops.aten.add.Tensor")"""
    
    input_tensor_ids: List[int] = dataclasses.field(default_factory=list)
    """List of input tensor IDs (using Python id())"""
    
    output_tensor_ids: List[int] = dataclasses.field(default_factory=list)
    """List of output tensor IDs (using Python id())"""


class DataflowDAG:
    """
    Dataflow DAG (Directed Acyclic Graph).
    
    Represents the flow of tensors through operations.
    """
    
    def __init__(self):
        self.nodes: Dict[int, NodeExecution] = {}
        """Map from node_id to NodeExecution"""
        
        self.tensors: Dict[int, torch.Tensor] = {}
        """Map from tensor id() to actual tensor"""
        
        self.tensor_metadata: Dict[int, Dict] = {}
        """Map from tensor id() to metadata (shape, dtype, device, mean, std)"""
        
        self.edges: List[Tuple[int, int]] = []
        """(producer, consumer) node id pairs. Derived; see build_edges."""
    
    def inputs_of(self, node_id: int) -> List[torch.Tensor]:
        """The tensors a node consumed, in order.

        A node stores ids, not tensors, so that a tensor used by twenty nodes is
        held once and described once. This is the join back.
        """
        return self._resolve(self.nodes[node_id].input_tensor_ids)

    def outputs_of(self, node_id: int) -> List[torch.Tensor]:
        """The tensors a node produced, in order."""
        return self._resolve(self.nodes[node_id].output_tensor_ids)

    def _resolve(self, tensor_ids: List[int]) -> List[torch.Tensor]:
        # A loaded DAG has metadata for every id but tensors only for those that
        # were saved, so a missing one is expected rather than an error.
        return [self.tensors[i] for i in tensor_ids if i in self.tensors]

    def build_edges(self) -> List[Tuple[int, int]]:
        """
        Derive the edges from what each node consumed and produced.
        
        An edge runs from the node that produced a tensor to the node that took
        it as an input. Nothing records that directly -- a node knows only the
        ids of its own inputs and outputs -- so it is recovered by walking the
        nodes in execution order and remembering who last wrote each tensor.
        
        Walking in order, and recording a node's outputs only after reading its
        inputs, gives two things for free: an in-place op that reads and writes
        the same tensor does not become its own predecessor, and every edge runs
        from a lower node id to a higher one.
        """
        producer: Dict[int, int] = {}
        seen = set()
        self.edges = []
        
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            for tensor_id in node.input_tensor_ids:
                src = producer.get(tensor_id)
                if src is None or (src, node_id) in seen:
                    continue
                seen.add((src, node_id))
                self.edges.append((src, node_id))
            # After the inputs: an id reused by an in-place write belongs to
            # this node only for whatever comes next.
            for tensor_id in node.output_tensor_ids:
                producer[tensor_id] = node_id
        
        return self.edges
    
    def _extract_tensor_metadata(self, tensor: torch.Tensor) -> Dict:
        """Extract metadata from a tensor."""
        info = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        
        # Add statistics for floating point tensors
        if tensor.numel() > 0 and tensor.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            try:
                t_float = tensor.float() if tensor.dtype != torch.float32 else tensor
                info["mean"] = float(t_float.mean().item())
                # std of a single element is undefined: torch warns and returns
                # nan, and a nan then fails every comparison it takes part in.
                info["std"] = (
                    float(t_float.std().item()) if t_float.numel() > 1 else 0.0
                )
            except Exception:
                pass
        
        return info
    
    def add_node(self, node: NodeExecution, inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        """
        Add a node and extract tensor IDs and metadata.
        
        Args:
            node: NodeExecution (tensor IDs will be populated)
            inputs: Input tensors
            outputs: Output tensors
        """
        # Extract input tensor IDs and metadata
        node.input_tensor_ids = []
        for tensor in inputs:
            if tensor is not None and isinstance(tensor, torch.Tensor):
                tensor_id = id(tensor)
                node.input_tensor_ids.append(tensor_id)
                self.tensors[tensor_id] = tensor
                # Compute metadata only once, when first seen
                if tensor_id not in self.tensor_metadata:
                    self.tensor_metadata[tensor_id] = self._extract_tensor_metadata(tensor)
        
        # Extract output tensor IDs and metadata
        node.output_tensor_ids = []
        for tensor in outputs:
            if tensor is not None and isinstance(tensor, torch.Tensor):
                tensor_id = id(tensor)
                node.output_tensor_ids.append(tensor_id)
                self.tensors[tensor_id] = tensor
                # Compute metadata only once, when first seen
                if tensor_id not in self.tensor_metadata:
                    self.tensor_metadata[tensor_id] = self._extract_tensor_metadata(tensor)
        
        # Add node to DAG
        self.nodes[node.node_id] = node
    
    def to_dict(self) -> Dict:
        """
        Export to JSON-serializable dictionary.
        
        Returns:
            Dictionary with nodes and tensor metadata
        """
        # Export nodes (tensor IDs already stored in node)
        nodes_data = [
            {
                "id": node.node_id,
                "name": node.node_name,
                "op_type": node.op_type,
                "target": node.target,
                "input_tensor_ids": node.input_tensor_ids,
                "output_tensor_ids": node.output_tensor_ids,
            }
            for node in self.nodes.values()
        ]
        
        # Export tensor metadata (pre-computed during add_node)
        tensors_data = {}
        for tensor_id, metadata in self.tensor_metadata.items():
            tensors_data[str(tensor_id)] = metadata
        
        # Derived rather than stored, so an export is never stale.
        return {
            "nodes": nodes_data,
            "edges": [list(e) for e in self.build_edges()],
            "tensors": tensors_data
        }
    
    def save(self, json_path: str, tensor_path: Optional[str] = None):
        """
        Save DAG to JSON and tensor files.
        
        Args:
            json_path: Path to save JSON metadata
            tensor_path: Path to save tensor values (.pt file).
                        If None, auto-generates path based on json_path.
        """
        
        if tensor_path is None:
            base = os.path.splitext(json_path)[0]
            tensor_path = f"{base}_tensors.pt"
        
        # Save metadata to JSON
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save actual tensors to .pt
        torch.save(self.tensors, tensor_path)
    
    @staticmethod
    def load(json_path: str, tensor_path: Optional[str] = None) -> 'DataflowDAG':
        """
        Load DAG from JSON and tensor files.
        
        Args:
            json_path: Path to JSON metadata file
            tensor_path: Path to tensor values file (.pt).
                        If None, auto-detects based on json_path.
        
        Returns:
            DataflowDAG object with tensors loaded
        """
        
        if tensor_path is None:
            base = os.path.splitext(json_path)[0]
            tensor_path = f"{base}_tensors.pt"
        
        # Load metadata
        with open(json_path) as f:
            data = json.load(f)
        
        # Load tensors
        loaded_tensors = {}
        if os.path.exists(tensor_path):
            loaded_tensors = torch.load(tensor_path, weights_only=False)
        
        dag = DataflowDAG()
        
        # Load tensors using their original IDs (from when DAG was constructed)
        for tensor_id_str, tensor in loaded_tensors.items():
            tensor_id = int(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
            dag.tensors[tensor_id] = tensor
        
        # Load tensor metadata
        tensors_metadata = data.get("tensors", {})
        for tensor_id_str, metadata in tensors_metadata.items():
            tensor_id = int(tensor_id_str) if isinstance(tensor_id_str, str) else tensor_id_str
            dag.tensor_metadata[tensor_id] = metadata
        
        # Reconstruct nodes with original tensor IDs
        for node_data in data["nodes"]:
            node = NodeExecution(
                node_id=node_data["id"],
                node_name=node_data["name"],
                op_type=node_data["op_type"],
                target=node_data["target"],
                input_tensor_ids=node_data.get("input_tensor_ids", []),
                output_tensor_ids=node_data.get("output_tensor_ids", []),
            )
            
            dag.nodes[node.node_id] = node
        
        # Rebuild rather than read back: the nodes are the source of truth, and
        # a file written before edges were exported still loads.
        dag.build_edges()
        return dag


class DataflowPlugin(OpPlugin):
    """
    Plugin that records dataflow (inputs/outputs) for each operation.
    """
    
    def __init__(self, dag: DataflowDAG, clone_outputs: bool = False):
        """
        Initialize dataflow plugin.
        
        Args:
            dag: DataflowDAG to record to
            clone_outputs: Whether to clone output tensors.
        """
        self.dag = dag
        self.clone_outputs = clone_outputs
        self.recorded_ops = set()
        
    @property
    def priority(self) -> int:
        """High priority (execute early)."""
        return 10
    
    def _capture_tensor(
        self,
        t: torch.Tensor,
        clone: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Capture tensor (clone or reference).
        
        Args:
            t: Tensor to capture
            clone: Whether to clone the tensor
        
        Returns:
            Tensor (cloned or referenced) or None if not a tensor
        """
        if not isinstance(t, torch.Tensor):
            return None
        
        # Decide whether to clone or reference
        # Cloning preserves values before replay modifies them
        # Referencing maintains refcount to prevent GC
        return t.detach().clone() if clone else t
    
    def _capture_value(
        self,
        val: Any,
        clone: bool = False
    ) -> Any:
        """
        Capture any value (tensor, list, scalar).
        
        Args:
            val: Value to capture
            clone: Whether to clone tensors
        
        Returns:
            Tensor (or None) for tensors, list for lists, original value for others
        """
        if isinstance(val, torch.Tensor):
            return self._capture_tensor(val, clone=clone)
        elif isinstance(val, dict):
            return {k: self._capture_value(v, clone=clone) for k, v in val.items()}
        elif isinstance(val, tuple):
            return tuple(self._capture_value(v, clone=clone) for v in val)
        elif isinstance(val, list):
            return [self._capture_value(v, clone=clone) for v in val]
        else:
            return val
    
    def before_execute(
        self,
        op_id: int,
        op_name: str,
        args: tuple,
        kwargs: dict
    ) -> dict:
        """
        Capture inputs before execution.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Context dict with captured inputs
        """
        if op_id in self.recorded_ops:
            # Already recorded this op (e.g., during replay)
            return {"skip_recording": True}
        
        # Capture inputs (never clone inputs)
        input_values = [self._capture_value(arg, clone=False) for arg in args]
        kwarg_values = {k: self._capture_value(v, clone=False) for k, v in kwargs.items()}
        
        return {
            "skip_recording": False,
            "input_values": input_values,
            "kwarg_values": kwarg_values
        }
    
    def after_execute(
        self,
        op_id: int,
        op_name: str,
        output: Any,
        context: dict
    ) -> Any:
        """
        Capture outputs after execution and record to DAG.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            output: Operation output
            context: Context from before_execute
        
        Returns:
            Unchanged output
        """
        if context.get("skip_recording", False):
            return output
        
        # Capture outputs (clone if configured)
        output_values = self._capture_value(output, clone=self.clone_outputs)
        input_tensors = extract_tensors(context["input_values"])
        output_tensors = extract_tensors(output_values)

        execution = NodeExecution(
            node_id=op_id,
            node_name=op_name,
            op_type=context.get("_op_type", "unknown"),
            target=context.get("_op_target", "unknown"),
        )

        self.dag.add_node(execution, input_tensors, output_tensors)
        self.recorded_ops.add(op_id)

        logger.debug(f"Recorded dataflow for op {op_id}: {op_name}")
        
        return output_values
    
    def wrap_execute(
        self,
        op_callable: Callable,
        context: dict
    ) -> Any:
        """
        No wrapping needed for dataflow - just pass through.
        
        Args:
            op_callable: Operation to execute
            context: Context from before_execute
        
        Returns:
            Operation output
        """
        return op_callable()

