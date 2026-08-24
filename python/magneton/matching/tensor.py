"""Tensor comparison utilities for finding equivalent tensors."""

from typing import Dict, List, Tuple

import torch
from magneton.dataflow import DataflowDAG
import numpy as np

FP_TYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
INT_TYPES = (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64)

def _get_node_input_specs(dag: DataflowDAG, node) -> List[Tuple[int, Dict]]:
    """Get input tensor metadata for a node."""
    input_specs = []
    for tensor_id in node.input_tensor_ids:
        metadata = dag.tensor_metadata.get(tensor_id)
        if metadata is not None:
            input_specs.append((tensor_id, metadata))
    return input_specs


def _get_node_output_specs(dag: DataflowDAG, node) -> List[Tuple[int, Dict]]:
    """Get output tensor metadata for a node."""
    output_specs = []
    for tensor_id in node.output_tensor_ids:
        metadata = dag.tensor_metadata.get(tensor_id)
        if metadata is not None:
            output_specs.append((tensor_id, metadata))
    return output_specs


def tensors_equivalent(
    spec1: Dict,
    spec2: Dict,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    stat_tolerance: float = 1e-3,
    stats1: Tuple[float, float] = None,
    stats2: Tuple[float, float] = None,
) -> bool:
    """Check if two tensor specifications are equivalent."""
    # Check shape
    if np.prod(tensor1.shape) != np.prod(tensor2.shape):
        return False

    if tensor1.is_complex() != tensor2.is_complex():
        return False
    if tensor1.is_floating_point() != tensor2.is_floating_point():
        return False

    if tensor1.is_complex():
        magnitude1, magnitude2 = tensor1.abs(), tensor2.abs()
        mean1, std1 = magnitude1.mean().item(), magnitude1.std().item()
        mean2, std2 = magnitude2.mean().item(), magnitude2.std().item()
        if abs(mean1 - mean2) > stat_tolerance * (abs(mean1) + abs(mean2) + 1e-8):
            return False
        if abs(std1 - std2) > stat_tolerance * (abs(std1) + abs(std2) + 1e-8):
            return False
        return True

    # Optionally check statistics
    if tensor1.dtype in FP_TYPES:
        if tensor2.dtype not in FP_TYPES:
            return False
        mean1, std1 = stats1 if stats1 is not None else (
            tensor1.float().mean().item(), tensor1.float().std().item())
        mean2, std2 = stats2 if stats2 is not None else (
            tensor2.float().mean().item(), tensor2.float().std().item())
        if abs(mean1 - mean2) > stat_tolerance * (abs(mean1) + abs(mean2) + 1e-8):
            return False
        if abs(std1 - std2) > stat_tolerance * (abs(std1) + abs(std2) + 1e-8):
            return False
    elif tensor1.dtype in INT_TYPES:
        if tensor2.dtype not in INT_TYPES:
            return False
        if tensor1.flatten().ne(tensor2.flatten()).any():
            return False
    else:
        return False
    return True


def find_equivalent_node_pairs(
    dag1: DataflowDAG,
    dag2: DataflowDAG,
    stat_tolerance: float = 1e-3,
) -> List[Tuple[int, int]]:
    """Find all pairs of equivalent nodes between two DAGs."""
    equivalent_pairs = []

    stats: Dict[int, Tuple[float, float]] = {}

    def signature(dag, node) -> Tuple:
        out = []
        for tensor_id in node.output_tensor_ids:
            tensor = dag.tensors.get(tensor_id)
            if tensor is None:
                return ()
            out.append((int(np.prod(tensor.shape)), tensor.dtype in FP_TYPES))
        return tuple(out)

    buckets: Dict[Tuple, List] = {}
    for node2 in dag2.nodes.values():
        buckets.setdefault(signature(dag2, node2), []).append(node2)

    for node1 in dag1.nodes.values():
        candidates = buckets.get(signature(dag1, node1))
        if not candidates:
            continue
        output_specs1 = _get_node_output_specs(dag1, node1)
        for node2 in candidates:
            output_specs2 = _get_node_output_specs(dag2, node2)
            if len(output_specs1) != len(output_specs2):
                continue
            for (tensor_id1, spec1), (tensor_id2, spec2) in zip(output_specs1, output_specs2):
                tensor1 = dag1.tensors[tensor_id1]
                tensor2 = dag2.tensors[tensor_id2]
                if not tensors_equivalent(
                    spec1, spec2, tensor1, tensor2,
                    stat_tolerance=stat_tolerance,
                    stats1=_stats(tensor_id1, tensor1, stats),
                    stats2=_stats(tensor_id2, tensor2, stats),
                ):
                    break
            else:
                equivalent_pairs.append((node1.node_id, node2.node_id))

    return equivalent_pairs


def _stats(tensor_id: int, tensor: torch.Tensor, cache: Dict[int, Tuple[float, float]]):
    """The mean and standard deviation of a tensor, computed once."""
    if tensor.dtype not in FP_TYPES:
        return None
    hit = cache.get(tensor_id)
    if hit is None:
        as_float = tensor.float()
        hit = (as_float.mean().item(),
               as_float.std().item() if as_float.numel() > 1 else 0.0)
        cache[tensor_id] = hit
    return hit
