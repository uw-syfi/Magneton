"""
Example: Dataflow Comparison - Fused vs Separate addmm

This example demonstrates how to use the dataflow recording plugin
to compare two different implementations and verify their outputs are the same.

Based on pytorch-141210: comparing torch.addmm (fused) vs bias + torch.mm (separate).
"""

import os
import torch
from magneton import eprof
import magneton


class FusedAddmmModule(torch.nn.Module):
    """Module using fused torch.addmm operation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mat1, mat2, bias):
        return torch.addmm(bias, mat1, mat2)


class SeparateAddmmModule(torch.nn.Module):
    """Module using separate mm and add operations."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mat1, mat2, bias):
        return bias + torch.mm(mat1, mat2)


def profile_model(model, model_name, mat1, mat2, bias, devices):
    """Profile a model and export its dataflow."""
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
    )
    
    
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config)
    with magneton.record(
        model, ([mat1, mat2, bias], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        output = compiled_model(mat1, mat2, bias)
    
    # Export files
    trace_path = f"trace_{model_name}.json"
    dataflow_path = f"dataflow_{model_name}.json"
    backend.export_chrome_trace(trace_path)
    prof.export_dataflow(dataflow_path)
    
    return output, prof, trace_path, dataflow_path


def print_dataflow_stats(prof, model_name):
    """Print dataflow statistics."""
    print(f"\n{model_name} Dataflow:")
    print(f"  Operations: {len(prof.dataflow_dag.nodes)}")
    print(f"  Tensors: {len(prof.dataflow_dag.tensors)}")
    
    for node_id, node in prof.dataflow_dag.nodes.items():
        print(f"  [{node_id}] {node.node_name}")
        print(f"      Type: {node.op_type}")
        print(f"      Target: {node.target}")
        
        # Print output tensor statistics using tensor IDs
        for i, tensor_id in enumerate(node.output_tensor_ids):
            out_tensor = prof.dataflow_dag.tensors.get(tensor_id)
            if out_tensor is not None and out_tensor.numel() > 0:
                print(f"      Output[{i}]: shape={list(out_tensor.shape)}, "
                      f"mean={out_tensor.float().mean().item():.6f}, "
                      f"std={out_tensor.float().std().item():.6f}")


def main():
    # Problem size (reduced for quick testing)
    data_size = 512
    h1 = 1024
    h2 = 2048
    
    # Create models
    fused_model = FusedAddmmModule()
    separate_model = SeparateAddmmModule()
    
    # Create input tensors (use same seed for reproducibility)
    torch.manual_seed(42)
    bias = torch.randn(h2, device='cuda', dtype=torch.half)
    mat1 = torch.randn(data_size, h1, device='cuda', dtype=torch.half)
    mat2 = torch.randn(h1, h2, device='cuda', dtype=torch.half)
    
    # Get CUDA device IDs
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    print("=" * 80)
    print("Dataflow Comparison: Fused vs Separate addmm")
    print("=" * 80)
    print(f"Problem size: mat1={mat1.shape}, mat2={mat2.shape}, bias={bias.shape}")
    print(f"Expected output shape: ({data_size}, {h2})")
    print(f"Device: {mat1.device}")
    print(f"Dtype: {mat1.dtype}")
    print()
    
    # Profile fused implementation
    print("Profiling fused implementation (torch.addmm)...")
    fused_output, fused_prof, fused_trace, fused_dataflow = profile_model(
        fused_model, "fused", mat1, mat2, bias, devices
    )
    print(f"  ✓ Output shape: {fused_output.shape}")
    print(f"  ✓ Chrome trace: {fused_trace}")
    print(f"  ✓ Dataflow DAG: {fused_dataflow}")
    
    # Profile separate implementation
    print("\nProfiling separate implementation (bias + torch.mm)...")
    separate_output, separate_prof, separate_trace, separate_dataflow = profile_model(
        separate_model, "separate", mat1, mat2, bias, devices
    )
    print(f"  ✓ Output shape: {separate_output.shape}")
    print(f"  ✓ Chrome trace: {separate_trace}")
    print(f"  ✓ Dataflow DAG: {separate_dataflow}")
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("Output Comparison")
    print("=" * 80)
    
    # Check if outputs are close
    are_close = torch.allclose(fused_output, separate_output, rtol=1e-3, atol=1e-3)
    print(f"Outputs are close (rtol=1e-3, atol=1e-3): {are_close}")
    
    if are_close:
        print("✓ Both implementations produce the same results!")
    else:
        print("✗ Outputs differ!")
        diff = (fused_output - separate_output).abs()
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")
    
    # Print output statistics
    print(f"\nFused output:    mean={fused_output.float().mean().item():.6f}, "
          f"std={fused_output.float().std().item():.6f}")
    print(f"Separate output: mean={separate_output.float().mean().item():.6f}, "
          f"std={separate_output.float().std().item():.6f}")
    
    # Print dataflow statistics
    print("\n" + "=" * 80)
    print("Dataflow Statistics")
    print("=" * 80)
    print_dataflow_stats(fused_prof, "Fused")
    print_dataflow_stats(separate_prof, "Separate")
    
    # Compare dataflow structures
    print("\n" + "=" * 80)
    print("Dataflow Structure Comparison")
    print("=" * 80)
    
    assert fused_prof.dataflow_dag is not None
    assert separate_prof.dataflow_dag is not None
    
    print(f"Fused operations:    {len(fused_prof.dataflow_dag.nodes)}")
    print(f"Separate operations: {len(separate_prof.dataflow_dag.nodes)}")
    
    if len(fused_prof.dataflow_dag.nodes) < len(separate_prof.dataflow_dag.nodes):
        print("✓ Fused implementation uses fewer operations (more efficient)")
    elif len(fused_prof.dataflow_dag.nodes) > len(separate_prof.dataflow_dag.nodes):
        print("✓ Separate implementation uses fewer operations")
    else:
        print("= Both implementations use the same number of operations")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Compare Chrome traces:")
    print("     - chrome://tracing -> Load trace_fused.json")
    print("     - chrome://tracing -> Load trace_separate.json")
    print("  2. Compare dataflow DAGs:")
    print("     - diff <(jq -S . dataflow_fused.json) <(jq -S . dataflow_separate.json)")
    print("  3. Analyze operation counts:")
    print("     - cat dataflow_fused.json | jq '.nodes | length'")
    print("     - cat dataflow_separate.json | jq '.nodes | length'")
    print("  4. Check output statistics:")
    # A node stores tensor ids; the statistics are in the shared `tensors`
    # table, keyed by those ids.
    print("     - jq '[.nodes[].output_tensor_ids[]] | map(tostring)"
          " as $ids | .tensors' dataflow_fused.json")
    print("     - jq '.tensors | to_entries[] | {id: .key, mean: .value.mean}'"
          " dataflow_fused.json")
    print("  5. Follow the dataflow edges (producer -> consumer node ids):")
    print("     - cat dataflow_fused.json | jq '.edges'")
    print()


if __name__ == "__main__":
    main()

