"""Test for pytorch-141210: Fused vs Separate addmm operations with replay."""

import os
import pytest
import torch
from magneton import eprof
import magneton
from magneton.eprof import EnergyBackend


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_addmm_with_replay():
    """Test fused addmm with replay enabled."""
    data_size = 1024  # Reduced for testing
    h1 = 2048  # Reduced for testing
    h2 = 4096  # Reduced for testing
    
    fused_addmm_model = FusedAddmmModule()
    
    bias = torch.randn(h2, device='cuda', dtype=torch.half)
    mat1 = torch.randn(data_size, h1, device='cuda', dtype=torch.half)
    mat2 = torch.randn(h1, h2, device='cuda', dtype=torch.half)
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.TracingConfig(
        record_shapes=False,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
        max_num_replay_ops=100,
    )
    
    with magneton.record(
        fused_addmm_model, ([mat1, mat2, bias], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config),
        record_dataflow=False,
    ) as (prof, model):
        output = model(mat1, mat2, bias)
    
    # Verify output shape
    assert output.shape == (data_size, h2)
    
    # Verify profiler collected data
    assert prof.backend.profiler.trace is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_separate_addmm_with_replay():
    """Test separate addmm operations with replay enabled."""
    data_size = 1024  # Reduced for testing
    h1 = 2048  # Reduced for testing
    h2 = 4096  # Reduced for testing
    
    separate_addmm_model = SeparateAddmmModule()
    
    bias = torch.randn(h2, device='cuda', dtype=torch.half)
    mat1 = torch.randn(data_size, h1, device='cuda', dtype=torch.half)
    mat2 = torch.randn(h1, h2, device='cuda', dtype=torch.half)
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.TracingConfig(
        record_shapes=False,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
        max_num_replay_ops=100,
    )
    
    with magneton.record(
        separate_addmm_model, ([mat1, mat2, bias], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config),
        record_dataflow=False,
    ) as (prof, model):
        output = model(mat1, mat2, bias)
    
    # Verify output shape
    assert output.shape == (data_size, h2)
    
    # Verify profiler collected data
    assert prof.backend.profiler.trace is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_vs_separate_comparison():
    """Test that both fused and separate produce similar results."""
    data_size = 512
    h1 = 1024
    h2 = 2048
    
    fused_model = FusedAddmmModule()
    separate_model = SeparateAddmmModule()
    
    bias = torch.randn(h2, device='cuda', dtype=torch.half)
    mat1 = torch.randn(data_size, h1, device='cuda', dtype=torch.half)
    mat2 = torch.randn(h1, h2, device='cuda', dtype=torch.half)
    
    # Run without profiling to compare outputs
    with torch.no_grad():
        fused_output = fused_model(mat1, mat2, bias)
        separate_output = separate_model(mat1, mat2, bias)
    
    scale = separate_output.abs().max().item()
    tolerance = 4 * torch.finfo(torch.half).eps * scale
    difference = (fused_output - separate_output).abs().max().item()
    assert difference <= tolerance, (
        f"fused and separate differ by {difference:.4f}, more than {tolerance:.4f} "
        f"({tolerance / (torch.finfo(torch.half).eps * scale):.1f} ULP at scale {scale:.1f})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

