"""
Test for pytorch-141210 with HuggingFace GPT2: Replay with transformers model.

This test profiles a GPT2 model with replay enabled, demonstrating
the plugin system with a real-world transformer model.
"""

import os
import pytest
import torch

from magneton import eprof
import magneton
from magneton.eprof import EnergyBackend

# An optional dependency (the `test` extra) that also wants network access to
# fetch the checkpoint. Imported through importorskip so that not having it
# skips this module rather than erroring collection for the whole suite.
transformers = pytest.importorskip("transformers")
GPT2LMHeadModel = transformers.GPT2LMHeadModel


def test_gpt2_with_replay():
    """Test GPT2 model with replay enabled."""
    # Use smaller model and inputs for testing
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(
        dtype=torch.float32, device="cuda"
    )
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
        max_num_replay_ops=20,  # Reduced for testing
    )
    
    # Smaller input for testing
    input_ids = torch.randint(0, 50257, (2, 256), device="cuda")
    
    with magneton.record(
        model, ([input_ids], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config),
        record_dataflow=False,
    ) as (prof, compiled_model):
        output = compiled_model(input_ids)
    
    # Verify output structure
    assert hasattr(output, 'logits')
    assert output.logits.shape == (2, 256, 50257)
    
    # Verify profiler collected data
    assert prof.backend.profiler.trace is not None


def test_gpt2_without_replay():
    """Test GPT2 model without replay (baseline)."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(
        dtype=torch.float32, device="cuda"
    )
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    # Smaller input for testing
    input_ids = torch.randint(0, 50257, (2, 256), device="cuda")
    
    with magneton.record(
        model, ([input_ids], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config),
        record_dataflow=False,
    ) as (prof, compiled_model):
        output = compiled_model(input_ids)
    
    # Verify output structure
    assert hasattr(output, 'logits')
    assert output.logits.shape == (2, 256, 50257)
    
    # Verify profiler collected data
    assert prof.backend.profiler.trace is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

