"""Test for dataflow recording plugin."""

import os
import json
import tempfile
import pytest
import torch
from magneton import eprof
import magneton
from magneton.eprof import EnergyBackend


class SimpleModel(torch.nn.Module):
    """Simple model for testing dataflow recording."""

    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MatmulModel(torch.nn.Module):
    """Simple matmul model for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        c = torch.matmul(a, b)
        d = torch.relu(c)
        return d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataflow_only():
    """Test dataflow recording without replay."""
    model = MatmulModel()
    
    a = torch.randn(128, 256, device='cuda')
    b = torch.randn(256, 512, device='cuda')
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    with magneton.record(
        model, ([a, b], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config),
    ) as (prof, compiled_model):
        output = compiled_model(a, b)
    
    # Verify output
    assert output.shape == (128, 512)
    
    # Verify dataflow DAG was created
    assert prof.dataflow_dag is not None
    assert len(prof.dataflow_dag.nodes) > 0
    
    # Export and verify dataflow
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        dataflow_path = f.name
    
    try:
        prof.export_dataflow(dataflow_path)
        
        # Verify exported file
        with open(dataflow_path, 'r') as f:
            dataflow_data = json.load(f)
        
        assert 'nodes' in dataflow_data
        assert 'edges' in dataflow_data
        assert len(dataflow_data['nodes']) > 0
        
        # Verify node structure
        for node in dataflow_data['nodes']:
            assert 'id' in node
            assert 'name' in node
            assert 'op_type' in node
            assert 'target' in node
            assert 'input_tensor_ids' in node
            assert 'output_tensor_ids' in node
    finally:
        if os.path.exists(dataflow_path):
            os.remove(dataflow_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataflow_with_replay():
    """Test dataflow recording combined with replay."""
    model = MatmulModel()
    
    a = torch.randn(128, 256, device='cuda')
    b = torch.randn(256, 512, device='cuda')
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.ReplayConfig(
        replay=True,
        replay_rounds=10,  # Small number for testing
        replay_cuda_graph=True,
        max_num_replay_ops=10,
    )
    
    with magneton.record(
        model, ([a, b], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config),
    ) as (prof, compiled_model):
        output = compiled_model(a, b)
    
    # Verify output
    assert output.shape == (128, 512)
    
    # Verify dataflow DAG was created
    assert prof.dataflow_dag is not None
    assert len(prof.dataflow_dag.nodes) > 0
    
    # Export and verify dataflow
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        dataflow_path = f.name
    
    try:
        prof.export_dataflow(dataflow_path)
        
        # Verify exported file
        with open(dataflow_path, 'r') as f:
            dataflow_data = json.load(f)
        
        assert 'nodes' in dataflow_data
        assert 'edges' in dataflow_data
        assert len(dataflow_data['nodes']) > 0
        
        dag = prof.dataflow_dag
        assert any(node.output_tensor_ids for node in dag.nodes.values())
        for node_id, node in dag.nodes.items():
            outputs = dag.outputs_of(node_id)
            assert len(outputs) == len(node.output_tensor_ids), "an output was lost"
            for output_tensor in outputs:
                assert isinstance(output_tensor, torch.Tensor)
                # Tensor should still be valid (not garbage collected)
                assert output_tensor.device.type == 'cuda'
    finally:
        if os.path.exists(dataflow_path):
            os.remove(dataflow_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataflow_edges():
    """Test that dataflow edges are correctly built."""
    model = SimpleModel(256).to('cuda')
    
    x = torch.randn(32, 256, device='cuda')
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    with magneton.record(
        model, ([x], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config),
    ) as (prof, compiled_model):
        compiled_model(x)
    
    # Build edges
    prof.dataflow_dag.build_edges()
    
    # Verify edges exist (operations should be connected)
    assert len(prof.dataflow_dag.edges) > 0
    
    # Verify edge structure
    for src, dst in prof.dataflow_dag.edges:
        assert src in prof.dataflow_dag.nodes
        assert dst in prof.dataflow_dag.nodes
        # Source should come before destination
        assert src < dst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataflow_export_error_without_config():
    """Test that export_dataflow raises error when dataflow not enabled."""
    model = MatmulModel()
    
    a = torch.randn(128, 256, device='cuda')
    b = torch.randn(256, 512, device='cuda')
    
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    
    tracing_config = eprof.TracingConfig(
        record_shapes=True,
        record_stack=False,
    )
    
    with magneton.record(
        model, ([a, b], {}),
        backend=EnergyBackend(devices=devices, tracing_config=tracing_config),
        record_dataflow=False,
    ) as (prof, compiled_model):
        compiled_model(a, b)
    
    # Should raise error when trying to export without dataflow enabled
    with pytest.raises(RuntimeError, match="nothing was recorded"):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            prof.export_dataflow(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

