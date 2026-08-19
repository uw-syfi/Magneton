# Magneton python tests

pytest tests for magneton: recording a dataflow graph, matching two of them,
and the energy backend underneath. eprof is the energy part rather than the
whole, so tests that need the extension say so and skip without it.

## Test Files

### 1. `test_pytorch_76012.py`
Tests custom Triton-based LayerNorm vs PyTorch's built-in LayerNorm with replay enabled.

**Features tested:**
- Custom CUDA kernels with Triton
- Replay plugin for energy measurement
- Comparison between custom and standard implementations

**Run:**
```bash
pytest tests/python/test_pytorch_76012.py -v
```

### 2. `test_pytorch_141210.py`
Tests fused vs separate addmm operations with replay.

**Features tested:**
- Fused operations (`torch.addmm`)
- Separate operations (`bias + torch.mm`)
- Replay plugin with auto-tuning
- Energy comparison between implementations

**Run:**
```bash
pytest tests/python/test_pytorch_141210.py -v
```

### 3. `test_pytorch_141210_hf.py`
Tests HuggingFace GPT2 model with replay.

**Features tested:**
- Real-world transformer model (GPT2)
- Replay plugin with large models
- With and without replay comparison

**Requirements:**
- `transformers` library
- GPT2 model will be downloaded on first run

**Run:**
```bash
pytest tests/python/test_pytorch_141210_hf.py -v
```

### 4. `test_dataflow.py` ⭐ **New Plugin System**
Comprehensive tests for the new dataflow recording plugin.

**Features tested:**
- Dataflow recording (standalone)
- Dataflow + Replay (combined plugins)
- DAG construction and edge building
- Tensor preservation with cloning
- JSON export functionality
- Error handling

**Run:**
```bash
pytest tests/python/test_dataflow.py -v
```

## Running All Tests

```bash
# Run all Python tests
pytest tests/python/ -v

# Run with coverage
pytest tests/python/ -v --cov=eprof --cov-report=html

# Run specific test
pytest tests/python/test_dataflow.py::test_dataflow_only -v

# Run tests matching pattern
pytest tests/python/ -k "dataflow" -v
```

## Requirements

- PyTorch with CUDA support
- magneton installed
- the `magneton-eprof` extension, for anything measuring energy
- pytest
- Optional: transformers (for GPT2 tests)
- Optional: triton (for custom kernel tests)

## Test Structure

All tests follow pytest conventions:
- Test functions start with `test_`
- Use `@pytest.mark.skipif` for conditional tests
- Use fixtures where appropriate
- Include assertions to verify behavior

## Key Differences from Examples

The tests are adapted from the `examples/` directory with these changes:

1. **Pytest format**: Functions instead of main scripts
2. **Reduced sizes**: Smaller models/inputs for faster testing
3. **Assertions**: Verify outputs and profiler state
4. **Cleanup**: Temporary files are properly cleaned up
5. **Skip markers**: Tests skip gracefully when requirements are missing

## Plugin System Tests

The `test_dataflow.py` file specifically demonstrates the new pluggable architecture:

- **Dataflow Plugin**: Records input/output tensors and builds a DAG
- **Replay Plugin**: Re-executes operations for accurate energy measurement
- **Combined**: Both plugins working together with proper tensor cloning

### Example: recording alone

No backend, so no measurement and nothing from eprof.

```python
with magneton.record(model, ([x], {})) as (rec, compiled):
    compiled(x)

rec.export_dataflow("dataflow.json")
```

### Example: recording, with energy

The backend supplies the replay plugin, which re-runs each operation until
NVML has more than one power sample to integrate between.

```python
backend = magneton.EnergyBackend(
    devices=[0],
    replay_config=eprof.ReplayConfig(replay=True, replay_rounds="auto"),
)
with magneton.record(
    model, ([x], {}), backend=backend, clone_outputs=True,
) as (rec, compiled):
    compiled(x)

run = rec.run("mine")          # graph, tensors and cost together
backend.export_chrome_trace("trace.json")
```

## Continuous Integration

These tests are designed to run in CI environments:
- GPU tests are skipped when CUDA is unavailable
- Optional dependencies are handled gracefully
- Reduced problem sizes for faster execution
- Temporary files are cleaned up

## Troubleshooting

**CUDA not available:**
- Tests will be skipped automatically
- Use `pytest -v` to see skip reasons

**Transformers not installed:**
- `test_pytorch_141210_hf.py` will be skipped
- Install with: `pip install transformers`

**Triton not installed:**
- `test_pytorch_76012.py` will fail
- Install with: `pip install triton`

**Out of memory:**
- Reduce batch sizes in test files
- Use smaller models
- Set `CUDA_VISIBLE_DEVICES=0` to use single GPU

