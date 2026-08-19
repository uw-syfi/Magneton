import json
import os
from typing import Tuple

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.optimizer import MegatronOptimizer

from magneton import eprof
import magneton

TRAIN_MODE = False

# How many times the measured region runs the step. NVML reports at about a
# kilohertz and energy is the integral of power between readings, so a single
# forward pass can finish inside one sampling interval and be credited nothing.
MEASURED_REPEATS = 20


class _TraceableFork:
    """What `get_cuda_rng_tracker().fork()` returns once patched.

    megatron guards its dropout with that tracker so that tensor-parallel ranks
    draw different masks. It is a `@contextlib.contextmanager`, and dynamo
    refuses to inline one under `fullgraph=True` -- "SKIPPED INLINING
    <code object __enter__ ... contextlib.py>" -- which is what stopped this
    model being traced at all.

    A context manager written as a class traces without complaint, and with
    `tensor_model_parallel_size=1` there is no second rank for the forked
    stream to differ from: the fork selects which RNG state dropout draws from,
    not whether dropout happens. Checked against the unpatched model, the
    outputs are identical.

    Both places that fork are reached whatever `sequence_parallel` is set to --
    attention forks when it is off, the embedding when it is on -- so patching
    the tracker is the one change that covers both.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_traceable() -> None:
    """Let dynamo trace megatron's forward as a single graph.

    One patch, and only for dynamo's benefit. The model itself is untouched:
    the tensor-parallel linear and the region boundaries are megatron's own,
    autograd.Functions and all, and are profiled as they are.
    """
    from megatron.core import parallel_state as mpu_state
    from megatron.core.tensor_parallel import random as mpu_random

    if mpu_state.get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError(
            "the fork is only a no-op at tensor_model_parallel_size=1"
        )

    mpu_random.get_cuda_rng_tracker().fork = lambda *a, **k: _TraceableFork()


def setup_distributed() -> None:
    """Initialize distributed training."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    rng_tracker = get_cuda_rng_tracker()
    rng_tracker.add('model-parallel-rng', 1234)


def create_model() -> GPTModel:
    """Create minimal GPT model using megatron-core."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=512,
        ffn_hidden_size=512,
        num_attention_heads=32,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
    )
    
    layer_spec = get_gpt_layer_local_spec()
    
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=32768,
        max_sequence_length=4096,
        pre_process=True,
        post_process=True,
    ).cuda()
    
    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()
    
    return model


def create_optimizer(model: GPTModel) -> MegatronOptimizer:
    """Create optimizer using megatron-core."""
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[model],
    )
    
    return optimizer


def create_dummy_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a dummy batch of data for training."""
    batch_size = 1
    seq_length = 128
    vocab_size = 32768
    
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long, device='cuda')
    
    position_ids = torch.arange(seq_length, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    attention_mask = torch.tril(torch.ones((seq_length, seq_length), device='cuda', dtype=torch.bool))
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    labels = torch.cat([tokens[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')], dim=1)
    
    return tokens, position_ids, attention_mask, labels


def training_step(model: GPTModel, optimizer: MegatronOptimizer, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
    """Perform one training step."""
    model.train()
    
    tokens, position_ids, attention_mask, labels = batch
    
    output = model(tokens, position_ids, attention_mask)
    
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), 
        labels.view(-1), 
        ignore_index=0
    )

    if TRAIN_MODE:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()


def main() -> None:
    setup_distributed()
    make_traceable()
    model = create_model()
    optimizer = create_optimizer(model)

    model(*create_dummy_batch()[:-1])

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
    )
    # No replay here. It re-runs each operation to stretch the window NVML
    # integrates over, but megatron's decoder reaches its tensor-parallel
    # linear through an autograd.Function, and running that repeatedly inside
    # the graph aborts in C++ with std::bad_alloc. Repeating the whole step
    # gives the sampler the same amount to integrate over, from work the model
    # actually does.
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config)
    with magneton.record(
        model, (create_dummy_batch()[:-1], {}),
        backend=backend,
        record_dataflow=False,
    ) as (prof, compiled_model):
        assert compiled_model is not None
        batch = create_dummy_batch()
        for _ in range(MEASURED_REPEATS):
            training_step(compiled_model, optimizer, batch)
    torch.cuda.synchronize()
    backend.export_chrome_trace("trace.json")

    rows = backend.per_op_table()
    print(f"\n{'GPU us':>10}  {'mJ':>8}  operation")
    for r in rows[:12]:
        print(f"{r.gpu_time_ns / 1e3 / MEASURED_REPEATS:10.1f}  "
              f"{r.gpu_energy_j * 1e3 / MEASURED_REPEATS:8.3f}  {r.op_name[:52]}")
    total_us = sum(r.gpu_time_ns for r in rows) / 1e3 / MEASURED_REPEATS
    total_mj = sum(r.gpu_energy_j for r in rows) * 1e3 / MEASURED_REPEATS
    print(f"\n{total_us:10.1f}  {total_mj:8.3f}  total, per step")

    with open("per_op.json", "w") as fh:
        json.dump(
            [
                {"op": r.op_name, "gpu_us": r.gpu_time_ns / 1e3 / MEASURED_REPEATS,
                 "energy_mj": r.gpu_energy_j * 1e3 / MEASURED_REPEATS,
                 "kernels": r.num_kernels}
                for r in rows
            ],
            fh, indent=2,
        )
    print("Wrote trace.json and per_op.json")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
