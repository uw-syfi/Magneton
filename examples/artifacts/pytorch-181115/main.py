from contextlib import nullcontext
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

from magneton import eprof
from magneton.eprof import attribution

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 10


class CustomModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        ] * num_layers
        )
        self.output_proj = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


def warmup(rank):
    x = torch.randn(4096, 4096, device=f"cuda:{rank}")
    y = x @ x
    dist.all_reduce(y)


def worker_normal(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    input_dim = 128
    hidden_dim = 768
    batch_size = 128
    model = CustomModel(input_dim, hidden_dim, 1, 1024).to(f"cuda:{rank}")
    ddp_model = DDP(model, device_ids=[rank])
    inputs = [
        torch.randn(batch_size, input_dim, device=f"cuda:{rank}")
        for _ in range(NUM_INPUTS + rank * 3)
    ]

    for _ in range(10):
        warmup(rank)
    dist.barrier()

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=True,
        with_modules=False,
        with_memory=False,
    )
    energy_config = eprof.config.EnergyConfig(
        profile_energy=True,
        energy_profile_device=devices,
    )

    profiler = (
        eprof.Profiler(
            activities=[eprof.ActivityType.CPU, eprof.ActivityType.CUDA],
            tracing_config=tracing_config,
            energy_config=energy_config,
        )
        if rank == 0
        else nullcontext()
    )

    with profiler:
        num_inputs = 0
        for i in range(NUM_INPUTS):
            num_inputs += 1
            with torch.profiler.record_function(f"forward_{i}"):
                loss = ddp_model(inputs[i]).sum()
            with torch.profiler.record_function(f"backward_{i}"):
                loss.backward()

        for i in range(NUM_INPUTS, NUM_INPUTS + rank * 3):
            num_inputs += 1
            with torch.profiler.record_function(f"forward_{i}"):
                loss = model(inputs[i]).sum()
            with torch.profiler.record_function(f"backward_{i}"):
                loss.backward()

    if not isinstance(profiler, nullcontext):
        profiler.export_chrome_trace("normal.json")
        print(f"\nRank {rank}, per operation, balanced:")
        print(attribution.format_table(profiler.per_op_table(), top=10))


    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")
    dist.destroy_process_group()


def worker_uneven(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    input_dim = 128
    hidden_dim = 768
    batch_size = 128
    model = CustomModel(input_dim, hidden_dim, 1, 1024).to(f"cuda:{rank}")
    ddp_model = DDP(model, device_ids=[rank])
    inputs = [
        torch.randn(batch_size, input_dim, device=f"cuda:{rank}")
        for _ in range(NUM_INPUTS + rank * 3)
    ]

    for _ in range(10):
        warmup(rank)
    dist.barrier()

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=True,
        with_modules=False,
        with_memory=False,
    )
    energy_config = eprof.config.EnergyConfig(
        profile_energy=True,
        energy_profile_device=devices,
    )

    profiler = (
        eprof.Profiler(
            activities=[eprof.ActivityType.CPU, eprof.ActivityType.CUDA],
            tracing_config=tracing_config,
            energy_config=energy_config,
        )
        if rank == 0
        else nullcontext()
    )

    num_inputs = 0
    with profiler:
        with Join([ddp_model]):
            for i, input in enumerate(inputs):
                num_inputs += 1
                with torch.profiler.record_function(f"forward_{i}"):
                    loss = ddp_model(input).sum()
                with torch.profiler.record_function(f"backward_{i}"):
                    loss.backward()

    if not isinstance(profiler, nullcontext):
        profiler.export_chrome_trace("uneven.json")
        print(f"\nRank {rank}, per operation, imbalanced:")
        print(attribution.format_table(profiler.per_op_table(), top=10))


    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")
    dist.destroy_process_group()


def main():
    mp.spawn(worker_normal, nprocs=WORLD_SIZE, join=True) # type: ignore
    mp.spawn(worker_uneven, nprocs=WORLD_SIZE, join=True) # type: ignore


if __name__ == "__main__":
    main()
