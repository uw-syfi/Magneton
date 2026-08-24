from contextlib import nullcontext
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

from magneton import eprof

BACKEND = "nccl"
WORLD_SIZE = 2
SEQ_LEN = 1024
VOCAB_SIZE = 50257
EOS_ID = 50256
PAD_ID = 50256
BASE_SEED = 0xE9F0

RANK_PARAMS = {
    0: dict(n_samples=28, mean_log=5.65, sigma_log=0.55),
    1: dict(n_samples=28, mean_log=5.95, sigma_log=0.55),
}


def make_rank_samples(rank):
    p = RANK_PARAMS[rank]
    rng = np.random.default_rng(BASE_SEED + rank)
    lengths = np.clip(
        rng.lognormal(mean=p["mean_log"], sigma=p["sigma_log"], size=p["n_samples"]),
        8,
        SEQ_LEN - 1,
    ).astype(int)
    return [rng.integers(0, VOCAB_SIZE, size=int(L), dtype=np.int64) for L in lengths]


def pack(seqs):
    seqs = sorted(seqs, key=len, reverse=True)
    bins = []
    for s in seqs:
        for b in bins:
            used = sum(len(x) for x in b) + len(b)
            if used + len(s) + 1 <= SEQ_LEN:
                b.append(s)
                break
        else:
            bins.append([s])
    out = []
    for b in bins:
        cat = np.concatenate(
            [np.concatenate([x, np.array([EOS_ID], dtype=np.int64)]) for x in b]
        )
        cat = np.pad(cat, (0, SEQ_LEN - len(cat)), constant_values=PAD_ID)
        out.append(torch.from_numpy(cat).long())
    return out


def build_model(rank):
    config = GPT2Config(
        n_layer=1,
        n_head=12,
        n_embd=768,
        n_positions=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
    )
    return GPT2LMHeadModel(config).to(f"cuda:{rank}")


def warmup(rank):
    x = torch.randn(4096, 4096, device=f"cuda:{rank}")
    y = x @ x
    dist.all_reduce(y)


def _worker(rank, use_join, trace_name):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    torch.manual_seed(BASE_SEED + rank)
    model = build_model(rank)
    ddp_model = DDP(model, device_ids=[rank])

    batches = [t.to(f"cuda:{rank}") for t in pack(make_rank_samples(rank))]
    print(f"Rank {rank}: {len(batches)} packed batches of length {SEQ_LEN}")

    count = torch.tensor([len(batches)], device=f"cuda:{rank}")
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    min_batches = int(count.item())

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
        join_ctx = Join([ddp_model]) if use_join else nullcontext()
        with join_ctx:
            for i, batch in enumerate(batches):
                ids = batch.unsqueeze(0)
                m = ddp_model if (use_join or i < min_batches) else model
                with torch.profiler.record_function(f"forward_{i}"):
                    loss = m(input_ids=ids, labels=ids).loss
                with torch.profiler.record_function(f"backward_{i}"):
                    loss.backward()

    if not isinstance(profiler, nullcontext):
        profiler.export_chrome_trace(trace_name)

    print(f"Rank {rank} finished {len(batches)} batches (use_join={use_join}).")
    dist.destroy_process_group()


def worker_normal(rank):
    _worker(rank, use_join=False, trace_name="normal_gpt.json")


def worker_uneven(rank):
    _worker(rank, use_join=True, trace_name="uneven_gpt.json")


def main():
    mp.spawn(worker_normal, nprocs=WORLD_SIZE, join=True)  # type: ignore
    mp.spawn(worker_uneven, nprocs=WORLD_SIZE, join=True)  # type: ignore


if __name__ == "__main__":
    main()
