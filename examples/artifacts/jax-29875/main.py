"""JAX issue 29875: a depthwise 1D convolution, against torch's."""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

import magneton
from magneton import eprof

BATCH, DIM, LENGTH = 128, 256, 512
ROUNDS = 50
# Fewer, because the recorded pass waits on every primitive.
RECORD_ROUNDS = 10


def jax_conv1d(inputs, kernel, bias=None):
    """The implementation from the issue, unchanged."""
    output = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NLC", "LIO", "NLC"),
        feature_group_count=inputs.shape[-1],
    )
    if bias is not None:
        output = output + bias
    return output


def build():
    """One convolution, expressed twice."""
    conv = torch.nn.Conv1d(DIM, DIM, 3, padding=1, groups=DIM).cuda()
    torch_input = torch.randn(BATCH, DIM, LENGTH, device="cuda")

    # torch weight is (out, in/groups, width); jax wants (width, in, out).
    kernel = jnp.asarray(conv.weight.detach().permute(2, 1, 0).cpu().numpy())
    bias = jnp.asarray(conv.bias.detach().cpu().numpy())
    # torch lays a signal out as (N, C, L); jax as (N, L, C).
    jax_input = jnp.asarray(torch_input.permute(0, 2, 1).cpu().numpy())

    return conv, torch_input, jax_input, kernel, bias


def check_they_agree(conv, torch_input, jax_input, kernel, bias):
    with torch.no_grad():
        expected = conv(torch_input).permute(0, 2, 1).cpu().numpy()
    got = np.asarray(jax.jit(jax_conv1d)(jax_input, kernel, bias))
    worst = float(np.abs(expected - got).max())
    print(f"the two convolutions agree to {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError("they do not compute the same thing; stop here")


def measure_jax(jitted, jax_input, kernel, bias, devices):
    """The fused program, which is what a caller would actually run."""
    jax.block_until_ready(jitted(jax_input, kernel, bias))  # compile first

    torch.cuda.synchronize()
    started = time.perf_counter()
    with eprof.Profiler(backend="jax", devices=devices) as (prof, _):
        for _ in range(ROUNDS):
            jax.block_until_ready(jitted(jax_input, kernel, bias))
    wall_ms = (time.perf_counter() - started) * 1e3 / ROUNDS

    ops = prof.per_hlo_op()
    return wall_ms, ops


def record_jax_side(jax_input, kernel, bias, devices):
    """The same two convolutions again, recorded operation by operation."""
    jax_backend = eprof.EnergyBackend(devices=devices, framework="jax")
    with magneton.record(
        lambda a: jax_conv1d(a, kernel, bias), ([jax_input], {}),
        framework="jax", backend=jax_backend,
    ) as (rec, run_it):
        for _ in range(RECORD_ROUNDS):
            run_it(jax_input)
    return rec.run("jax")


def record_torch_side(conv, torch_input, devices):
    torch_backend = eprof.EnergyBackend(devices=devices)
    with magneton.record(
        conv, ([torch_input], {}), backend=torch_backend, clone_outputs=True
    ) as (rec, compiled):
        for _ in range(RECORD_ROUNDS):
            compiled(torch_input)
    torch.cuda.synchronize()
    return rec.run("torch")


def measure_torch(conv, torch_input, devices):
    with torch.no_grad():
        conv(torch_input)  # warm up outside the measurement
    torch.cuda.synchronize()

    started = time.perf_counter()
    with eprof.Profiler(
        tracing_config=eprof.TracingConfig(record_shapes=True),
        energy_config=eprof.EnergyConfig(
            profile_energy=True, energy_profile_device=devices
        ),
    ) as (prof, _):
        with torch.no_grad():
            for _ in range(ROUNDS):
                conv(torch_input)
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1e3 / ROUNDS

    return wall_ms, prof.per_op_table()


def main():
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    print(f"jax {jax.__version__}, torch {torch.__version__} "
          f"on {jax.devices()[0].device_kind}")
    print(f"depthwise conv1d, batch {BATCH}, {DIM} channels, length {LENGTH}\n")

    conv, torch_input, jax_input, kernel, bias = build()
    check_they_agree(conv, torch_input, jax_input, kernel, bias)

    jax_wall, jax_ops = measure_jax(
        jax.jit(jax_conv1d), jax_input, kernel, bias, devices
    )
    jax_run = record_jax_side(jax_input, kernel, bias, devices)

    torch_wall, torch_rows = measure_torch(conv, torch_input, devices)
    torch_run = record_torch_side(conv, torch_input, devices)

    jax_us = sum(o.gpu_time_ns for o in jax_ops) / 1e3 / ROUNDS
    jax_mj = sum(o.gpu_energy_j for o in jax_ops) * 1e3 / ROUNDS
    torch_us = sum(r.gpu_time_ns for r in torch_rows) / 1e3 / ROUNDS
    torch_mj = sum(r.gpu_energy_j for r in torch_rows) * 1e3 / ROUNDS

    header = f"{'':8} {'wall ms':>9} {'GPU us':>9} {'mJ':>9}"
    print()
    print(header)
    print("-" * len(header))
    print(f"{'torch':8} {torch_wall:9.3f} {torch_us:9.1f} {torch_mj:9.3f}")
    print(f"{'jax':8} {jax_wall:9.3f} {jax_us:9.1f} {jax_mj:9.3f}")

    print(f"\nJAX takes {jax_wall / torch_wall:.2f}x the wall time, "
          f"{jax_us / torch_us:.2f}x the GPU time and "
          f"{jax_mj / torch_mj:.2f}x the energy, for the same convolution.")

    print("\nWhich kernel each one chose:")
    for op in jax_ops[:3]:
        for k in op.kernels:
            print(f"  jax    {k.gpu_energy_j * 1e3 / ROUNDS:8.3f} mJ  "
                  f"{k.gpu_time_us / ROUNDS:8.1f} us  {k.name[:46]}")
    for row in torch_rows[:3]:
        print(f"  torch  {row.gpu_energy_j * 1e3 / ROUNDS:8.3f} mJ  "
              f"{row.gpu_time_ns / 1e3 / ROUNDS:8.1f} us  {row.op_name[:46]}")

    report = magneton.compare.compare(jax_run, torch_run)
    print()
    print(report)

    with open("energy.json", "w") as fh:
        json.dump(
            {
                "rounds": ROUNDS,
                "shape": {"batch": BATCH, "channels": DIM, "length": LENGTH},
                "jax": {
                    "wall_ms": jax_wall, "gpu_us": jax_us, "energy_mj": jax_mj,
                    "per_op": [
                        {"name": o.name,
                         "gpu_us": o.gpu_time_us / ROUNDS,
                         "energy_mj": o.energy_mj / ROUNDS,
                         "kernels": [{"name": k.name,
                                      "gpu_us": k.gpu_time_us / ROUNDS,
                                      "energy_mj": k.gpu_energy_j * 1e3 / ROUNDS}
                                     for k in o.kernels]}
                        for o in jax_ops
                    ],
                },
                "torch": {
                    "wall_ms": torch_wall, "gpu_us": torch_us, "energy_mj": torch_mj,
                    "per_op": [
                        {"name": r.op_name,
                         "gpu_us": r.gpu_time_ns / 1e3 / ROUNDS,
                         "energy_mj": r.gpu_energy_j * 1e3 / ROUNDS}
                        for r in torch_rows
                    ],
                },
            },
            fh, indent=2,
        )
    print("\nWrote energy.json")


if __name__ == "__main__":
    main()
