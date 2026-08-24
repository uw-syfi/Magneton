"""JAX issue 28614: what a short-time Fourier transform costs in energy."""

import json
import os

import jax
import jax.numpy as jnp

from magneton import eprof

SAMPLES = 5000
BATCH = 256
N_FFT = 2048
ROUNDS = 20


def main():
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}")
    print(f"batch {BATCH}, {SAMPLES} samples, n_fft {N_FFT}\n")

    window = jnp.hanning(N_FFT)
    signal = jax.random.normal(jax.random.PRNGKey(0), (BATCH, SAMPLES))

    def stft(x):
        _, _, spectrum = jax.scipy.signal.stft(
            x, window=window, nperseg=N_FFT, noverlap=N_FFT // 2
        )
        return jnp.abs(spectrum)

    def batched(x):
        return jax.vmap(stft)(x).mean()

    jitted = jax.jit(batched)
    jax.block_until_ready(jitted(signal))  # compile outside the measurement

    with eprof.Profiler(backend="jax", devices=devices) as (prof, _):
        for _ in range(ROUNDS):
            jax.block_until_ready(jitted(signal))

    ops = prof.per_hlo_op()
    gpu_ns = sum(o.gpu_time_ns for o in ops)
    energy_j = sum(o.gpu_energy_j for o in ops)
    launches = sum(o.num_launches for o in ops)

    print(f"{launches // ROUNDS} kernel launches per call, "
          f"{gpu_ns / 1e6 / ROUNDS:.2f} ms GPU, "
          f"{energy_j * 1e3 / ROUNDS:.2f} mJ per call\n")
    print("Per JAX operation, largest energy first, with the kernels that ran it:")
    for op in ops:
        share = 100 * op.gpu_energy_j / energy_j if energy_j else 0
        print(f"  {op.energy_mj / ROUNDS:8.3f} mJ  {op.gpu_time_us / ROUNDS:9.1f} us  "
              f"{share:5.1f}%  {op.name}")
        if len(op.kernels) == 1 and op.kernels[0].name == op.name.replace(".", "_"):
            continue
        for kernel in op.kernels:
            print(f"      {kernel.gpu_energy_j * 1e3 / ROUNDS:8.3f} mJ  "
                  f"{kernel.gpu_time_us / ROUNDS:9.1f} us  "
                  f"x{kernel.num_launches // ROUNDS:<3} {kernel.name[:44]}")

    with open("energy.json", "w") as fh:
        json.dump(
            {
                "launches_per_call": launches / ROUNDS,
                "gpu_ms_per_call": gpu_ns / 1e6 / ROUNDS,
                "energy_mj_per_call": energy_j * 1e3 / ROUNDS,
                "per_op": [
                    {"name": o.name,
                     "energy_mj": o.energy_mj / ROUNDS,
                     "gpu_us": o.gpu_time_us / ROUNDS,
                     "launches": o.num_launches // ROUNDS,
                     "share_pct": 100 * o.gpu_energy_j / energy_j if energy_j else 0,
                     "kernels": [
                         {"name": k.name,
                          "energy_mj": k.gpu_energy_j * 1e3 / ROUNDS,
                          "gpu_us": k.gpu_time_us / ROUNDS,
                          "launches": k.num_launches // ROUNDS}
                         for k in o.kernels
                     ]}
                    for o in ops
                ],
            },
            fh, indent=2,
        )
    print("\nWrote energy.json")


if __name__ == "__main__":
    main()
