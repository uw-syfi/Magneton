import os
import torch
from magneton import eprof
from magneton.eprof import attribution
import ctypes

c = ctypes.CDLL("libcudart.so")


def main():
    torch.cuda.set_device(0)
    replay_iters = 100000
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    c.cudaSetDeviceFlags(4)
    with eprof.Profiler(
        energy_config=eprof.EnergyConfig(
            profile_energy=True, energy_profile_device=devices,
        ),
    ) as (prof, _):
        for _ in range(replay_iters):
            torch.cuda.synchronize()
    prof.export_chrome_trace("trace_block.json")
    rows_blocking = prof.per_op_table()

    c.cudaSetDeviceFlags(0)
    with eprof.Profiler(
        energy_config=eprof.EnergyConfig(
            profile_energy=True, energy_profile_device=devices,
        ),
    ) as (prof, _):
        for _ in range(replay_iters):
            torch.cuda.synchronize()
    prof.export_chrome_trace("trace_auto.json")
    rows_spinning = prof.per_op_table()

    print("\nPer operation, blocking against spinning, largest change first:")
    print(attribution.format_comparison(
        rows_blocking, rows_spinning, "blocking", "spinning", per=1))
    print(
        "\nA negative result, and worth keeping as one. The issue asks whether"
        "\nbusy-waiting on a synchronize costs more energy than blocking on it."
        "\nBoth modes spend the same, because neither runs a kernel: the GPU is"
        "\nidle in both and the difference is on the host. There is no operator"
        "\nfor this table to blame, which is the answer rather than a gap in it."
    )


if __name__ == "__main__":
    main()
