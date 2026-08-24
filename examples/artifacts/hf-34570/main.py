"""transformers issue 34570: the wrong eigensolver for a symmetric matrix."""

import json
import os
import time

import torch

import magneton
from magneton import eprof

SIZE = 1024

REPEATS = 10


class Eigvals(torch.nn.Module):
    """The general eigensolver, which is what the issue reports."""

    def forward(self, covariance):
        return torch.linalg.eigvals(covariance).real.sort().values


class Eigvalsh(torch.nn.Module):
    """The symmetric one, which is what the matrix deserves."""

    def forward(self, covariance):
        return torch.linalg.eigvalsh(covariance).sort().values


def measure(model, covariance, devices, repeats):
    """What one solver costs per call: wall clock, GPU time, GPU energy."""
    backend = eprof.EnergyBackend(
        devices=devices,
        tracing_config=eprof.TracingConfig(record_shapes=True),
    )
    with magneton.record(
        model, ([covariance], {}), backend=backend, clone_outputs=True
    ) as (rec, compiled):
        assert compiled is not None
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            compiled(covariance)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - started

    rows = backend.per_op_table()
    return backend, rec, rows, {
        "wall_ms": wall_s * 1e3 / repeats,
        "gpu_us": sum(r.gpu_time_ns for r in rows) / 1e3 / repeats,
        "energy_mj": sum(r.gpu_energy_j for r in rows) * 1e3 / repeats,
        "kernels": sum(r.num_kernels for r in rows) // max(1, repeats),
    }


def main():
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]

    torch.manual_seed(42)
    random_matrix = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float32)
    covariance = torch.mm(random_matrix, random_matrix.t()) / SIZE

    agree = torch.allclose(
        Eigvals()(covariance), Eigvalsh()(covariance), atol=1e-3
    )
    print(f"{SIZE}x{SIZE} covariance; the two solvers agree: {agree}")
    if not agree:
        raise RuntimeError("the two solvers disagree; the comparison below is moot")

    results, runs = {}, []
    for name, model in (("eigvals", Eigvals()), ("eigvalsh", Eigvalsh())):
        backend, rec, rows, totals = measure(model, covariance, devices, REPEATS)
        backend.export_chrome_trace(f"trace_{name}.json")
        # Read the run out before the next one starts.
        runs.append(rec.run(name))
        results[name] = {**totals, "repeats": REPEATS, "top_ops": [
            {"op": r.op_name, "gpu_us": r.gpu_time_ns / 1e3 / REPEATS,
             "energy_mj": r.gpu_energy_j * 1e3 / REPEATS}
            for r in rows[:5]
        ]}

    a, b = results["eigvals"], results["eigvalsh"]
    header = f"{'':10} {'wall ms':>10} {'GPU us':>11} {'GPU mJ':>9} {'kernels':>9}"
    print()
    print(header)
    print("-" * len(header))
    for name in ("eigvals", "eigvalsh"):
        r = results[name]
        print(f"{name:10} {r['wall_ms']:10.2f} {r['gpu_us']:11.1f} "
              f"{r['energy_mj']:9.2f} {r['kernels']:9d}")

    print(
        f"\nPer call, eigvalsh finishes in {a['wall_ms'] / b['wall_ms']:.0f}x less "
        f"time for the same eigenvalues."
    )
    print(
        "\nThe GPU columns say why, and it is not what the issue's wall clock\n"
        f"suggests. eigvals launches {a['kernels']} kernels and spends "
        f"{a['gpu_us']:.0f} us on the GPU, against\n"
        f"eigvalsh's {b['kernels']} kernels and {b['gpu_us']:.0f} us: the general "
        "solver is not slow because it\ndoes more GPU work, it is slow because it "
        "does almost none. torch has no\nCUDA path for non-symmetric eigenvalues, "
        "so the matrix is copied to the host,\nfactorised there, and copied back -- "
        "which is why its GPU energy is the\nlower of the two while its latency is "
        f"{a['wall_ms'] / b['wall_ms']:.0f}x the worse.\n"
        "\nSo the fix is worth more than an energy number taken on the GPU alone\n"
        "would show: it moves the work onto the device at all."
    )

    print("\nWhat eigvals does on the GPU:")
    for op in a["top_ops"]:
        print(f"  {op['energy_mj']:8.3f} mJ  {op['gpu_us']:9.1f} us  {op['op'][:50]}")

    report = magneton.compare.compare(*runs)
    print()
    print(report)

    with open("comparison.json", "w") as fh:
        json.dump({"totals": results, "regions": report.to_dict()}, fh, indent=2)
    print("\nWrote comparison.json, trace_eigvals.json and trace_eigvalsh.json")


if __name__ == "__main__":
    main()
