import torch
from magneton import eprof
from magneton.eprof import attribution
import magneton
import json
import os

# Repeated inside the measured region so the power sampler has something to
# integrate over: NVML reports at roughly a kilohertz. Both sides repeat equally.
MEASURED_REPEATS = 40

# Looser than the default 1e-3 because the two sides do not compute the same
# numbers: TF32 keeps about ten bits of mantissa against FP32's twenty-four.
# The comparison is of a tensor's mean and standard deviation, which average the
# per-element error down, so this is far more room than the aggregate needs.
PRECISION_TOLERANCE = 0.05


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.matmul

    def forward(self, x, w, y):
        return torch.matmul(x, w, out=y)


def main():
    batch_size, hidden_dim, output_dim = 1024, 16, 1024
    x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32) 
    w = torch.randn(hidden_dim, output_dim, device="cuda", dtype=torch.float32)
    y = x @ w

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=True,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,     
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )

    model = Matmul()
    torch.set_float32_matmul_precision("high")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        model, ([x, w, y], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            compiled_model(x, w, y)
    backend.export_chrome_trace("allow_tf32.json")
    rows_tf32 = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_0 = magneton.compare.Run.of(prof, "tf32")

    torch.set_float32_matmul_precision("highest")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        model, ([x, w, y], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            compiled_model(x, w, y)
    backend.export_chrome_trace("disable_tf32.json")
    rows_fp32 = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_1 = magneton.compare.Run.of(prof, "fp32")

    print("\nPer operation, tf32 against fp32, largest change first:")
    print(attribution.format_comparison(
        rows_tf32, rows_fp32, "tf32", "fp32", per=MEASURED_REPEATS))

    report = magneton.compare.compare(run_0, run_1, stat_tolerance=PRECISION_TOLERANCE)
    print()
    print(report)
    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json")



if __name__ == "__main__":
    main()
