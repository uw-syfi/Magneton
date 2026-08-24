import json
import os
import logging
import torch
from transformers import GPT2LMHeadModel

from magneton import eprof
from magneton.eprof import attribution
import magneton

# Repeated inside the measured region so the power sampler has something to
# integrate over: NVML reports at roughly a kilohertz. Both sides repeat equally.
MEASURED_REPEATS = 40

PRECISION_TOLERANCE = 0.05


logging.basicConfig(level=logging.DEBUG)


def main():
    original_model = GPT2LMHeadModel.from_pretrained("gpt2").to(
        dtype=torch.float32, device="cuda"
    )
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
    )
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
        max_num_replay_ops=60,
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )

    input_ids = torch.randint(0, 50257, (1, 1024), device="cuda")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        original_model, ([input_ids], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            _ = compiled_model(input_ids)
    backend.export_chrome_trace("trace.json")
    rows_fp32 = backend.per_op_table()
    # Read out now: the next run resets the buffers this one points at.
    run_0 = magneton.compare.Run.of(prof, "fp32")

    torch.set_float32_matmul_precision("high")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        original_model, ([input_ids], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            _ = compiled_model(input_ids)
    backend.export_chrome_trace("trace_tf32.json")
    rows_tf32 = backend.per_op_table()
    # Read out now: the next run resets the buffers this one points at.
    run_1 = magneton.compare.Run.of(prof, "tf32")


    print("\nPer operation, fp32 against tf32, largest change first:")
    print(attribution.format_comparison(
        rows_fp32, rows_tf32, "fp32", "tf32", per=MEASURED_REPEATS))

    report = magneton.compare.compare(run_0, run_1, stat_tolerance=PRECISION_TOLERANCE)
    print()
    print(report)
    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json")



if __name__ == "__main__":
    main()
