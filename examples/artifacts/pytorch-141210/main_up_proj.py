import json
import os
import torch
from magneton import eprof
from magneton.eprof import attribution
import magneton

MEASURED_REPEATS = 40



class FusedAddmmModule(torch.nn.Module):
    def forward(self, mat1, mat2, bias):
        return torch.addmm(bias, mat1, mat2)


class SeparateAddmmModule(torch.nn.Module):
    def forward(self, mat1, mat2, bias):
        return bias + torch.mm(mat1, mat2)


def profile(data_size, h1, h2, dtype=torch.float32):
    fused_model = FusedAddmmModule()
    separate_model = SeparateAddmmModule()

    bias = torch.randn(h2, device="cuda", dtype=dtype)
    mat1 = torch.randn(data_size, h1, device="cuda", dtype=dtype)
    mat2 = torch.randn(h1, h2, device="cuda", dtype=dtype)

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
        max_num_replay_ops=10,
        replay_once_per_op=False,
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        fused_model, ([mat1, mat2, bias], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        for _ in range(MEASURED_REPEATS):
            compiled_model(mat1, mat2, bias)
    backend.export_chrome_trace("trace_up_proj_fused.json")
    rows_fused = backend.per_op_table()
    # Read out now: the next run resets the buffers this one points at.
    run_0 = magneton.compare.Run.of(prof, "fused")

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        separate_model, ([mat1, mat2, bias], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        for _ in range(MEASURED_REPEATS):
            compiled_model(mat1, mat2, bias)
    backend.export_chrome_trace("trace_up_proj_separate.json")
    rows_separate = backend.per_op_table()
    # Read out now: the next run resets the buffers this one points at.
    run_1 = magneton.compare.Run.of(prof, "separate")


    print("\nPer operation, fused against separate, largest change first:")
    print(attribution.format_comparison(
        rows_fused, rows_separate, "fused", "separate", per=MEASURED_REPEATS))

    report = magneton.compare.compare(run_0, run_1)
    print()
    print(report)
    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json")



if __name__ == "__main__":
    # GPT-2 small up_proj (mlp.c_fc) at batch=8, seq=1024, fp32:
    #   mat1 (8*1024, 768), mat2 (768, 3072), bias (3072,)
    profile(data_size=8192, h1=768, h2=3072, dtype=torch.float32)
