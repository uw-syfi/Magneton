import os
from diffusers.models import FluxTransformer2DModel
import torch
from magneton import eprof
from magneton.eprof import attribution
import magneton

# Set by profile_for_comparison; main() alone has no use for a dataflow
# recording and would only pay for it.
_RECORD_DATAFLOW = False
profiler_holder = []


MODEL_ID = "black-forest-labs/FLUX.1-dev"
NUM_LAYERS = 1

def main():
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        out_channels=None,
        num_layers=0,
        num_single_layers=32,
    ).to(dtype=torch.bfloat16, device="cuda")

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=_RECORD_DATAFLOW,
        clone_outputs=_RECORD_DATAFLOW,
    )

    hidden_states = torch.randn(1, 4096, 64, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(
        1, 512, 4096, device="cuda", dtype=torch.bfloat16
    )
    pooled_projections = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)
    timestep = torch.ones(1, device="cuda", dtype=torch.bfloat16)
    img_ids = torch.randn(4096, 3, device="cuda", dtype=torch.bfloat16)
    txt_ids = torch.randn(512, 3, device="cuda", dtype=torch.bfloat16)
    guidance = None
    joint_attention_kwargs = {}
    example_inputs = [
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
    ]

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config)
    with magneton.record(
        model, (example_inputs, {"return_dict": False}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        assert compiled_model is not None
        compiled_model(*example_inputs, return_dict=False)

    profiler_holder.append(prof)
    backend.export_chrome_trace("trace.json")
    print("\npipeline, per operation:")
    print(attribution.format_table(backend.per_op_table(), top=12, per=1))


def profile_for_comparison(label):
    """Profiles this configuration and returns it as a `compare.Run`."""
    from magneton import compare

    global _RECORD_DATAFLOW
    _RECORD_DATAFLOW = True
    profiler_holder.clear()
    main()
    if not profiler_holder:
        raise RuntimeError("main() did not profile anything")
    return compare.Run.of(profiler_holder[-1], label)


if __name__ == "__main__":
    main()
