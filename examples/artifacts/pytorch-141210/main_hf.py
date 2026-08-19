import os
import logging
import torch
from transformers import GPT2LMHeadModel

from magneton import eprof
from magneton.eprof import attribution
import magneton

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
        max_num_replay_ops=65,
    )

    input_ids = torch.randint(0, 50257, (8, 1024), device="cuda")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        original_model, ([input_ids], {}),
        backend=backend,
        record_dataflow=False,
    ) as (prof, compiled_model):
        _ = compiled_model(input_ids)
    backend.export_chrome_trace("trace.json")
    print("\ngpt2, per operation:")
    print(attribution.format_table(backend.per_op_table(), top=12, per=1))


if __name__ == "__main__":
    main()
