import json
import os
import torch
import torch.nn.functional as F

from magneton import eprof
from magneton.eprof import attribution
import magneton

# Repeated inside the measured region so the power sampler has something to
# integrate over: NVML reports at roughly a kilohertz, and a single call here is
# a few milliseconds. Both sides repeat equally.
MEASURED_REPEATS = 40


torch.manual_seed(42)


class OriginalCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        token_logprobs = F.log_softmax(logits, dim=-1)
        gathered_logprobs = torch.gather(
            token_logprobs, 2, labels.unsqueeze(-1)
        ).squeeze(-1)
        mask = (labels != -100).float()
        loss = (gathered_logprobs * mask).sum(dim=-1)
        return -loss


class NewCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        token_loss = F.cross_entropy(
            logits.permute(0, 2, 1),
            labels,
            reduction="none",
        )
        loss = token_loss.sum(dim=-1)
        return loss


def main():
    batch_size, seq_len, vocab_size = 2, 512, 50257
    logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda:0")
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0")
    labels[labels % 3 == 0] = -100
    labels_safe = labels.detach()
    labels_safe[labels_safe == -100] = 0

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tracing_config = eprof.config.TracingConfig(
        record_shapes=False,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    # One config for both runs: this compares two cross-entropy implementations,
    # so the profiler has to be set up identically for the numbers to mean
    # anything next to each other.
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        replay_cuda_graph=True,
    )
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        OriginalCrossEntropy(), ([logits, labels], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            compiled_model(logits, labels)

    backend.export_chrome_trace("trace_original.json")
    rows_original = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_0 = magneton.compare.Run.of(prof, "original")

    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config, replay_config=replay_config)
    with magneton.record(
        NewCrossEntropy(), ([logits, labels], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model), torch.no_grad():
        assert compiled_model is not None
        for _ in range(MEASURED_REPEATS):
            compiled_model(logits, labels)
    backend.export_chrome_trace("trace_new.json")
    rows_new = backend.per_op_table()

    # Read out now: the next run resets the buffers this one points at.
    run_1 = magneton.compare.Run.of(prof, "new")

    print("\nPer operation, original against new, largest change first:")
    print(attribution.format_comparison(
        rows_original, rows_new, "original", "new", per=MEASURED_REPEATS))

    report = magneton.compare.compare(run_0, run_1)
    print()
    print(report)
    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json")



if __name__ == "__main__":
    main()
