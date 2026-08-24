"""vLLM issue 9471: FlashInfer decode with and without tensor cores."""

import argparse
import json
import os
import sys

import torch

from magneton import eprof
from magneton.eprof import attribution
import magneton
from magneton import compare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEASURED_REPEATS = 4


def profile_once(label: str, prefix: str) -> None:
    """Profiles one configuration and writes it out."""
    from main import EngineArgs, SampleRequest, main as build_requests  # noqa: F401

    import dataclasses

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    cores = os.environ.get("VLLM_FLASHINFER_FORCE_TENSOR_CORES", "0")
    print(f"  VLLM_FLASHINFER_FORCE_TENSOR_CORES={cores}")

    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    os.environ["VLLM_USE_V1"] = "0"

    engine = EngineArgs(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        enforce_eager=True,
        dtype="float16",
        gpu_memory_utilization=float(os.environ.get("EPROF_GPU_FRACTION", "0.5")),
    )
    llm = LLM(**dataclasses.asdict(engine))

    prompts = [TokensPrompt(prompt_token_ids=list(range(128)))]
    sampling = SamplingParams(n=1, temperature=1.0, max_tokens=1, ignore_eos=True)

    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    backend = eprof.EnergyBackend(
        devices=devices,
        tracing_config=eprof.config.TracingConfig(
            record_shapes=True, record_stack=False,
            with_modules=False, with_memory=False,
        ),
    )
    profiler = magneton.record(
        runner.model,
        backend=backend,
        dataflow_config=magneton.DataflowConfig(
            record_dataflow=True, clone_outputs=True,
        ),
    )
    runner.model = profiler.compiled_model

    llm.generate(prompts, sampling, use_tqdm=False)

    with profiler as (prof, _):
        for _ in range(MEASURED_REPEATS):
            llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    run = compare.Run.of(prof, label)
    run.save(prefix)
    with open(f"{prefix}_per_op.json", "w") as fh:
        json.dump(attribution.records_to_dicts(backend.per_op_table()), fh, indent=2)
    print(f"  wrote {prefix}_dataflow.json, {prefix}_tensors.pt, {prefix}_cost.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PREFIX",
                        help="profile this configuration and write it there")
    parser.add_argument("--label", help="what to call this side in the report")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="two prefixes written by --save")
    args = parser.parse_args()

    if args.save:
        cores = os.environ.get("VLLM_FLASHINFER_FORCE_TENSOR_CORES", "0")
        label = args.label or ("tensor cores" if cores == "1" else "no tensor cores")
        profile_once(label, args.save)
        return 0

    if args.compare:
        a, b = (compare.Run.load(prefix) for prefix in args.compare)
        report = compare.compare(a, b)
        print(report)

        rows = []
        for prefix in args.compare:
            with open(f"{prefix}_per_op.json") as fh:
                rows.append([attribution.PerOpRecord(**d) for d in json.load(fh)])
        print("\nPer operation, largest change first:")
        print(attribution.format_comparison(
            rows[0], rows[1], a.label, b.label, per=MEASURED_REPEATS))
        with open("comparison.json", "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print("\nWrote comparison.json")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
