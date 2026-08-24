"""Benchmark offline inference throughput for vLLM."""

import dataclasses
import random
import os
import time
from typing import Callable, List
import torch
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TextPrompt

from magneton import eprof
from magneton.eprof import attribution
import magneton

# Set by profile_for_comparison; main() alone has no use for a dataflow
# recording and would only pay for it.
_RECORD_DATAFLOW = False
profiler_holder = []


os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

MODEL = "gpt2"
INPUT_LEN = 1024
OUTPUT_LEN = 1
NUM_PROMPTS = 64


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking."""

    prompt: str
    prompt_len: int
    expected_output_len: int


def run_vllm(
    requests: List[SampleRequest],
    engine_args: EngineArgs,
    output_trace: str,
) -> float:
    from vllm import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))

    def wrapper_with_record_function(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(func.__name__):
                return func(*args, **kwargs)

        return wrapper

    llm.llm_engine.step = wrapper_with_record_function(llm.llm_engine.step)

    # Add the requests to the engine.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            )
        )

    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=_RECORD_DATAFLOW,
        clone_outputs=_RECORD_DATAFLOW,
    )
    replay_config = eprof.config.ReplayConfig(
        replay=True,
        replay_rounds="auto",
        max_num_replay_ops=42,
        replay_cuda_graph=True,
    )
    backend = eprof.EnergyBackend(devices=devices, replay_config=replay_config)
    with magneton.record(
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model, None,
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model = compiled_model
        start = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=True)
        end = time.perf_counter()
    
    profiler_holder.append(prof)
    backend.export_chrome_trace(output_trace)

    print("\nvLLM prefill, per operation:")
    print(attribution.format_table(backend.per_op_table(), top=15))
    return end - start


def main(args: EngineArgs):
    random.seed(42)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    vocab_size = tokenizer.vocab_size
    requests = []
    for _ in range(NUM_PROMPTS):
        request_tokenizer = tokenizer
        # Synthesize a prompt with the given input length.
        candidate_ids = [
            random.randint(0, vocab_size - 1) for _ in range(INPUT_LEN)
        ]
        # As tokenizer may add additional tokens like BOS, we need to try
        # different lengths to get the desired input length.
        candidate_prompt = ""
        for _ in range(5):  # Max attempts to correct
            candidate_prompt = request_tokenizer.decode(candidate_ids)
            tokenized_len = len(request_tokenizer.encode(candidate_prompt))

            if tokenized_len == INPUT_LEN:
                break

            # Adjust length based on difference
            diff = INPUT_LEN - tokenized_len
            if diff > 0:
                candidate_ids.extend(
                    [random.randint(100, vocab_size - 100) for _ in range(diff)]
                )
            else:
                candidate_ids = candidate_ids[:diff]
        
        assert candidate_prompt != ""
        requests.append(
            SampleRequest(
                prompt=candidate_prompt,
                prompt_len=INPUT_LEN,
                expected_output_len=OUTPUT_LEN,
            )
        )

    elapsed_time = run_vllm(requests, args, "trace.json")
    total_num_tokens = sum(
        request.prompt_len + request.expected_output_len for request in requests
    )
    total_output_tokens = sum(request.expected_output_len for request in requests)
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )


def profile_for_comparison(label):
    """Profiles this configuration and returns it as a `compare.Run`."""
    from magneton import compare

    global _RECORD_DATAFLOW
    _RECORD_DATAFLOW = True
    profiler_holder.clear()
    main(EngineArgs(model=MODEL, tokenizer=MODEL, enforce_eager=True))
    if not profiler_holder:
        raise RuntimeError("main() did not profile anything")
    return compare.Run.of(profiler_holder[-1], label)


if __name__ == "__main__":
    args = EngineArgs(model=MODEL, tokenizer=MODEL, enforce_eager=True)
    main(args)
