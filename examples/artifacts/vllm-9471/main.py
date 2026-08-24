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

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

MODEL = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
INPUT_LEN = 1
OUTPUT_LEN = 2
NUM_PROMPTS = 2048


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
        record_dataflow=False,
    ) as (prof, compiled_model):
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model = compiled_model
        start = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=True)
        end = time.perf_counter()
    
    assert output_trace is not None
    backend.export_chrome_trace(output_trace)

    print("\nvLLM decode, per operation:")
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
                candidate_ids = candidate_ids[:diff] or candidate_ids[:1]
        
        assert candidate_prompt != ""
        requests.append(
            SampleRequest(
                prompt=candidate_prompt,
                prompt_len=INPUT_LEN,
                expected_output_len=OUTPUT_LEN,
            )
        )

    use_tensor_core = os.environ.get("VLLM_FLASHINFER_FORCE_TENSOR_CORES", "0")
    elapsed_time = run_vllm(requests, args, f"trace_{use_tensor_core}.json")
    total_num_tokens = sum(
        request.prompt_len + request.expected_output_len for request in requests
    )
    total_output_tokens = sum(request.expected_output_len for request in requests)
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )


if __name__ == "__main__":
    args = EngineArgs(model=MODEL, tokenizer=MODEL, enforce_eager=True)
    main(args)
