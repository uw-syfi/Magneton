"""vLLM vs HuggingFace Dataflow Comparison"""

import dataclasses
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs

import time

from magneton import eprof
import magneton
from magneton.matching import MatchConfig, match_graphs


MODEL_NAME = os.environ.get("EPROF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
GPU_FRACTION = float(os.environ.get("EPROF_GPU_FRACTION", "0.5"))
INPUT_LENGTH = 128
BATCH_SIZE = 1


def create_inputs(tokenizer, length=128):
    """Create reproducible inputs for both models."""
    torch.manual_seed(42)
    input_ids = torch.randint(100, tokenizer.vocab_size - 100, (1, length), device="cuda")
    return input_ids


def load_hf_model(model_name):
    """Load HuggingFace model."""
    print(f"  Loading HuggingFace {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(dtype=torch.float16, device="cuda")
    model.eval()
    print("  ✓ Model loaded")
    return model


def profile_hf(model, input_ids, devices):
    """Profile HuggingFace model with dataflow recording."""
    print("  Configuring profiler...")
    
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )
    
    
    print("  Profiling forward pass...")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config)
    with magneton.record(
        model, ([input_ids], {}),
        backend=backend,
        dataflow_config=dataflow_config,
    ) as (prof, compiled_model):
        output = compiled_model(input_ids)
    
    print("  ✓ Profiling complete")
    return output, prof


def load_vllm_model(model_name, tokenizer, input_ids):
    """Load vLLM model and create generation requests."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.config import CompilationConfig
    
    print(f"  Loading vLLM {model_name}...")
    
    # Set environment variables for vLLM
    os.environ["VLLM_USE_V1"] = "0"
    
    # Create engine with minimal configuration
    engine_args = EngineArgs(
        model=model_name,
        tokenizer=model_name,
        enforce_eager=True,
        # max_num_batched_tokens=512,
        # max_num_seqs=1,
        dtype="float16",
        gpu_memory_utilization=GPU_FRACTION,
        compilation_config=CompilationConfig(custom_ops=["none"])
    )
    
    llm = LLM(**dataclasses.asdict(engine_args))
    
    # Use token IDs directly to avoid decode/re-encode mismatch
    # Convert to list of ints
    prompt_token_ids = input_ids[0].cpu().tolist()
    
    # Create vLLM-style prompts with token IDs
    prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
    
    # Sampling params: generate minimal tokens to simulate forward pass
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1,  # Minimal generation, focus on prefill phase
        ignore_eos=True,
    )
    
    print(f"  ✓ Model loaded (input: {len(prompt_token_ids)} tokens)")
    return llm, prompts, sampling_params


def profile_vllm(llm, prompts, sampling_params, devices):
    """Profile vLLM model with dataflow recording."""
    print("  Configuring profiler...")
    
    tracing_config = eprof.config.TracingConfig(
        record_shapes=True,
        record_stack=False,
        with_modules=False,
        with_memory=False,
    )
    
    dataflow_config = magneton.DataflowConfig(
        record_dataflow=True,
        clone_outputs=True,
    )
    
    
    # Extract the model from vLLM engine
    vllm_model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    
    print("  Profiling with lazy compilation through generate() API...")
    backend = eprof.EnergyBackend(devices=devices, tracing_config=tracing_config)
    profiler = magneton.record(
        vllm_model,
        backend=backend,
        dataflow_config=dataflow_config,
    )
    # Give the engine the instrumented model before measuring anything.
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    runner.model = profiler.compiled_model

    print("  Warming up with the instrumentation attached...")
    llm.generate(prompts, sampling_params, use_tqdm=False)

    with profiler as (prof, compiled_model):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    print("  ✓ Profiling complete")
    return outputs, prof


def compare_outputs(hf_output, vllm_output) -> None:
    """Compare outputs from both models."""
    # Extract HF logits
    hf_logits = hf_output.logits if hasattr(hf_output, 'logits') else hf_output
    
    print(f"  HF output shape: {hf_logits.shape}")
    print(f"  HF logits: mean={hf_logits.float().mean().item():.6f}, "
          f"std={hf_logits.float().std().item():.6f}")
    
    # vLLM returns List[RequestOutput]
    assert isinstance(vllm_output, list)
    assert len(vllm_output) == 1
    assert len(vllm_output[0].outputs) == 1
    print(f"  vLLM output: {vllm_output[0].outputs[0].text}")


def compare_dataflows(hf_prof, vllm_prof, hf_per_node, vllm_per_node):
    """Find the smallest pieces of the two graphs that compute the same thing."""
    hf_dag = hf_prof.dataflow_dag
    vllm_dag = vllm_prof.dataflow_dag
    assert hf_dag is not None and vllm_dag is not None

    hf_dag.build_edges()
    vllm_dag.build_edges()
    print(f"  HuggingFace: {len(hf_dag.nodes)} operations, {len(hf_dag.edges)} edges")
    print(f"  vLLM:        {len(vllm_dag.nodes)} operations, {len(vllm_dag.edges)} edges")

    difference = len(vllm_dag.nodes) - len(hf_dag.nodes)
    if difference < 0:
        share = -difference / len(hf_dag.nodes) * 100
        print(f"  vLLM uses {-difference} fewer operations ({share:.1f}% fewer)")
    elif difference > 0:
        print(f"  vLLM uses {difference} more operations")
    else:
        print("  Both use the same number of operations")

    print("\n  Matching subgraphs...")
    started = time.time()
    matches = match_graphs(hf_dag, vllm_dag, MatchConfig(min_subgraph_size=2))
    print(f"  {len(matches)} minimal equivalent subgraph pairs in {time.time() - started:.0f}s")

    if not matches:
        print("  No equivalent regions found.")
        return matches

    covered = len(set().union(*(m.graph1_nodes for m in matches)))
    print(f"  covering {covered} of {len(hf_dag.nodes)} HuggingFace operations")

    report_matched_subgraphs(matches, hf_dag, vllm_dag, hf_per_node, vllm_per_node)
    return matches


def cost_of(dag, node_ids, per_node):
    """What a set of DAG nodes cost: GPU time and energy, summed."""
    gpu_ns = energy_j = cpu_ns = 0.0
    for node_id in node_ids:
        node = dag.nodes.get(node_id)
        if node is None:
            continue
        record = per_node.get(f"forward_{node.node_name}")
        if record is None:
            continue
        gpu_ns += record.gpu_time_ns
        cpu_ns += record.cpu_time_ns
        energy_j += record.gpu_energy_j
    return cpu_ns, gpu_ns, energy_j


def describe(dag, node_ids, limit=3):
    """What a subgraph is, as the operations it performs."""
    seen = []
    for node_id in sorted(node_ids):
        target = str(dag.nodes[node_id].target)
        # `<built-in function add>` and `<built-in method addmm of type object
        # at 0x...>` are what FX records; the address is noise.
        target = target.split(" of type object")[0].strip("<>")
        target = target.replace("built-in function ", "").replace("built-in method ", "")
        target = target.replace("function ", "").split(" at 0x")[0]
        if target and target != "None" and target not in seen:
            seen.append(target)
        if len(seen) == limit:
            break
    return ", ".join(seen) or "?"


def report_matched_subgraphs(matches, hf_dag, vllm_dag, hf_per_node, vllm_per_node):
    """Print every matched pair with what each side cost."""
    print("\n" + "-" * 80)
    print("Matched subgraphs: what each equivalent part costs")
    print("-" * 80)

    rows = []
    for match in matches:
        hf_cost = cost_of(hf_dag, match.graph1_nodes, hf_per_node)
        vllm_cost = cost_of(vllm_dag, match.graph2_nodes, vllm_per_node)
        rows.append((match, hf_cost, vllm_cost))

    # Costliest first: the parts worth looking at are the expensive ones.
    rows.sort(key=lambda r: -r[1][1])

    print(f"    {'ops':>11}  {'HuggingFace':>19}  {'vLLM':>19}   what it computes")
    print(f"    {'hf / vllm':>11}  {'GPU us / mJ':>19}  {'GPU us / mJ':>19}")
    for match, (_, hf_gpu, hf_j), (_, vllm_gpu, vllm_j) in rows:
        hf_size, vllm_size = match.size()
        print(
            f"    {hf_size:4d} / {vllm_size:4d}  "
            f"{hf_gpu / 1e3:11.1f} / {hf_j * 1e3:5.2f}  "
            f"{vllm_gpu / 1e3:11.1f} / {vllm_j * 1e3:5.2f}   "
            f"{describe(hf_dag, match.graph1_nodes)}"
        )

    hf_nodes = set().union(*(m.graph1_nodes for m, _, _ in rows))
    vllm_nodes = set().union(*(m.graph2_nodes for m, _, _ in rows))
    _, hf_gpu, hf_j = cost_of(hf_dag, hf_nodes, hf_per_node)
    _, vllm_gpu, vllm_j = cost_of(vllm_dag, vllm_nodes, vllm_per_node)

    print(f"\n    {'total':>11}  {hf_gpu / 1e3:11.1f} / {hf_j * 1e3:5.2f}  "
          f"{vllm_gpu / 1e3:11.1f} / {vllm_j * 1e3:5.2f}"
          f"   ({len(hf_nodes)} / {len(vllm_nodes)} distinct ops)")
    print("    Rows do not add up to this: regions meet at their cut points, so"
          " a node can belong to two of them.")
    if hf_gpu and hf_j:
        print(f"    Over the matched parts, vLLM takes {vllm_gpu / hf_gpu:.2f}x the "
              f"GPU time and {vllm_j / hf_j:.2f}x the energy of HuggingFace.")


def export_results(hf_prof, vllm_prof):
    """Report what was written."""
    print("\nResults written:")
    for name in ("trace_hf.json", "trace_vllm.json",
                 "dataflow_hf.json", "dataflow_vllm.json"):
        print(f"  ✓ {name}")


def main():
    print("=" * 80)
    print("vLLM vs HuggingFace Dataflow Comparison")
    print("=" * 80)
    
    # Setup
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = create_inputs(tokenizer, INPUT_LENGTH)
    
    print("\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Input length: {INPUT_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("  Device: cuda:0")
    print(f"  Input shape: {input_ids.shape}")
    
    # Profile HuggingFace
    print("\n" + "-" * 80)
    print("Phase 1: HuggingFace Profiling")
    print("-" * 80)
    hf_model = load_hf_model(MODEL_NAME)
    hf_output, hf_prof = profile_hf(hf_model, input_ids, devices)

    hf_prof.backend.export_chrome_trace("trace_hf.json")
    hf_prof.export_dataflow("dataflow_hf.json")
    hf_per_node = hf_prof.costs()
    
    # Profile vLLM
    print("\n" + "-" * 80)
    print("Phase 2: vLLM Profiling")
    print("-" * 80)
    llm, vllm_prompts, sampling_params = load_vllm_model(MODEL_NAME, tokenizer, input_ids)
    vllm_output, vllm_prof = profile_vllm(llm, vllm_prompts, sampling_params, devices)
    vllm_prof.backend.export_chrome_trace("trace_vllm.json")
    vllm_prof.export_dataflow("dataflow_vllm.json")
    vllm_per_node = vllm_prof.costs()
    
    # Compare outputs
    print("\n" + "-" * 80)
    print("Phase 3: Output Comparison")
    print("-" * 80)
    compare_outputs(hf_output, vllm_output)
    # Compare dataflows
    print("\n" + "-" * 80)
    print("Phase 4: Dataflow Comparison")
    print("-" * 80)
    compare_dataflows(hf_prof, vllm_prof, hf_per_node, vllm_per_node)

    print("\n" + "-" * 80)
    print("Phase 5: Export Results")
    print("-" * 80)
    export_results(hf_prof, vllm_prof)


if __name__ == "__main__":
    main()

