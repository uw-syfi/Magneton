# Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging

[![NSDI](https://img.shields.io/badge/NSDI-2027-2e7d32.svg)](https://www.usenix.org/conference/nsdi27)
[![arXiv](https://img.shields.io/badge/arXiv-2512.08365-b31b1b.svg)](https://arxiv.org/abs/2512.08365)
[![paper](https://img.shields.io/badge/paper-PDF-blue.svg)](docs/paper.pdf)

**To appear at NSDI '27.**

Yi Pan<sup>1,3,\*</sup>, Wenbo Qian<sup>2,\*</sup>, Dedong Xie<sup>1</sup>,
Ruiyan Hu<sup>1</sup>, Yigong Hu<sup>2</sup>, Baris Kasikci<sup>1</sup>

<sup>1</sup>University of Washington &nbsp;&nbsp;
<sup>2</sup>Boston University &nbsp;&nbsp;
<sup>3</sup>UC Berkeley &nbsp;&nbsp;
<sup>\*</sup>Equal contribution

---

> The training and deployment of machine learning (ML) models have become
> extremely energy-intensive. While existing optimization efforts focus
> primarily on hardware energy efficiency, a significant but overlooked source
> of inefficiency is **software energy waste** caused by poor software design.
> This often includes redundant or poorly designed operations that consume more
> energy without improving performance. These inefficiencies arise in widely
> used ML frameworks and applications, yet developers often lack the visibility
> and tools to detect and diagnose them.
>
> We propose **differential energy debugging**, a novel approach that leverages
> the observation that prominent ML systems often implement similar
> functionality with vastly different energy consumption. Building on this
> insight, we design and implement **Magneton**, an energy profiler that
> compares energy consumption between similar ML systems at the operator level
> and automatically pinpoints code regions and configuration choices
> responsible for excessive energy use.

## The idea in one example

Hugging Face Transformers and vLLM both serve GPT-2, and both compute a GELU.
Transformers reaches it through a custom implementation that launches **five
CUDA kernels**; vLLM uses a fused kernel that cuts the reads from HBM.

Magneton finds this without being told where to look. It records what each
system computed, matches the two graphs by the tensors flowing through them —
not by the names of anything — and reports the operators that correspond along
with what each side spent. The GELU comes back at **77.4% less energy** in
vLLM, worth **12% of end-to-end energy** for the model.

## Results

Applied to **9 popular ML systems** across LLM inference, general ML
frameworks, and image generation:

| | |
|---|---|
| known energy inefficiencies diagnosed | **16** |
| previously unknown issues discovered | **8** |
| of those confirmed by developers | **7** |
| tracing overhead | 4.4% (Transformers), 5.9% (vLLM) |
| offline diagnosis | under 2 minutes, all cases |

A few of the diagnoses, reproduced in `examples/`:

- **Hugging Face #14450** — TF32 is worth 2.2× the time and 2.7× the energy,
  and Magneton locates it on `aten::addmm` inside a 709-operation transformer
- **PyTorch #141822** — the cost is `aten::_log_softmax`, 24 mJ against 7370 mJ;
  the issue report blamed `cross_entropy`, one level up
- **PyTorch discussion #181115** — a rank waiting for its peers burns 131 J
  in the collective, the largest single line in the run
- **JAX #28614** — a batched STFT spends 96.6% of its energy on a convolution
  and 1% on the transform it exists to compute

## Getting started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is not installed
git clone --recursive https://github.com/uw-syfi/Magneton.git && cd Magneton
uv venv --python 3.10 && source .venv/bin/activate
uv pip install ./python
```

That installs `magneton`, which is pure Python and records, matches and
compares with nothing but PyTorch.

Energy measurement is an extra, carrying a Rust and C++ extension built against
CUPTI and NVML. It is built in place, so the machine needs Rust, a CUDA toolkit
and clang 18 or newer. Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

For clang 18 and the CUDA toolkit see
[docs/ARTIFACT_EVALUATION.md](docs/ARTIFACT_EVALUATION.md). Then:

```bash
uv pip install torch==2.8.0
uv pip install maturin setuptools wheel cmake ninja pybind11
EPROF_USE_PREBUILT=0 uv pip install --no-build-isolation './python[energy]'
```

Install torch first and pin it: the extension links the torch that will import
it, which is also why the build cannot be isolated. `magneton` requires
`torch>=2.7,<2.11`.

`scripts/ae.py` does all of this for you, and `scripts/ae.py doctor` checks a
machine for the toolchain first. See
[docs/ARTIFACT_EVALUATION.md](docs/ARTIFACT_EVALUATION.md).

### Comparing two systems

```python
import magneton
from magneton import eprof

with magneton.record(model_a, ([x], {}),
                     backend=eprof.EnergyBackend(devices=[0])) as (rec, run):
    run(x)
a = rec.run("transformers")

with magneton.record(model_b, ([x], {}),
                     backend=eprof.EnergyBackend(devices=[0])) as (rec, run):
    run(x)
b = rec.run("vllm")

print(magneton.compare.compare(a, b))
```

The two systems need not be written in the same framework. Magneton records
PyTorch, JAX and TensorFlow programs into the same representation and matches
them on the values that flow between operations:

```python
magneton.record(fn, ([x], {}), framework="jax")
magneton.record(fn, ([x], {}), framework="tf")
```

## Reproducing the results

`examples/` contains 25 reproductions of reported issues across PyTorch,
Hugging Face, vLLM, Megatron-LM, Stable Diffusion, diffusers, JAX and
TensorFlow. Their requirements contradict each other, so one script builds an
environment per group and runs everything:

```bash
scripts/ae.py
```

Roughly 30–45 minutes end to end. See
[docs/ARTIFACT_EVALUATION.md](docs/ARTIFACT_EVALUATION.md) for what each
environment is for, how to read the output, and how to run a single case.

## Repository layout

| | |
|---|---|
| `python/magneton/` | recording, semantic matching, comparison |
| `python/magneton/eprof/` | the energy profiler: capture and attribution |
| `lib/` | its Rust and C++ half — libkineto, CUPTI, NVML |
| `examples/` | the 25 reproduced issues |
| `docs/` | the paper, and the artifact evaluation guide |

## Citation

```bibtex
@inproceedings{pan2027magneton,
  title     = {Magneton: Optimizing Energy Efficiency of ML Systems via
               Differential Energy Debugging},
  author    = {Pan, Yi and Qian, Wenbo and Xie, Dedong and Hu, Ruiyan and
               Hu, Yigong and Kasikci, Baris},
  booktitle = {24th USENIX Symposium on Networked Systems Design and
               Implementation (NSDI 27)},
  year      = {2027},
  publisher = {USENIX Association},
}
```
