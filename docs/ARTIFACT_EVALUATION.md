# Artifact evaluation

Clone with submodules -- the profiler's C++ build needs them:

```sh
git clone --recursive https://github.com/uw-syfi/Magneton.git && cd Magneton
```

If you already cloned without `--recursive`, run `git submodule update --init`.

Everything else is driven by one script:

```sh
scripts/ae.py
```

That checks the machine, builds the environments the examples need, runs every
example, and prints a table of what passed. Nothing else has to be installed or
configured by hand.

If you would rather take it a step at a time:

```sh
scripts/ae.py doctor            # can this machine do it at all?
scripts/ae.py list              # what will run, and in which environment
scripts/ae.py build             # environments only
scripts/ae.py run               # examples only
scripts/ae.py report            # the table again, from the last run
```

and to narrow it:

```sh
scripts/ae.py run --env base
scripts/ae.py run --example pytorch-141210
```

## What the machine needs

`scripts/ae.py doctor` checks all of this and names whatever is missing.

| | |
|---|---|
| an NVIDIA GPU | every example profiles CUDA work; there is no CPU-only path |
| a CUDA toolkit | found via `CUDA_HOME`, `nvcc`, or `/usr/local/cuda` |
| CUPTI | its headers and libraries, from the toolkit or `nvidia-cuda-cupti`; `EPROF_CUPTI_DIR` overrides where to look |
| clang 18 or newer | the C++ uses C++20 `std::format` |
| [uv](https://docs.astral.sh/uv/) | creates the environments |
| [cargo](https://rustup.rs) | builds the profiler |
| a TOML parser | python 3.11+, or `pip install tomli`, to read `examples/manifest.toml` |
| ~60 GiB disk | mostly the frameworks: vLLM, TensorFlow and Megatron are large |

`doctor` runs without the TOML parser, so it can be the thing that tells you
about it.

The reported numbers were measured on an NVIDIA H200. Any recent NVIDIA GPU
should work; absolute figures are specific to the hardware.

### Installing them

uv and Rust:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

On Ubuntu 22.04 the compilers are older than this needs, so they come from
their own repositories:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main"
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install -y clang-18 gcc-13 g++-13 libboost-all-dev
```

The CUDA toolkit and an NVIDIA driver are assumed to be on the machine already;
`scripts/ae.py doctor` reports whether it can find them.

g++-13 is there for its libstdc++ rather than to compile anything: clang picks
up the newest one it finds, and `std::format` needs a recent one. If `clang++-18`
is not the default compiler, point `CXX` at it.

Only the energy extension needs any of this. `pip install ./python` installs
magneton alone, which records, matches and compares with nothing but torch. The
extension is built from `lib/eprof` in this checkout:

```sh
uv venv --python 3.10 && source .venv/bin/activate
uv pip install torch==2.8.0
uv pip install maturin setuptools wheel cmake ninja pybind11
EPROF_USE_PREBUILT=0 uv pip install --no-build-isolation './python[energy]'
```

Both preceding lines are required. The extension links the libtorch that will
import it, so torch must be installed first and the build cannot be isolated;
and because it cannot be isolated, pip installs no build backend either, so the
build dependencies have to be present already. They are declared in
`lib/eprof/pyproject.toml` under `[build-system] requires`, which is the list
`scripts/ae.py` installs into each environment it makes.

Pin torch to the supported range, `torch>=2.7,<2.11`. The build links the CUPTI
that torch's `nvidia-cuda-cupti` dependency provides; outside that range no such
wheel is available and the extension cannot collect kernels.

## What it costs

Roughly 30–45 minutes end to end on a warm network, most of it downloading
frameworks. The profiler itself builds in about 6 minutes the first time and
under a minute for each environment after that.

Two examples are slow by design: `pytorch-181115` runs a distributed workload
for about two minutes, and `sd-279` compiles a Stable Diffusion attention block
for about four.

`pytorch-181115` is the one example that needs two GPUs -- it is about what one
rank costs while it waits for another, which needs two ranks to be true. If you
narrow the run with `CUDA_VISIBLE_DEVICES`, leave it two devices wide, or that
example fails with `invalid device ordinal` while the rest still pass.

## Why there are seven environments

`base`, `hf`, `vllm`, `megatron`, `tf`, `jax` and `diffusers`. The frameworks
the examples exercise have incompatible requirements:

- vLLM 0.9.1 requires `torch==2.7.0` exactly.
- The Hugging Face examples are pinned to `transformers==4.51.3` with torch 2.7.
- Megatron-LM core is pinned to torch 2.7.
- TensorFlow brings its own CUDA stack.
- diffusers is pinned to a specific commit.

Everything else runs on torch 2.8. `scripts/ae.py list` prints the current
split, which it reads from `examples/manifest.toml` along with the environment
each example needs and the artifacts it should produce.

The extension is built once per environment, since it links the libtorch that
will import it; environments sharing a torch version share the build.
`magneton` is pure python and is installed into each environment alongside it.

## Reading the results

The table reports `passed` only if the example exited 0 **and** wrote every
artifact it declares, **and** those artifacts parse — an example can exit 0
without having written anything usable. For a chrome trace the script also
reports how many events and power samples it contains, so an empty trace is
visible rather than merely well-formed.

Per-example output goes to `.ae/logs/<name>.log`, and the whole table is also
written to `.ae/results.json`.

The traces themselves are written next to each example, and can be opened in
`chrome://tracing` or [Perfetto](https://ui.perfetto.dev). Power samples appear
as `[Power]` events carrying a `Power Usage` value in milliwatts.

## If something fails

`scripts/ae.py` keeps going past a failing environment, so one framework that
will not install does not cost you the rest. Re-run a single one after fixing
it:

```sh
scripts/ae.py build --env vllm
scripts/ae.py run --env vllm
```

`--rebuild` discards an environment and starts it over. The environments,
logs and results all live under `.ae/`, which is disposable — deleting it costs
only the time to build again.

## Comparing two systems

Most of the issues are the same shape: something is done two ways and one is
worse. The totals in each `README.md` say by how much; the comparison says
*where*. Each example that can be compared prints a table of the regions the two
runs compute the same way, with what each spent on it, and writes
`comparison.json` beside its traces.

```
         ops              pytorch               custom   what it computes
       a / b          GPU us / mJ          GPU us / mJ
    4 /    7      74958.4 / 29774.75       5797.7 / 2908.52   layer_norm_channels_first
```

That is `pytorch-76012`: the custom channels-first layer norm takes a twelfth of
the GPU time and a tenth of the energy, and the table says which region.

The regions come from `magneton.matching` -- the two graphs are divided at the
tensors they agree on, and what lies between successive agreements can only
correspond. `magneton.compare` attaches the per-operation cost to those
regions.

Which cost depends on what measured the run. `magneton` records the graph and
matches it with nothing but torch, which is enough to say the two sides compute
the same thing; `magneton.CudaEventTiming` adds latency; `eprof` adds energy,
and is what every example here uses:

```python
from magneton import eprof

backend = eprof.EnergyBackend(devices=[0], tracing_config=...)
with magneton.record(model, ([x], {}), backend=backend,
                     dataflow_config=magneton.DataflowConfig(
                         record_dataflow=True, clone_outputs=True)) as (rec, compiled):
    for _ in range(MEASURED_REPEATS):
        compiled(x)
run = rec.run("a label")        # read out before the next run starts
```

### Where the two sides cannot share a process

`vllm-9471`, `vllm-10811` and `diffusers-12131` differ by a library version or a
source patch, so each side has to be its own process. Those have a
`compare_variants.py`:

```sh
python compare_variants.py --save before
# apply the patch named in README.md
python compare_variants.py --save after
python compare_variants.py --compare before after
```

`--save` writes the graph, its tensors and the per-operation cost, so the
comparison afterwards reports both structure and energy.
