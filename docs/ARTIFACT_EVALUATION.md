# Artifact evaluation

Everything is driven by one script:

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
| clang 18 or newer | the C++ uses C++20 `std::format` |
| [uv](https://docs.astral.sh/uv/) | creates the environments |
| [cargo](https://rustup.rs) | builds the profiler |
| ~60 GiB disk | mostly the frameworks: vLLM, TensorFlow and Megatron are large |

Developed and measured on an H200. Any recent NVIDIA GPU should work; the
numbers in the examples' READMEs are of course specific to the hardware.

On Ubuntu 22.04 the compilers are older than this needs, so they come from
their own repositories:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main"
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install -y clang-18 gcc-13 g++-13 libboost-all-dev
```

g++-13 is there for its libstdc++ rather than to compile anything: clang picks
up the newest one it finds, and `std::format` needs a recent one. If `clang++-18`
is not the default compiler, point `CXX` at it.

Only the energy extension needs any of this. `pip install ./python` installs
magneton alone, which records, matches and compares with nothing but torch;
`EPROF_USE_PREBUILT=0 pip install --no-build-isolation './python[energy]'`
builds the extension from `lib/eprof` in this checkout. `--no-build-isolation`
is required rather than preferred: the extension links the libtorch that will
import it, and an isolated build would fetch a second one for its build scripts
to find.

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

## Why there are six environments

The examples exercise frameworks whose requirements contradict each other. They
cannot share one environment, and the split is not a matter of taste:

- **vLLM 0.9.1 requires `torch==2.7.0` exactly.**
- **transformers newer than 4.51.3** builds a `DynamicCache` in GPT-2's forward
  that torch 2.7's dynamo cannot trace, so the Hugging Face examples are pinned
  to that version *and* that torch together.
- **TensorFlow** brings its own CUDA stack.

Everything not held back by one of those runs on torch 2.8.

`examples/manifest.toml` is where this lives — which environment each example
needs, and what each one should produce. It is the only place that knowledge
exists; the script reads it and has no lists of its own.

`eprof` is built once per environment, because the extension links the libtorch
that will import it; environments sharing a torch version share the build.
`magneton` is pure python and needs no build at all -- it is installed into
each environment alongside, so that a run always has a matched pair.

## Reading the results

The table reports `passed` only if the example exited 0 **and** wrote every
artifact it declares, **and** those artifacts parse. That last part matters: an
example can exit 0 having silently produced nothing, and two of them have. For
a chrome trace the script also reports how many events and power samples it
contains, so an empty trace is visible rather than merely well-formed.

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

### Small graphs give unstable regions

`pytorch-76012` and `pytorch-153195` have four to seven nodes, most of them
inputs. What each side *cost* is stable to a tenth of a percent run to run, but
which nodes end up inside the matched region is not: with that few nodes there
is barely a spine to cut on, and a different cut moves the totals a lot. Expect
the ratio those two print to move between runs, and read the per-operation
table instead. The report says so itself when a matched region turns out to
contain none of the work.

The larger comparisons -- `hf-14450` at 709 operations, `sd-279` at 136 -- do
not have this problem and repeat to within a few percent.

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

### Where it does not apply

- `tf-60772` profiles TensorFlow work with no torch model, so there is no
  operator graph to match. Per-operation energy and the trace only.
- `pytorch-28224` profiles a plain block for the same reason, and its finding is
  about synchronisation mode rather than operators.
- `pytorch-181115` is about a rank waiting for others; the cost is idleness, not
  a region of the graph.
- `jax-*` record no dataflow graph. The matcher is framework-agnostic, but the
  recorder that builds its input is FX-only.
