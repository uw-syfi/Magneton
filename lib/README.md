# eprof

The energy profiler: Rust is the host, and the C++ that remains is a leaf. This
is one part of magneton rather than the whole of it -- `python/magneton/` is
the system, `magneton.eprof` is the python side of what is built here, and
`magneton_eprof` is the module it imports.

Rust is the host. It owns every event, the tree they form, the per-thread
collection queues, the python tracer's caches, and the run itself. The C++ that
remains is a leaf: it exists to read what only libtorch and CPython can reach —
an `at::RecordFunction`, a `c10::IValue`, a `PyFrameObject` — and hand the
result over.

One directory per subsystem, and each holds both its halves. A subsystem's C
ABI is its own business, so its header sits next to the C++ that implements it
and the Rust that calls it; no directory declares a boundary it is not on.

A header's name says which way the calls go. `<name>.h` declares C++ that Rust
calls down into; `rust_<name>.h` declares Rust that C++ calls back up into.
There is no third kind — every header in `lib/*/include/` is one or the other.

| directory | what it is |
|---|---|
| `eprof-storage/` | where a run accumulates: the event array, and the tree built over it. Depends on nothing, links nothing, tests in under a second. |
| `eprof-energy/` | the NVML power sampler: a polling thread, and nothing else. Depends on nothing. |
| `eprof-kineto/` | vendored libkineto, the C ABI added to it (`libkineto/include/kineto_c_api.h`), and the Rust that drives a trace's lifecycle |
| `eprof-python/` | the CPython tracer (`PyEval_SetProfile`, frame walking, the GIL) and the Rust that interns callsites and replays them |
| `eprof-torch/` | the `RecordFunction` callbacks, the state torch reports allocations into, and the Rust run they feed |
| `eprof/` | the run driver, attribution, and the `magneton_eprof` extension module |

`eprof-storage/` is a crate rather than part of `eprof/` for one reason: all three
subsystems write into the array, and kineto reads back the torch ops already in
it to correlate against. If it lived with the code that drives them, the
dependency would run both ways.

## Building it

`cargo build`, or `maturin develop` from `eprof/` to install into a virtualenv.
That produces the `magneton_eprof` module, whose surface is declared in
`eprof/magneton_eprof.pyi` -- hand-written, and kept honest by `tests/python/test_stub.py`.
There is no separate C++ step: each crate compiles its own C++ from its own
`build.rs`, and `eprof-kineto/build.rs` drives cmake for vendored libkineto. `pip
install .` runs the same thing through `setup.py`.

Nothing is configured by hand. `eprof-utils/` asks the environment for all of it -- the
python being built against, where that python's torch put its headers and
libraries, which CUDA is installed and under what version suffix -- so a
checkout builds where it stands. The overrides, for when the answer is wrong:
`EPROF_DEVICE`, `EPROF_TORCH_DIR`, `EPROF_CUPTI_DIR`, `CUDA_HOME`, `CXX`.

Two constraints that are not preferences. The compiler must be clang 18 or
newer, because libkineto and the C++ leaves use C++20 `std::format`. And every
translation unit must be compiled with `NDEBUG`, because libtorch guards whole
*members* on it -- `at::RecordFunction::inputs_valid_` is inside an `#ifndef
NDEBUG` -- so a file that disagrees with the libtorch it links against computes
different field offsets and reads past the end of the object.

## The C++ that cannot move

Six files, and each is on the boundary for the same reason — **torch or CPython
calls into us**, so there has to be a C++ object or a C++ signature at the
address it holds.

| file | what it does |
|---|---|
| `eprof-torch/src/capture.cpp` | reads an `at::RecordFunction` on the way in — name, scope, input shapes and dtypes, the `c10::IValue`s it carries |
| `eprof-torch/src/metadata.cpp` | prints an `IValue`, and pulls a collective's description out of the `ParamCommsDebugInfo` c10d leaves in thread-local storage |
| `eprof-torch/src/state.cpp` | the per-run state torch reports allocations into |
| `eprof-torch/src/driver.cpp` | registering and removing the callback pair |
| `eprof-python/src/tracer.cpp` | the CPython tracer |
| `eprof-kineto/src/backend.cpp` | one libtorch registration lookup: whether a backend claimed the PrivateUse1 device slot |

The two that pin the shape:

- `ThreadLocalState` derives from `c10::MemoryReportingInfoBase` and is pushed
  onto `c10::ThreadLocalDebugInfo`. The allocator then calls
  `reportMemoryUsage` / `reportOutOfMemory` on it *virtually*. A C ABI can put
  a function behind a pointer, but not a vtable at an address torch already
  holds.
- The `RecordFunction` enter callback must return a
  `std::unique_ptr<at::ObserverContext>` — a C++ type with ownership semantics
  that cannot cross a C boundary.

Everything else in those files is a read that needs a libtorch or CPython type
to perform, and hands its result straight over.

## Keeping the boundary honest

Two failure modes, opposite directions.

**Exports that lost their callers.** As C++ shrinks, an entry point stops being
called but stays `#[no_mangle] extern "C"`, and the header keeps promising it.
One sweep found 39 at once; a later one found 42 more in `eprof-storage/`. To re-check,
list the Rust exports:

```sh
grep -rhoE 'pub (unsafe )?extern "C" fn [a-z_0-9]+' lib/*/src/*.rs \
  | grep -oE '[a-z_0-9]+$' | sort -u
```

then look for each name in `lib/*/src/*.cpp` and `lib/*/include/*.h`. No match
means dead. A name that matches only inside Rust should be an ordinary Rust
function, not an export.

**Exports the linker cannot see.** Static linking is one left-to-right pass, and
the rlibs come before the C++ archives — so a Rust entry point that only C++
calls is never extracted, and the link fails on an undefined symbol.
`eprof/src/keep.rs` names all of them in a `#[used]` static, which is why that
file exists and why it has to stay in step with the list above.

The `RecordFunction` and autograd sections below are adapted from PyTorch's
profiler README; `eprof-python/src/tracer.cpp` is derived from its python
tracer. Both are under the license in `eprof-python/LICENSE.pytorch`.

## `RecordFunction`

`RecordFunction` is how PyTorch instruments CPU-side events. It sits in the
dispatcher around every op, and callers can register callbacks that run at each
guard. The profiler registers a pair (`eprof-torch/src/driver.cpp`): the enter
stamps the op into the Rust op store and pushes a kineto correlation id, the
exit completes it. The machinery is designed to be cheap when no callback is
registered, but it is not free when one is — which is why capture does the
minimum per op and defers everything it can to materialization.

`with torch.profiler.record_function` is the python binding, commonly used to
annotate module-level events.

## Autograd integration

Two things come from the autograd engine, and together they are what lets a
backward op be matched to the forward op that caused it (`eprof-kineto/src/fwdbwd.rs`,
surfaced in a chrome trace as `fwd_bwd` flow events):

- **Sequence number** — a per-thread index assigned to each forward op whose
  inputs require gradients. The backward op triggered by it gets the same
  number.
- **Forward thread id** — which thread the forward op ran on. Needed because a
  sequence number is only unique within a thread, so the pair `(tid, seq)` is
  the actual key.

An op whose inputs do not require gradients has no sequence number, which is
why the matcher only considers ops carrying one.
