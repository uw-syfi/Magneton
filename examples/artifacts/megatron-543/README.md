# Megatron-LM

A one-layer Megatron-core GPT decoder, profiled per operation.

```sh
python main.py
```

Writes `trace.json` and `per_op.json`. On an H200, per training step:

| GPU us | mJ | operation |
|---|---|---|
| 155.9 | 17.947 | `aten::mm` |
| 10.7 | 1.232 | `aten::_log_softmax` |
| 9.8 | 1.129 | `aten::native_layer_norm` |
| 8.6 | 0.995 | `aten::native_dropout` |
| **221.7** | **25.523** | total |

## What main.py patches, and why

One thing, and it is dynamo's rather than the model's. `make_traceable()` is
only a no-op at `tensor_model_parallel_size=1`, which it asserts. The patched
model's output is bit-identical to the unpatched one, eager and traced.

**The CUDA RNG tracker's `fork()`** is a `@contextlib.contextmanager`, and
dynamo refuses to inline one under `fullgraph=True` — `SKIPPED INLINING <code
object __enter__ ... contextlib.py>`. A context manager written as a class
traces without complaint, and with one rank there is no second RNG stream for
the forked one to differ from.

Note that flipping `config.sequence_parallel` is *not* enough on its own, which
is worth knowing because it looks like it should be: attention forks when
sequence parallelism is off and the embedding forks when it is on, so no value
of the flag avoids both.

Nothing else is patched. The tensor-parallel linear and the region boundaries
are megatron's own `torch.autograd.Function`s and are profiled as they are.

That was not true until recently: applying any autograd.Function inside a
profiled region aborted with `std::bad_alloc` from
`torch::autograd::Function::apply`, and this example replaced three of them to
get around it. The cause was the profiler putting its per-thread state in
`DebugInfoKind::PROFILER_STATE` — the slot torch reads back through
`ProfilerStateBase::get()`, which casts whatever it finds — while deriving from
a different base. Ordinary operators never call that getter; autograd.Function
does. Fixed in `lib/eprof-torch/include/state.h`, covered by
`tests/python/test_autograd_function.py`.
