# JAX issue 5647

[jax-ml/jax#5647](https://github.com/jax-ml/jax/issues/5647) —
`jax.scipy.linalg.expm` computed every Padé approximant and threw four away.

`expm` picks an approximant by the norm of its argument. It computed all five
and selected one with `lax.select`, so the four it did not need were evaluated
anyway, matrix multiplications and all. The fix computes only the chosen branch,
with `lax.switch`.

```sh
python main.py
```

## What it measures

The issue was reported as a latency problem. This measures the other half: what
the discarded branches drew in energy. Per call, 256x256, on an H200:

| | before: us / mJ | after: us / mJ | saved |
|---|---|---|---|
| Padé-3 | 634.5 / 72.91 | 545.8 / 63.21 | 14% time, 13% energy |
| Padé-5 | 636.2 / 73.44 | 554.1 / 64.42 | 13% time, 12% energy |
| Padé-13 | 635.7 / 75.04 | 573.8 / 67.60 | 10% time, 10% energy |

On CPU, removing the unused branches is worth 2.1x to 2.6x (`result.txt`). On a
GPU it is worth about a tenth of that, and the per-operation table says why: the
LU factorisation and triangular solves that finish `expm` are the same work
either way and they dominate. Padé-3 discards four branches and still only saves
13%, because what it discards was never the expensive part.

That gap between the CPU figure and the GPU one is the point. The saving is
real; it is a slice of a much smaller share than the reported number suggests.

The table `main.py` prints is per JAX operation, with the kernels that ran each
one underneath -- `custom-call.2.0`, the LU factorisation, is six cuSOLVER
kernels of which `getrf_pivot` is almost all. `eprof.Profiler.per_hlo_op()`
produces it.

## Files

| | |
|---|---|
| `main.py` | both implementations and the measurement |
| `result.txt` | the CPU numbers the comparison above refers to |

`main.py` also writes `expm_before_fix.mlir` and `expm_after_fix.mlir`, the
StableHLO each version compiles to. That is where the wasted work is visible:
the after-fix module contains a `stablehlo.case`, so one branch of five runs,
and the before-fix module contains none -- every approximant is computed
unconditionally and a `select` picks between the results. They are generated
rather than committed, so they always match the code beside them and the jax
that is installed.
