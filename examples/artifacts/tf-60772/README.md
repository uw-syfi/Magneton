# TensorFlow Issue 60772

## Notice

To reproduce the issue with TensorFlow version higher than 2.12.0, you need to modify the `tensorflow/python/ops/math_ops.py` file.
```python
if input.dtype == dtypes.bool: # line 2455
```
should be changed to
```python
if False:
```

## What it measures now

Both ways of counting are recorded as dataflow graphs and matched against each
other, the same as the torch examples -- `magneton.record(..., framework="tf")`
walks the graph a `tf.function` builds and runs the operations out of it one at
a time. On an H200, per call:

| operation | GPU us | mJ |
|---|---|---|
| `Cast` | 106.6 | 13.7 |
| `Sum` | 263.3 | 31.0 |

2 equivalent regions covering all three operations on each side, and
`cast_and_sum` takes **1.00x the GPU time**: on this TensorFlow the two are the
same program. The redundant comparison is guarded by `if input.dtype ==
dtypes.bool` in `math_ops.py`, so a boolean input already skips it. Reproducing
the reported behaviour means changing that line as below, after which the
recording shows the extra operation and what it cost.

## Evaluation (as reported)

The original `count_nonzero` implementation calls three kernels: not_equal, cast_bool and reduce_kernel. The total latency is 0.46ms, and the power is 269W, so the energy consumption is 123.74mJ.

The optimized version calls two kernels: cast_bool and reduce_kernel. The total latency is 0.36ms, and the power is also 269W, so the energy consumption is 96.84mJ.

The energy difference is 27.8%.

## Why there is no subgraph comparison here

The other issues are diagnosed by matching the two versions' dataflow graphs
against each other and reporting what each region cost. That needs the FX graph
`torch.compile` produces, and this example has no torch model at all -- it
profiles TensorFlow work through eprof's kineto side, which sees kernels but no
operator graph. The per-operation table and the trace are what this one offers.
