# JAX issue 29875

[jax-ml/jax#29875](https://github.com/jax-ml/jax/issues/29875) — a depthwise 1D
convolution is about three times slower in JAX than in PyTorch, "and draws
noticeably more power".

```sh
python main.py
```

The latency is in the issue. The power is asserted there and never measured,
which is what this adds.

## What it measures

Batch 128, 256 channels, length 512, per call on an H200:

| | wall ms | GPU us | mJ |
|---|---|---|---|
| torch | 1.457 | 185.1 | 28.420 |
| jax | 2.211 | 576.4 | 85.500 |

**JAX takes 3.11x the GPU time and 3.01x the energy for the same convolution.**
The energy follows the time almost exactly, which is worth knowing: the slower
kernel is not drawing less power for longer, it is doing the same work less
efficiently and paying for the whole of it.

The kernel each one dispatches is what the issue points at, and both show up
here by name:

```
jax      80.717 mJ    544.2 us  void cudnn::cnn::conv2d_grouped_direct_kernel<
torch    28.420 mJ    185.1 us  aten::_conv_depthwise2d
```

which matches the issue's report of `conv2d_grouped_direct_kernel` against
`conv_depthwise2d_forward_kernel_generic`. The reported timings were 517 us and
179 us; this measures 576 us and 185 us on the same class of device.

## Same convolution, not just the same shape

The issue gives each framework its own random weights, which is enough to
compare timings but leaves open whether the two computed the same thing.
`main.py` transposes torch's weights into the layout JAX wants and checks the
outputs against each other — they agree to 5e-07 — so what remains between them
is how each chose to run it, not what it ran.

## Why there is no dataflow comparison

The other comparisons here divide two recorded graphs at the tensors they agree
on. Recording a graph means instrumenting an FX graph, which needs dynamo, and
JAX has no equivalent. Per-operation cost on each side of the same computation
is the comparison available across two frameworks.
