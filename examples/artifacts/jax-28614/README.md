# JAX issue 28614

[jax-ml/jax#28614](https://github.com/jax-ml/jax/issues/28614) — a batched
short-time Fourier transform is far slower than the transform it is made of.

```sh
python main.py
```

## What it measures

Per-JAX-operation GPU time and energy for `jax.scipy.signal.stft` under `vmap`,
over a batch of 256 signals. On an H200:

| operation | mJ | GPU us | share |
|---|---|---|---|
| `cudnn-conv.1.0` | 116.4 | 917.3 | **96.6%** |
| `fft.1.0` | 1.2 | 9.4 | 1.0% |
| `loop_convert_fusion` | 0.8 | 6.0 | 0.6% |
| the other five fusions | 1.6 | 12.6 | 1.8% |

The FFT this function exists to compute is **1% of its energy**. Almost all of
it goes to a convolution — `jax.scipy.signal.stft` builds its frames with
`conv_general_dilated_patches`, and under `vmap` that lowers to a batched cuDNN
convolution which costs more than the transform by two orders of magnitude.

Grouping is by HLO operation, which is the level the program is written at, and
each one is broken down into the kernels that ran it:

```
 115.239 mJ   916.5 us   96.6%  cudnn-conv.1.0
      81.255 mJ    646.2 us  x1   sm80_xmma_fprop_implicit_gemm_indexed_wo_sme
      33.984 mJ    270.3 us  x2   void cudnn::engines_precompiled::nchwToNhwcK
```

`eprof.Profiler.per_hlo_op()` is what produces that, and `energy.json` carries
the same tree. Energy is worked out per kernel launch and summed afterwards,
not the other way round: board power is shared between whatever was running at
the time, so what a launch is charged depends on its neighbours.

## Files

| | |
|---|---|
| `main.py` | the measurement; writes `energy.json` |
