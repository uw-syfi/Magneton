# transformers issue 34570

`torch.linalg.eigvals` where `torch.linalg.eigvalsh` would do.

A covariance matrix is symmetric, so its eigenvalues are real and the symmetric
solver can have them. The general solver does not know that. Both return the
same eigenvalues.

```sh
python main.py
```

## What it measures

Wall clock, GPU time and GPU energy for each solver, per call, plus a
per-operation breakdown. On an H200 at 1024x1024:

| | wall ms | GPU us | GPU mJ | kernels |
|---|---|---|---|---|
| `eigvals` | 1118.6 | 1065.5 | 121.3 | 2 |
| `eigvalsh` | 10.4 | 9839.8 | 1420.6 | 165 |

`eigvalsh` finishes in **108x less time** for the same answer.

## Why all three columns are here

They disagree, and that is the point. The slow solver draws *less* GPU energy
than the fast one, and launches two kernels to its 165.

torch has no CUDA implementation of non-symmetric eigenvalues. `eigvals` copies
the matrix to the host, factorises it on the CPU, and copies the result back --
its two kernels are those copies, and the second of them is most of its GPU
energy. So it is not slow because it does more work on the device; it is slow
because it does almost none.

Reading the energy column on its own would report the bug as an improvement.
That is the case for measuring latency next to it.

## Why there is no subgraph comparison

Other examples here divide two recorded graphs at the tensors they agree on and
report the regions between. These two never produce an agreeing tensor:
`eigvals` returns `complex64` in no particular order, `eigvalsh` returns
`float32` ascending. The values match only after sorting and discarding a zero
imaginary part -- true of eigenvalues, and not something a tensor comparison
should assume on their behalf. One operator against one operator is the level
this question lives at.
