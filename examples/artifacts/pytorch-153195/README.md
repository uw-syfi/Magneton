# PyTorch Issue 153195

## Evaluation

When `torch.set_float32_matmul_precision("high")` is set, the `matmul` operation uses tensor core with latency 3.95ms, and power is 240W.

When `torch.set_float32_matmul_precision("highest")` is set, the `matmul` operation uses CUDA core with latency 4.12ms, and power is 375W.
