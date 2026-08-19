# PyTorch Issue #141210

## Notice

The reproduction of this issue requires `nvidia-cublas-cu12==12.4.*`.

## Evaluation

For separate `add` and `mm` operations:
1. `mm`: latency 5.50ms, power 686W
2. `add`: latency 0.37ms, power 477W
The total latency is 5.87ms, and the total energy is 3.94J.

For fused `addmm` operation, the latency is 5.89ms, the power is 697W, and the total energy is 4.11J.
