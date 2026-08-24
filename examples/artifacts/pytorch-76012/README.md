# PyTorch Issue #76012

## Evaluation

The PyTorch implementation calls two operators:
1. `layer_norm`: latency 4.02ms, power 536W
2. `contiguous`: latency 1.91ms, power 595W
The total latency is 5.93ms, and the total energy is 3.291J.

The custom implementation call the triton kernel.
The total latency is 6.12ms, and the total energy is 2.533J.
