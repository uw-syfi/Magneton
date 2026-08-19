# Diffusers Issue 12131

## Notice

The full trace of the model with software-based replay is too large. So in this example, we don't enable replay.

To reproduce the fix, you need to apply the `bugfix.patch` file to the diffusers source code.

## Evaluation

The total latency of a single transformer block is 3.5ms, and the latency of the concat operation is 0.2ms. The power during the forward pass remains stable at 185.6W. Therefore, the energy difference is 6.1%.