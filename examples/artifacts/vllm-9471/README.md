# vLLM Issue 9471

## Notice

To reproduce the fixed version, you need to add the environment variable `VLLM_ENABLE_TENSOR_CORE=1` to the command.

## Evaluation

The original version has these operators in a layer:
1. `rms_norm`: latency 4.2us, power 210W
2. `fp8_quant`: latency 3.1us, power 152W
3. `kqv_proj`: latency 15.5us, power 693W
4. `rotary_emb`: latency 4.7us, power 203W
5. `flashinfer_decode`: latency 2.8us + 4.0us + 52.3us + 2.1us = 61.2us, power 291W
6. `fp8_quant`: latency 3.0us, power 152W
7. `o_proj`: latency 15.1us, power 532W
8. `rms_norm`: latency 4.2us, power 177W
9. `fp8_quant`: latency 3.1us, power 152W
10. `up_gat_proj`: latency 54.4us, power 695W
11. `silu_and_mul`: latency 11.3us, power 462W
12. `fp8_quant`: latency 6.4us, power 274W
13. `down_proj`: latency 38.8us, power 679W
The total latency is 225us, and the energy consumption is 111.7mJ.

In the fixed version, the latency of the compute kernel inside `flashinfer_decode` reduces to 9.2us. Therefore, the total latency becomes 181.9us, and the energy consumption is 99.16mJ.
