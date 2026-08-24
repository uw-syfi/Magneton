# HF Transformers Issue 14450

## Evaluation

The original version has these operators in a layer:
1. `layer_norm`: latency 5.0us, power 144W
2. `kqv_proj`: latency 101.1us, power 597W
3. `contiguous` * 3: latency (each) 4.4us, power 281W
4. `scaled_dot_product_attention`: latency 176.2us, power 408W
5. `o_proj`: latency 40.3us, power 469W
6. `add`: latency 2.7us, power 159.7W
7. `layer_norm`: latency 5.2us, power 305W
8. `up_proj`: latency 120.8us, power 680W
9. `gelu`: latency 37.9us, power 419W
10. `down_proj`: latency 144.2us, power 428W
The total latency is 646.6us, and the energy consumption is 313.7mJ.

In the optimized version, the energy consumption of these operators are changed:
2. `kqv_proj`: latency 21.0us, power 349W
5. `o_proj`: latency 11.1us, power 199W
8. `up_proj`: latency 26.3us, power 543W
10. `down_proj`: latency 24.3us, power 467W
The total latency is 322.9us, and the energy consumption is 129.4mJ.