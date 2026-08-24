# PyTorch Issue 141210 in HF Transformers

## Evaluation

The original version has these operators in a layer:
1. `layer_norm`: latency 29.8us, power 253W
2. `kqv_proj`: latency 713.7us, power 698W
3. `contiguous` * 3: latency (each) 23.4us, power 261W
4. `scaled_dot_product_attention`: latency 903us, power 483W
5. `o_proj`: latency 233.1us, power 697W
6. `add`: latency 21.3us, power 246.1W
7. `layer_norm`: latency 29.7us, power 254W
8. `up_proj`: latency 892us, power 701W
9. `gelu`: latency 53.9us, power 494W
10. `down_proj`: latency 894us, power 699W
The total latency is 3.84ms, and the total energy is 2.41J.

In the optimized version, the energy consumption of these operators are changed:
2. `kqv_proj` -> `matmul`: latency 636us, power 656W and `add`: latency 58.7us, power 444W
5. `o_proj` -> `matmul`: latency 210us, power 652W and `add`: latency 21.1us, power 446W
8. `up_proj` -> `matmul`: latency 812us, power 642W and `add`: latency 76us, power 388W
10. `down_proj` -> `matmul`: latency 851us, power 638W and `add`: latency 21us, power 389W
The total latency is 3.79ms, and the total energy is 2.19J.

The energy reduction ratio is 9.1%.
