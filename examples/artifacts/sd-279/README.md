# Stable Diffusion Issue 279

## Evaluation

The original version call these operators in the forward pass of a transformer layer:
1. `layer_norm`: latency 2.1ms, power 357W
2. `add`, latency 0.016ms, power 286W
3. `mul`, latency 0.214ms, power 617W
4. `add`, latency 0.213ms, power 619W
5. `linear`, latency 0.852ms, power 694W
6. `cat` * 3, latency (each) 0.331ms, power 448W
7. `scaled_dot_product_attention`, latency 9.1ms, power 691W
8. `linear`, latency 0.284ms, power 674W
9. `mul`, latency 0.214ms, power 623W
10. `add`, latency 0.497ms, power 656W
11. `layer_norm`: latency 2.1ms, power 357W
12. `add`, latency 0.016ms, power 286W
13. `mul`, latency 0.214ms, power 617W
14. `linear`, latency 1.142ms, power 669W
15. `gelu`, latency 0.533ms, power 694W
16. `linear`, latency 0.803ms, power 703W
17. `mul`, latency 0.214ms, power 623W
18. `add`, latency 0.497ms, power 656W
The total latency is 20ms, and the total energy is 12.04J.

In the optimized version, the energy consumption of these operators are changed:
5. `linear`: latency 0.301ms, power 691W
8. `linear`: latency 0.198ms, power 618W
14. `linear`: latency 0.397ms, power 688W
16. `linear`: latency 0.319ms, power 698W
The total latency becomes 18.2ms, and the total energy becomes 10.7J.

The energy reduction ratio is 12.5%.
