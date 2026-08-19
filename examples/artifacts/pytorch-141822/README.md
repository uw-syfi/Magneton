# PyTorch Issue #141822

## Evaluation

The PyTorch implementation calls two operators:
1. `cross_entropy`: latency 55ms, power 122W
2. `reduce`: latency 2us, power 119W
The total latency is 55ms, and the total energy is 6.71J.

The original implementation calls five operators:
1. `log_softmax`: latency 196us, power 614W
2. `gather`: latency 4.5us, power 115W
3. `ne`: latency 1.2us, power 114W
4. `float`: latency 3.5us, power 114W
5. `mul`: latency 1.5us, power 115W
6. `sum`: latency 2.4us, power 115W
7. `neg`: latency 1.2us, power 114W
The total latency is 210.3us, and the total energy is 0.122J.