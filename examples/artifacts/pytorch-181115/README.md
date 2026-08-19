# PyTorch Discussions 181115

For the case with the Join wrapper, where the first rank waits for the other ranks to finish, the total energy consumption is (0.121s + 0.145s) * (138W - 74W) + 0.024s * (124W - 74W) = 18.224J.

For the normal case that the first rank just stops, the total energy consumption is (0.121s + 0.145s) * (138W - 74W) = 17.024J.

In this workload, even if the imbalance ratio is only 23%, the energy difference can be as large as 7% of the total energy consumption.
