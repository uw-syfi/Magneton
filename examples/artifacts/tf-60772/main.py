"""TensorFlow issue 60772: count_nonzero doing more work than it needs to."""

import json
import os

import tensorflow as tf

import magneton
from magneton import eprof

REPEATS = 200


def count_nonzero(predicate):
    """What the issue reports."""
    return tf.math.count_nonzero(predicate, axis=1)


def cast_and_sum(predicate):
    """The same count, written without asking for the comparison."""
    return tf.math.reduce_sum(tf.cast(predicate, tf.int64), axis=1)


def record(fn, predicate, devices, label):
    backend = eprof.EnergyBackend(devices=devices)
    with magneton.record(fn, ([predicate], {}), framework="tf", backend=backend) as (
        rec,
        run_it,
    ):
        for _ in range(REPEATS):
            run_it(predicate)
    return rec.run(label), backend


def main():
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    print(f"tensorflow {tf.__version__}")

    with tf.device("/gpu:0"):
        data = tf.random.uniform(shape=(32, 1_000_000))
        predicate = tf.greater(data, 0.5)

        assert bool(tf.reduce_all(tf.equal(count_nonzero(predicate),
                                           cast_and_sum(predicate)))), \
            "the two ways of counting disagree"

        original, backend_a = record(count_nonzero, predicate, devices, "count_nonzero")
        backend_a.export_chrome_trace("trace.json")
        manual, backend_b = record(cast_and_sum, predicate, devices, "cast_and_sum")

    for run in (original, manual):
        print(f"\n{run.label}, per operation:")
        for name, cost in sorted(run.per_node.items()):
            print(f"  {cost.gpu_time_ns / 1e3 / REPEATS:9.1f} us  "
                  f"{cost.gpu_energy_j * 1e3 / REPEATS:8.3f} mJ  x{cost.num_kernels} "
                  f"{name}")

    report = magneton.compare.compare(original, manual)
    print()
    print(report)

    with open("comparison.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print("\nWrote comparison.json and trace.json")


if __name__ == "__main__":
    main()
