"""TensorFlow issue 60772: count_nonzero doing more work than it needs to.

`tf.math.count_nonzero` compares its input against zero before summing. On an
input that is already boolean that comparison is redundant, and the issue is
that it was performed anyway -- three kernels where two would do.

    python main.py

Records both ways of counting, matches them as graphs, and reports what each
operation cost.

TensorFlow has no dispatcher to interpose on and no interpreter to borrow, so
the recorder asks a `tf.function` for its concrete graph and runs the operations
out of it one at a time. Kernels are attributed to whichever operation was
running when they launched: CUPTI reports TensorFlow's kernels like any other,
but they reach the GPU without passing an aten operator, so there is nothing
else to charge them to.

**On this TensorFlow the two are the same.** The redundant comparison is guarded
by `if input.dtype == dtypes.bool` in `math_ops.py`, so a boolean input already
skips it, and the comparison below finds two identical graphs. Reproducing the
original behaviour means changing that line to `if False:`, which the README
gives; the recording then shows the extra operation and what it cost.
"""

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
