"""Golden-oracle regression tests.

These pin the shape of the exported event tree while the C++ `data::Result`
implementation still exists, so the Rust event-store migration can be proven
equivalent after that reference is deleted.

`py_stack` is the only coverage of the CPython tracer (src/torch/python/
tracer.cpp, the largest remaining C++ file) -- nothing else in the
suite runs with with_stack=True.

The checks run in a subprocess on purpose: with with_stack=True the tracer
records whatever the interpreter is doing, which under pytest includes pytest's
own import machinery and any plugins. Only a clean interpreter reproduces the
environment the goldens were frozen in.
"""

import os
import subprocess
import sys

import pytest
import torch

import golden_oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="golden oracle needs a CUDA device"
)

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_oracle.py")


@pytest.mark.parametrize("label", sorted(golden_oracle.CONFIGS))
def test_golden(label):
    proc = subprocess.run(
        [sys.executable, _SCRIPT, "check", label],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"golden '{label}' drifted; if the change is intended, re-freeze with\n"
        f"  python tests/python/golden_oracle.py freeze {label}\n\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )
