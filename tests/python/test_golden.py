"""Golden-oracle regression tests."""

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
