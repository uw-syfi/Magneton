"""magneton has to work without the extension module.

This is the invariant the whole layering exists to create. Recording a dataflow
graph, matching two of them, and timing with CUDA events need nothing but
torch; only energy needs `magneton_eprof`, which needs CUPTI, NVML and a build
toolchain that a machine may simply not have. That is why it is an extra --
`magneton[energy]` -- rather than a dependency.

The check blocks `magneton_eprof` at import time rather than trusting that nothing
imports it, because a single stray module-level import in magneton would
re-couple the two and no other test would notice: everything else here runs
where the extension is installed.
"""

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

BLOCKED = ("magneton_eprof",)

_SCRIPT = '''
import sys

class Blocker:
    """Refuses the two modules magneton must not need.

    find_spec is the current hook; find_module is the one python 3.11 and
    earlier also consult. Both are here so this does not quietly stop
    blocking anything on a newer interpreter, which would leave every test
    below passing for the wrong reason.
    """
    def find_spec(self, name, path=None, target=None):
        self._check(name)
        return None

    def find_module(self, name, path=None):
        self._check(name)
        return None

    def _check(self, name):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError(f"{{name}} is blocked by this test")

sys.meta_path.insert(0, Blocker())

{body}
print("OK")
'''


def run_without_eprof(body: str):
    """Runs `body` in a fresh interpreter that cannot import the extension."""
    script = _SCRIPT.format(blocked=BLOCKED, body=textwrap.dedent(body))
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"magneton needed the extension module:\n{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout


def test_magneton_imports_without_eprof():
    run_without_eprof("import magneton")


def test_the_public_surface_does_not_reach_eprof():
    run_without_eprof(
        """
        import magneton
        magneton.record
        magneton.compare.compare
        magneton.matching.match_graphs
        magneton.CudaEventTiming()
        magneton.DataflowDAG()
        magneton.Cost()
        """
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="records a CUDA run")
def test_recording_and_matching_without_eprof():
    """The whole point, end to end: two recordings compared on structure."""
    out = run_without_eprof(
        """
        import torch, magneton

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(32, 32)
            def forward(self, x):
                return torch.relu(self.a(x))

        x = torch.randn(4, 32, device="cuda")
        model = Net().cuda()

        runs = []
        for label in ("one", "two"):
            with magneton.record(model, ([x], {}), clone_outputs=True) as (rec, m):
                m(x)
            runs.append(rec.run(label))

        report = magneton.compare.compare(*runs)
        assert report.regions, "two recordings of one model should match"
        assert not runs[0].has_energy, "nothing measured energy here"
        print("regions:", len(report.regions))
        """
    )
    assert "regions:" in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="times a CUDA run")
def test_the_timing_backend_needs_no_eprof():
    run_without_eprof(
        """
        import torch, magneton

        model = torch.nn.Linear(32, 32).cuda()
        x = torch.randn(4, 32, device="cuda")
        with magneton.record(model, ([x], {}), backend=magneton.CudaEventTiming()) as (rec, m):
            m(x)
        run = rec.run("timed")
        assert run.per_node, "the timing backend reported nothing"
        assert not run.has_energy
        """
    )


def test_the_energy_subpackage_is_the_one_thing_that_does():
    """And that it says so in a way someone can act on.

    An ImportError naming `magneton_eprof` and nothing else would leave a reader
    guessing what to install; the message has to name the extra.
    """
    run_without_eprof(
        """
        import magneton
        try:
            from magneton import eprof
        except ImportError as exc:
            assert "magneton[energy]" in str(exc) or "MAGNETON" in str(exc) or \
                   "install" in str(exc), f"unhelpful message: {exc}"
        else:
            raise AssertionError("magneton.eprof should need the extension")
        """
    )
