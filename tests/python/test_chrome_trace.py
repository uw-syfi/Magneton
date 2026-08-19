"""What a saved chrome trace has to contain.

Both of the things checked here were broken and nothing noticed, because every
other test reads the in-memory event tree and never the file. A trace can be
unreadable, or silently missing the profiler's headline number, while the whole
suite stays green.

  - the file has to parse. The "Call stack" metadata value was written unquoted,
    so any run with record_stack=True produced JSON no viewer could load.
  - a [Power] event has to carry its reading. The per-kind metadata was lost
    when that builder moved out of C++, leaving power markers in the timeline
    with no power on them.
"""

import json
import os
import tempfile

import pytest
import torch

import magneton_eprof


def _trace(with_stack: bool, profile_energy: bool) -> dict:
    """Profiles a little GPU work and returns the saved trace, parsed."""
    devices = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    prof = magneton_eprof._Profiler(
        {magneton_eprof._ActivityType.CPU, magneton_eprof._ActivityType.CUDA},
        False,
        False,
        False,
        with_stack,
        with_stack,
        profile_energy,
        devices if profile_energy else [],
    )
    prof.start(set())
    a = torch.randn(256, 256, device="cuda")
    for _ in range(3):
        (a @ a).relu()
    torch.cuda.synchronize()
    result = prof.stop()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        result.save(path)
        with open(path) as fh:
            # json.load is the assertion: a trace that does not parse is a
            # trace nobody can open.
            return json.load(fh)
    finally:
        os.unlink(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_saved_trace_parses_with_python_frames_recorded():
    trace = _trace(with_stack=True, profile_energy=False)
    events = trace["traceEvents"]
    assert events, "the trace has no events"

    stacks = [e for e in events if "Call stack" in e.get("args", {})]
    assert stacks, "with_stack was on but nothing carries a call stack"
    # The value survived quoting as a string, not as whatever the parser could
    # make of an unquoted run of path characters.
    assert all(isinstance(e["args"]["Call stack"], str) for e in stacks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_power_events_carry_their_reading():
    trace = _trace(with_stack=False, profile_energy=True)
    power = [e for e in trace["traceEvents"] if e.get("name") == "[Power]"]
    if not power:
        pytest.skip("no power samples collected; NVML may be unavailable")

    for event in power:
        args = event.get("args", {})
        assert "Power Usage" in args, f"[Power] event with no reading: {args}"
        assert "Device Id" in args
        assert "Device Type" in args

    readings = [int(e["args"]["Power Usage"]) for e in power]
    # Milliwatts. A GPU that is switched on and running matmuls draws tens of
    # watts at least; zero would mean the field is present but never filled.
    assert min(readings) > 1000, f"implausible power readings: {readings[:5]}"
