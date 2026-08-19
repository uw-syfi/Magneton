"""Magneton: what two runs of a computation have in common, and where they differ.

Record a model's dataflow, match it against another recording, and report the
regions that compute the same thing. What you do with those regions is a choice
of backend -- structure alone answers whether two implementations agree,
latency answers which is faster, and energy answers what the difference drew.

    with magneton.record(model, ([x], {})) as (rec, compiled):
        compiled(x)
    a = rec.run("eager")

Recording needs nothing but torch, and so does `magneton.CudaEventTiming`.
Energy needs `magneton.eprof`, which wraps an extension module built against
CUPTI and NVML -- imported explicitly, and never from here, so that a machine
without the toolchain can still record and match:

    from magneton import eprof
    backend = eprof.EnergyBackend(devices=[0])
"""

from . import backends as backends
from . import compare as compare
from . import config as config
from . import matching as matching
from . import transform as transform
from . import utils as utils
from .backends.base import Cost as Cost
from .backends.base import CostBackend as CostBackend
from .backends.timing import CudaEventTiming as CudaEventTiming
from .config import DataflowConfig as DataflowConfig
from .dataflow import DataflowDAG as DataflowDAG
from .dataflow import NodeExecution as NodeExecution
from .plugin import OpPlugin as OpPlugin
from .plugin import PluginManager as PluginManager
from .recorder import BaseRecorder as BaseRecorder
from .recorder import record as record


def __getattr__(name):
    # Only reachable with the extension installed, and named lazily so that
    # importing magneton on a machine without it still works.
    if name == "EnergyBackend":
        from .eprof.backend import EnergyBackend

        return EnergyBackend
    if name == "Recorder":
        # The torch recorder under its old name; importing it is importing the
        # FX pass, which is why it is not at the top.
        from .recorders.fx import TorchRecorder

        return TorchRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Cost", "CostBackend", "CudaEventTiming", "DataflowConfig", "DataflowDAG",
    "BaseRecorder", "NodeExecution", "OpPlugin", "PluginManager", "backends",
    "compare", "config", "matching", "record", "transform", "utils",
]

