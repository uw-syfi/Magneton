"""eprof: the energy backend, and the profiler underneath it.

This is the one part of magneton that needs more than torch. It measures a run
through CUPTI and NVML by way of `magneton_eprof`, the Rust and C++ extension module,
and reports what each part of a model cost in joules as well as milliseconds.

    import magneton
    from magneton import eprof

    with magneton.record(model, ([x], {}),
                         backend=eprof.EnergyBackend(devices=[0])) as (rec, m):
        m(x)
    run = rec.run("eager")

`EnergyBackend` implements `magneton.CostBackend`, so it is one of the things
that can measure a recording, not a thing recordings are built around --
`magneton.CudaEventTiming` is the other, and needs none of this.

Importing this subpackage requires the extension. `magneton` itself never
imports it, so recording, matching and CUDA-event timing all keep working on a
machine where it was never built.

`Profiler` is the capture that `EnergyBackend` drives. Reach for it directly
only where there is no FX graph to record -- JAX and TensorFlow have none, so
profiling the region is the only thing available there.
"""

try:
    from magneton_eprof import _ActivityType as _ActivityType
except ImportError as exc:  # pragma: no cover - depends on how it was installed
    raise ImportError(
        "magneton.eprof needs the magneton_eprof extension module, which this "
        "installation does not have.\n"
        "\n"
        "  It comes with the energy extra:\n"
        "      uv pip install --no-build-isolation 'magneton[energy]'\n"
        "  or, from a checkout:\n"
        "      uv pip install --no-build-isolation './python[energy]'\n"
        "\n"
        "  Building it needs a CUDA toolkit and clang 18 or newer; see\n"
        "  docs/ARTIFACT_EVALUATION.md.\n"
        "\n"
        "  Only energy needs this. Recording a dataflow graph, matching two of\n"
        "  them, and timing with magneton.CudaEventTiming all work without it."
    ) from exc

from . import attribution as attribution
from . import treediff as treediff
from .backend import EnergyBackend as EnergyBackend
from .config import EnergyConfig as EnergyConfig
from .config import ReplayConfig as ReplayConfig
from .config import TracingConfig as TracingConfig
from .profiler import ActivityType as ActivityType
from .profiler import HloOpCost as HloOpCost
from .profiler import KernelCost as KernelCost
from .profiler import Profiler as Profiler
from .replay import ReplayPlugin as ReplayPlugin
