"""Ways of measuring a recorded run.

`base` is the contract a backend implements; `timing` is the one that needs
nothing but torch. The energy backend is not here -- it lives in
`magneton.eprof`, with the extension module it depends on, so that naming a
backend never drags in a build toolchain.
"""

from .base import Cost as Cost
from .base import CostBackend as CostBackend
from .timing import CudaEventTiming as CudaEventTiming

__all__ = ["Cost", "CostBackend", "CudaEventTiming"]
