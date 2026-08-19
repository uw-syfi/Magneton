"""How a recording is configured."""

import dataclasses


@dataclasses.dataclass
class DataflowConfig:
    """Configuration for dataflow recording."""

    record_dataflow: bool = False
    """Enable dataflow recording to capture input/output tensors."""
    clone_outputs: bool = False
    """Clone outputs to record in-place modifications."""
