"""Replaying an operation so that its energy can be measured at all.

Energy is the integral of board power over the time an operation ran, and NVML
reports power on the order of once a millisecond. An operation that finishes
faster than that falls between two samples and is credited nothing -- which is
most operations. Running it repeatedly until the window spans several samples
is what makes the number exist.

This implements magneton's plugin protocol structurally: it has the three
hooks and a priority, and imports nothing from magneton to say so. Whoever
drives the graph transform installs it; `magneton.backends.energy` is what
hands it over.
"""

import logging
import math
from typing import Any, Callable, Optional, Union

import torch


logger = logging.getLogger(__name__)


# Constants (moved from old replay.py)
MAX_REPLAY_ROUNDS_PER_GRAPH = 1000
"""Maximum iterations to capture in a single CUDA graph"""

MAX_REPLAY_ROUNDS_PER_OP = 1000000
"""Maximum total replay rounds per operation"""

MIN_REPLAY_TIME_MS = 2000
"""Target duration for auto-tuned replay (2 seconds)"""


def time_function(func: Callable, num_runs: int = 100) -> float:
    """
    Measure average execution time of a function using CUDA events.
    
    Args:
        func: Function to time
        num_runs: Number of runs to average over
    
    Returns:
        Average execution time in milliseconds
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record(stream=torch.cuda.current_stream())
    for _ in range(num_runs):
        func()
    end_event.record(stream=torch.cuda.current_stream())
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


class ReplayPlugin:
    """
    Plugin that replays operations for energy measurement.
    
    Replays operations multiple times to extend their duration,
    ensuring sufficient NVML power samples (~1000 Hz) for accurate
    energy measurement.
    """
    
    def __init__(
        self,
        replay_rounds: Union[int, str] = "auto",
        replay_cuda_graph: bool = True,
        max_num_replay_ops: Optional[int] = None,
        replay_once_per_op: bool = True,
    ):
        """
        Initialize replay plugin.

        Args:
            replay_rounds: Number of times to replay each operation.
                          Can be an integer or "auto" for auto-tuning.
            replay_cuda_graph: Whether to use CUDA graphs for replay.
                              CUDA graphs are more efficient for high replay counts.
            max_num_replay_ops: Maximum number of operations to replay.
                               If set, only the first N operations are replayed.
            replay_once_per_op: If True, each op is replayed only on its first
                               invocation across calls of the compiled model.
                               If False, replay fires on every invocation.
        """
        self.replay_rounds = replay_rounds
        self.replay_cuda_graph = replay_cuda_graph
        self.max_num_replay_ops = max_num_replay_ops
        self.replay_once_per_op = replay_once_per_op
        self.replayed_ops = set()

        logger.debug(
            f"ReplayPlugin initialized "
            f"(rounds={replay_rounds}, cuda_graph={replay_cuda_graph}, "
            f"max_ops={max_num_replay_ops}, once_per_op={replay_once_per_op})"
        )
    
    @property
    def priority(self) -> int:
        """Medium priority (execute after dataflow)."""
        return 50
    
    def before_execute(
        self,
        op_id: int,
        op_name: str,
        args: tuple,
        kwargs: dict
    ) -> dict:
        """
        Check if we should replay this op.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Context dict with replay decision
        """
        should_replay = True

        # Prefix cap: only replay the first N ops in FX order.
        if self.max_num_replay_ops is not None and op_id >= self.max_num_replay_ops:
            should_replay = False

        # Dedup across calls of the compiled model.
        if self.replay_once_per_op and op_id in self.replayed_ops:
            should_replay = False
        
        return {
            "should_replay": should_replay,
            "op_name": op_name
        }
    
    def after_execute(
        self,
        op_id: int,
        op_name: str,
        output: Any,
        context: dict
    ) -> Any:
        """
        Mark op as replayed.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            output: Operation output
            context: Context from before_execute
        
        Returns:
            Unchanged output
        """
        if context.get("should_replay", False):
            self.replayed_ops.add(op_id)
        return output
    
    def wrap_execute(
        self,
        op_callable: Callable,
        context: dict
    ) -> Any:
        """
        Execute operation with replay logic.
        
        Args:
            op_callable: Operation to execute
            context: Context from before_execute
        
        Returns:
            Operation output
        """
        if not context.get("should_replay", False):
            # No replay needed
            return op_callable()
        
        op_name = context.get("op_name", "unknown")
        
        # First execution (dry run)
        with torch.profiler.record_function(f"forward_{op_name}"):
            output = op_callable()
        
        def clone_output(output: Any) -> Any:
            if isinstance(output, torch.Tensor):
                return output.detach().clone()
            elif isinstance(output, list):
                return [clone_output(o) for o in output]
            elif isinstance(output, dict):
                return {k: clone_output(v) for k, v in output.items()}
            else:
                return output
        
        output_cloned = clone_output(output)
        
        # Determine replay rounds
        replay_rounds = self.replay_rounds
        if isinstance(replay_rounds, str):
            if replay_rounds == "auto":
                # Auto-tune based on operation duration, capped by what the
                # replay path below will actually accept. A plain loop refuses
                # more than MAX_REPLAY_ROUNDS_PER_GRAPH, so auto-tuning against
                # the higher per-op limit and then hitting that refusal is a
                # failure this can simply avoid: a fast op with cuda graphs off
                # would otherwise always ask for more rounds than are allowed.
                duration = time_function(op_callable, 100)
                limit = (
                    MAX_REPLAY_ROUNDS_PER_OP
                    if self.replay_cuda_graph
                    else MAX_REPLAY_ROUNDS_PER_GRAPH
                )
                replay_rounds = max(1, min(limit, int(MIN_REPLAY_TIME_MS / duration)))
                logger.debug(
                    f"Auto-tuned {op_name}: duration={duration:.2f}ms, "
                    f"replay_rounds={replay_rounds}"
                )
            else:
                raise ValueError(f"Invalid replay rounds: {replay_rounds}")
        
        assert isinstance(replay_rounds, int), f"replay_rounds must be int, got {type(replay_rounds)}"
        
        logger.debug(f"Replaying {op_name} {replay_rounds} times")
        
        # Execute replay
        if not self.replay_cuda_graph:
            # Simple loop replay
            if replay_rounds > MAX_REPLAY_ROUNDS_PER_GRAPH:
                raise ValueError(
                    f"Replay rounds {replay_rounds} is too large "
                    f"(max {MAX_REPLAY_ROUNDS_PER_GRAPH}). "
                    f"Use CUDA graph instead (replay_cuda_graph=True)."
                )
            
            for i in range(replay_rounds):
                with torch.profiler.record_function(f"replay_{op_name}_{i}"):
                    output = op_callable()
        else:
            # CUDA graph replay (more efficient)
            cuda_graph = torch.cuda.CUDAGraph()
            graph_replay_rounds = max(
                1, min(MAX_REPLAY_ROUNDS_PER_GRAPH, replay_rounds)
            )
            
            # Capture graph
            with torch.cuda.graph(cuda_graph):
                for _ in range(graph_replay_rounds):
                    output = op_callable()

            # Replay graph
            num_graph_replays = math.ceil(replay_rounds / graph_replay_rounds)
            for _ in range(num_graph_replays):
                cuda_graph.replay()
        
        return output_cloned


