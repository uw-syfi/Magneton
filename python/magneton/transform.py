"""
Pluggable FX graph transformation system.

This module provides the infrastructure for applying plugins to FX graph nodes,
enabling features like dataflow recording and operation replay.
"""

import logging
from typing import Any, Dict, List

import torch

from magneton.plugin import OpPlugin
from magneton.plugin import PluginManager
from magneton.utils.fx_utils import OpType, OpWrapper
from magneton.utils.tensor_utils import extract_tensors
import inspect


logger = logging.getLogger(__name__)

try:
    from torch._ops import HigherOrderOperator as _HIGHER_ORDER
except ImportError:  # pragma: no cover - torch older than the higher-order ops
    _HIGHER_ORDER = None


# Every wrapped operation runs inside a `record_function` scope named after the
# submodule that wraps it. That name is the whole contract between this
# transform and anything that measures the run: a profiler sees the scope in
# its trace, and the kernels launched inside it are the ones this operation is
# responsible for. Nothing else connects a cost back to a graph node.
#
# `WRAPPER_PREFIX` names the submodule; `ANNOTATION_PREFIX` is what a trace
# shows, since `forward` is the method the scope is opened in. Both are derived
# from one string so the two ends cannot drift apart.
WRAPPER_PREFIX = "op_wrapper_"
ANNOTATION_PREFIX = f"forward_{WRAPPER_PREFIX}"


class OpPluggableWrapper(OpWrapper):
    """
    Wrapper that applies plugins to operation execution.
    
    This wrapper coordinates multiple plugins, calling their hooks
    at appropriate times during operation execution.
    """
    
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        op_name: str,
        op_type: OpType,
        op_target: Any,
        op_id: int,
        plugin_manager: PluginManager,
    ):
        """
        Initialize pluggable wrapper.
        
        Args:
            gm: FX GraphModule
            op_name: Operation name
            op_type: Operation type
            op_target: Operation target
            op_id: Sequential operation ID
            plugin_manager: Manager coordinating all plugins
        """
        super().__init__(gm, op_name, op_type, op_target)
        self.op_name = op_name
        self.op_type = op_type
        self.op_target = op_target
        self.op_id = op_id
        self.plugin_manager = plugin_manager
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Execute operation with plugin hooks.
        
        Execution flow:
        1. Call before_execute on all plugins (in priority order)
        2. Execute operation (wrapped by plugins)
        3. Call after_execute on all plugins (in reverse priority order)
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation output (potentially modified by plugins)
        """
        with torch.profiler.record_function(f"forward_{self.op_name}"):
            # Phase 1: before_execute (collect context from all plugins)
            context = self.plugin_manager.before_execute(
                op_id=self.op_id,
                op_name=self.op_name,
                args=args,
                kwargs=kwargs
            )
            
            # Add operation metadata to each plugin's context
            for plugin_context in context.values():
                plugin_context["_op_type"] = self.op_type.value if hasattr(self.op_type, 'value') else str(self.op_type)
                plugin_context["_op_target"] = str(self.op_target)
            
            # Phase 2: wrap_execute (nested wrapping)
            def base_callable():
                return self.original_forward(*args, **kwargs)
            
            output = self.plugin_manager.wrap_execute(
                op_callable=base_callable,
                context=context
            )
            
            # Phase 3: after_execute (in reverse priority order)
            output = self.plugin_manager.after_execute(
                op_id=self.op_id,
                op_name=self.op_name,
                output=output,
                context=context
            )
            
            return output


def _computes_nothing(target: Any) -> bool:
    """Whether a `call_function` node is something to leave alone.

    Two kinds, and neither is an operation with a cost to attribute:

    **Higher-order operators** invoke a subgraph rather than doing work
    themselves, and take the graph modules to run as arguments. dynamo
    represents an autograd.Function, a checkpointed region, or a control flow
    construct this way.

    **Class constructors**, which build an object rather than a tensor.
    dynamo emits one for `torch.autograd.function.FunctionCtx`, the context an
    autograd.Function is handed. Running that through the plugin machinery --
    which exists to see tensors in and tensors out -- produced a ctx the
    subsequent apply could not use, and megatron's decoder aborted in C++ with
    std::bad_alloc.

    Both are copied through. The kernels any of it launches still appear in the
    trace and in the per-operator table; what is lost is their attribution to a
    node of the recorded graph, which is honest -- the work happened somewhere
    this pass never looked.
    """

    if inspect.isclass(target):
        return True
    try:
        from torch._ops import HigherOrderOperator
    except ImportError:  # pragma: no cover - very old torch
        return False
    return isinstance(target, HigherOrderOperator)


def _arg_transform(arg: Any, transform_map: Dict[int, Any]) -> Any:
    """
    Recursively transform arguments to reference new nodes.
    
    This helper maps old node references to new node references in the
    transformed graph, handling nested structures (lists, tuples, dicts).
    
    Args:
        arg: Argument to transform (can be FX node, list, tuple, dict, or scalar)
        transform_map: Map from old node ID to new node
    
    Returns:
        Transformed argument with updated node references
    """
    if isinstance(arg, torch.fx.Node):
        # Map old node to new node
        return transform_map.get(id(arg), arg)
    elif isinstance(arg, list):
        return [_arg_transform(a, transform_map) for a in arg]
    elif isinstance(arg, tuple):
        return tuple(_arg_transform(a, transform_map) for a in arg)
    elif isinstance(arg, dict):
        return {k: _arg_transform(v, transform_map) for k, v in arg.items()}
    else:
        return arg


def pluggable_pass(
    gm: torch.fx.GraphModule,
    plugins: List[OpPlugin],
    *args,
    **kwargs
) -> torch.fx.GraphModule:
    """
    Apply pluggable transformation to FX graph.
    
    This function transforms the FX graph by wrapping each operation node
    with OpPluggableWrapper, which coordinates the execution of all provided plugins.
    
    The transformation process:
    1. Create a new empty FX graph
    2. For each operation node (call_function, call_method, call_module):
       - Create an OpPluggableWrapper instance
       - Attach it to the GraphModule as a module attribute
       - Create a new call_module node in the new graph
    3. For non-operation nodes (get_attr, output):
       - Copy them directly to the new graph
    4. Replace the old graph with the new graph
    
    Args:
        gm: FX GraphModule to transform
        plugins: List of plugins to apply
        *args: Additional arguments (unused, for compatibility)
        **kwargs: Additional keyword arguments (unused, for compatibility)
    
    Returns:
        Transformed GraphModule with wrapped operations
    """
    if not plugins:
        logger.debug("No plugins provided, returning original graph")
        return gm
    
    logger.info(f"Applying pluggable pass with {len(plugins)} plugin(s)")
    logger.debug(f"Original graph:\n{gm.graph}")
    
    # Create new empty graph
    new_graph = torch.fx.Graph()
    
    # Create plugin manager
    plugin_manager = PluginManager(plugins)
    
    # Track operation IDs and node mappings
    op_id = 0
    transform_map: Dict[int, Any] = {}
    
    # Iterate through original graph nodes
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            new_placeholder_node = new_graph.node_copy(
                node,
                arg_transform=lambda x: _arg_transform(x, transform_map)
            )
            wrapper_name = f"{WRAPPER_PREFIX}{op_id}"
            setattr(gm, wrapper_name, OpPluggableWrapper(
                gm=gm,
                op_name=f"{wrapper_name}_input",
                op_type=OpType.INPUT,
                op_target=None,
                op_id=op_id,
                plugin_manager=plugin_manager,
            ))
            transform_map[id(node)] = new_graph.call_module(
                wrapper_name,
                (new_placeholder_node,),
                {},
            )
            op_id += 1
        elif node.op == "call_function" and _computes_nothing(node.target):
            # Not an operation with a cost to attribute; see _computes_nothing.
            transform_map[id(node)] = new_graph.node_copy(
                node,
                arg_transform=lambda x: _arg_transform(x, transform_map)
            )

        elif node.op == "call_function":
            # Wrap call_function nodes
            wrapper_name = f"{WRAPPER_PREFIX}{op_id}"
            setattr(
                gm,
                wrapper_name,
                OpPluggableWrapper(
                    gm=gm,
                    op_name=f"{wrapper_name}_{node.target}",
                    op_type=OpType.FUNCTION,
                    op_target=node.target,
                    op_id=op_id,
                    plugin_manager=plugin_manager,
                )
            )
            
            # Create new node in new graph that calls the wrapper module
            transform_map[id(node)] = new_graph.call_module(
                wrapper_name,
                tuple(_arg_transform(arg, transform_map) for arg in node.args),
                {k: _arg_transform(v, transform_map) for k, v in node.kwargs.items()},
            )
            
            logger.debug(f"Wrapped op {op_id}: {node.name} (call_function)")
            op_id += 1
        
        elif node.op == "call_method":
            # Wrap call_method nodes
            wrapper_name = f"{WRAPPER_PREFIX}{op_id}"
            setattr(
                gm,
                wrapper_name,
                OpPluggableWrapper(
                    gm=gm,
                    op_name=f"{wrapper_name}_{node.target}",
                    op_type=OpType.METHOD,
                    op_target=node.target,
                    op_id=op_id,
                    plugin_manager=plugin_manager,
                )
            )
            
            transform_map[id(node)] = new_graph.call_module(
                wrapper_name,
                tuple(_arg_transform(arg, transform_map) for arg in node.args),
                {k: _arg_transform(v, transform_map) for k, v in node.kwargs.items()},
            )
            
            logger.debug(f"Wrapped op {op_id}: {node.name} (call_method)")
            op_id += 1
        
        elif node.op == "call_module":
            # Wrap call_module nodes
            wrapper_name = f"{WRAPPER_PREFIX}{op_id}"
            setattr(
                gm,
                wrapper_name,
                OpPluggableWrapper(
                    gm=gm,
                    op_name=f"{wrapper_name}_{node.target}",
                    op_type=OpType.MODULE,
                    op_target=node.target,
                    op_id=op_id,
                    plugin_manager=plugin_manager,
                )
            )
            
            transform_map[id(node)] = new_graph.call_module(
                wrapper_name,
                tuple(_arg_transform(arg, transform_map) for arg in node.args),
                {k: _arg_transform(v, transform_map) for k, v in node.kwargs.items()},
            )
            
            logger.debug(f"Wrapped op {op_id}: {node.name} (call_module)")
            op_id += 1
        elif node.op == "get_attr":
            # Not an operation, so there is nothing to instrument: this is a
            # reference to something the module already holds. Usually a
            # parameter or buffer; also the subgraph a higher-order op will
            # call, which is how dynamo represents an autograd.Function.
            #
            # It still has to be copied. Dropping it leaves the generated code
            # referring to a name nothing ever bound -- megatron's decoder came
            # back as "UnboundLocalError: local variable 'fwd_body_0'
            # referenced before assignment", which reads like a torch bug and
            # is this pass losing a node.
            transform_map[id(node)] = new_graph.node_copy(
                node,
                arg_transform=lambda x: _arg_transform(x, transform_map)
            )

        elif node.op == "output":
            all_args = [_arg_transform(arg, transform_map) for arg in extract_tensors(node.args)]
            for arg in all_args:
                wrapper_name = f"{WRAPPER_PREFIX}{op_id}"
                setattr(gm, wrapper_name, OpPluggableWrapper(
                    gm=gm,
                    op_name=f"{wrapper_name}_output",
                    op_type=OpType.OUTPUT,
                    op_target=None,
                    op_id=op_id,
                    plugin_manager=plugin_manager,
                ))
                new_graph.call_module(
                    wrapper_name,
                    (arg,),
                    {},
                )
                op_id += 1
            transform_map[id(node)] = new_graph.node_copy(
                node,
                arg_transform=lambda x: _arg_transform(x, transform_map)
            )

        else:
            # FX has exactly six node kinds and all of them are handled above.
            # Falling through used to mean the node was dropped without a word,
            # which produces a graph that looks fine and cannot run.
            raise RuntimeError(
                f"pluggable_pass does not know what to do with a {node.op!r} "
                f"node ({node.name}); it would otherwise be dropped silently"
            )

    # Replace old graph with new graph
    logger.debug(f"Transformed graph:\n{new_graph}")
    gm.graph = new_graph
    gm.recompile()
    
    logger.info(f"Pluggable pass complete: wrapped {op_id} operations")
    
    return gm

