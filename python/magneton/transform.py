"""Pluggable FX graph transformation system."""

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


WRAPPER_PREFIX = "op_wrapper_"
ANNOTATION_PREFIX = f"forward_{WRAPPER_PREFIX}"


class OpPluggableWrapper(OpWrapper):
    """Wrapper that applies plugins to operation execution."""
    
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        op_name: str,
        op_type: OpType,
        op_target: Any,
        op_id: int,
        plugin_manager: PluginManager,
    ):
        """Initialize pluggable wrapper."""
        super().__init__(gm, op_name, op_type, op_target)
        self.op_name = op_name
        self.op_type = op_type
        self.op_target = op_target
        self.op_id = op_id
        self.plugin_manager = plugin_manager
    
    def forward(self, *args, **kwargs) -> Any:
        """Execute operation with plugin hooks."""
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
    """Whether a `call_function` node is something to leave alone."""

    if inspect.isclass(target):
        return True
    try:
        from torch._ops import HigherOrderOperator
    except ImportError:  # pragma: no cover - very old torch
        return False
    return isinstance(target, HigherOrderOperator)


def _arg_transform(arg: Any, transform_map: Dict[int, Any]) -> Any:
    """Recursively transform arguments to reference new nodes."""
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
    """Apply pluggable transformation to FX graph."""
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

