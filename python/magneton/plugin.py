"""What a plugin looks like, and what runs a set of them."""

import logging
from typing import Any, Callable, List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# What a plugin's priority is taken to be when it does not say. Structural
# implementers inherit no default, so the manager fills this in for them.
DEFAULT_PRIORITY = 100


@runtime_checkable
class OpPlugin(Protocol):
    """The three points at which a plugin sees an operation."""

    def before_execute(
        self,
        op_id: int,
        op_name: str,
        args: tuple,
        kwargs: dict
    ) -> dict:
        """Called before the operation runs."""
        raise NotImplementedError

    def after_execute(
        self,
        op_id: int,
        op_name: str,
        output: Any,
        context: dict
    ) -> Any:
        """Called after the operation runs."""
        raise NotImplementedError

    def wrap_execute(
        self,
        op_callable: Callable,
        context: dict
    ) -> Any:
        """Run the operation, or decide how it is run."""
        raise NotImplementedError



class PluginManager:
    """Manages multiple plugins and coordinates their execution."""
    
    def __init__(self, plugins: List[OpPlugin]):
        """Initialize plugin manager."""
        if not plugins:
            logger.warning("PluginManager created with no plugins")
        
        self.plugins = sorted(plugins, key=lambda p: getattr(p, "priority", DEFAULT_PRIORITY))
        
        logger.debug(
            f"PluginManager initialized with {len(self.plugins)} plugins: "
            f"{[p.__class__.__name__ for p in self.plugins]}"
        )
    
    def before_execute(
        self,
        op_id: int,
        op_name: str,
        args: tuple,
        kwargs: dict
    ) -> dict:
        """Execute all plugins' before_execute in priority order."""
        combined_context = {}
        
        for plugin in self.plugins:
            plugin_name = plugin.__class__.__name__
            try:
                plugin_context = plugin.before_execute(op_id, op_name, args, kwargs)
                combined_context[plugin_name] = plugin_context
            except Exception as e:
                logger.error(
                    f"Error in {plugin_name}.before_execute for op {op_name}: {e}"
                )
                combined_context[plugin_name] = {}
        
        return combined_context
    
    def after_execute(
        self,
        op_id: int,
        op_name: str,
        output: Any,
        context: dict
    ) -> Any:
        """Execute all plugins' after_execute in reverse priority order."""
        # Execute in reverse order (high priority to low priority)
        for plugin in reversed(self.plugins):
            plugin_name = plugin.__class__.__name__
            plugin_context = context.get(plugin_name, {})
            
            try:
                output = plugin.after_execute(op_id, op_name, output, plugin_context)
            except Exception as e:
                logger.error(
                    f"Error in {plugin_name}.after_execute for op {op_name}: {e}"
                )
        
        return output
    
    def wrap_execute(
        self,
        op_callable: Callable,
        context: dict
    ) -> Any:
        """Wrap execution with all plugins (nested by priority)."""
        # Build nested wrappers from lowest priority to highest
        # This ensures highest priority plugins wrap outermost
        wrapped_callable = op_callable
        
        for plugin in reversed(self.plugins):
            plugin_name = plugin.__class__.__name__
            plugin_context = context.get(plugin_name, {})
            
            # Create a closure that captures the current state
            # This is necessary to avoid late binding issues
            def make_wrapper(p, pc, wc):
                def wrapper():
                    try:
                        return p.wrap_execute(wc, pc)
                    except Exception as e:
                        logger.error(
                            f"Error in {p.__class__.__name__}.wrap_execute: {e}"
                        )
                        # Fallback: just execute the wrapped callable
                        return wc()
                return wrapper
            
            wrapped_callable = make_wrapper(plugin, plugin_context, wrapped_callable)
        
        return wrapped_callable()
