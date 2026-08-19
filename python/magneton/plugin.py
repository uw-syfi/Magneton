"""What a plugin looks like, and what runs a set of them.

`OpPlugin` is a Protocol rather than a base class, and deliberately so. The
transform in `magneton.transform` is what drives plugins, but not every plugin
belongs to magneton: a profiler that wants to replay operations to collect
enough power samples is doing measurement, not semantics, and it should be able
to supply that plugin without importing magneton to inherit from it.

Structural typing gives both ends what they need. Saying `class Mine(OpPlugin)`
still works and documents the intent; a class that simply defines the three
methods is equally acceptable and depends on nothing. `PluginManager` calls
only what is declared here.

The method bodies raise rather than being the usual `...`. A Protocol does not
enforce implementation the way an ABC did, so a subclass that forgets one would
otherwise inherit an empty body and silently return None -- which, for
`wrap_execute`, would replace the operation's output.
"""

import logging
from typing import Any, Callable, List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# What a plugin's priority is taken to be when it does not say. Structural
# implementers inherit no default, so the manager fills this in for them.
DEFAULT_PRIORITY = 100


@runtime_checkable
class OpPlugin(Protocol):
    """The three points at which a plugin sees an operation.

    - `before_execute` runs first and returns whatever the plugin wants handed
      back to it afterwards.
    - `wrap_execute` decides *how* the operation runs, and may run it more than
      once or not at all.
    - `after_execute` sees the output and may replace it.

    Example:
        >>> class Timing(OpPlugin):
        ...     def before_execute(self, op_id, op_name, args, kwargs):
        ...         return {"start": time.perf_counter()}
        ...
        ...     def wrap_execute(self, op_callable, context):
        ...         return op_callable()
        ...
        ...     def after_execute(self, op_id, op_name, output, context):
        ...         print(op_name, time.perf_counter() - context["start"])
        ...         return output
    """

    def before_execute(
        self,
        op_id: int,
        op_name: str,
        args: tuple,
        kwargs: dict
    ) -> dict:
        """Called before the operation runs.

        Args:
            op_id: Sequential operation ID
            op_name: Operation name (e.g., "add_1")
            args: Positional arguments to the operation
            kwargs: Keyword arguments to the operation

        Returns:
            Context handed back to this plugin's `after_execute` and
            `wrap_execute`. Anything the plugin needs to remember goes here
            rather than on the plugin, which is shared across operations.
        """
        raise NotImplementedError

    def after_execute(
        self,
        op_id: int,
        op_name: str,
        output: Any,
        context: dict
    ) -> Any:
        """Called after the operation runs.

        Args:
            op_id: Sequential operation ID
            op_name: Operation name
            output: What the operation produced
            context: The dict returned by this plugin's `before_execute`

        Returns:
            The output, modified or not. A plugin that only observes must
            still return what it was given.
        """
        raise NotImplementedError

    def wrap_execute(
        self,
        op_callable: Callable,
        context: dict
    ) -> Any:
        """Run the operation, or decide how it is run.

        The replay plugin executes `op_callable` many times so NVML has enough
        power samples to integrate; a caching plugin might not call it at all.
        A plugin with no opinion returns `op_callable()`.

        Plugins nest by priority, so the highest-priority one is outermost.

        Args:
            op_callable: Executes the operation once
            context: The dict returned by this plugin's `before_execute`

        Returns:
            The operation's output.
        """
        raise NotImplementedError

    # `priority` is an optional fourth member, and is deliberately not declared
    # here. A runtime_checkable Protocol requires every member it names to be
    # present, so declaring it would make `isinstance(plugin, OpPlugin)` reject
    # exactly the plugins this protocol exists to admit -- ones that implement
    # the three hooks and have no opinion about ordering. `PluginManager`
    # reads it with a DEFAULT_PRIORITY fallback instead.
    #
    #     @property
    #     def priority(self) -> int:
    #         return 10
    #
    # Lower runs earlier and outermost: `before_execute` low to high,
    # `after_execute` high to low, `wrap_execute` nesting so a low number wraps
    # a high one. It matters when one plugin's work must not be visible to
    # another's -- recording the dataflow (10) has to see the real single
    # execution, before replay (50) starts running the operation repeatedly.
    # Roughly: 0-20 must-be-first, 21-50 measurement, 51-100 ordinary, 101+
    # last. Unset means DEFAULT_PRIORITY.


class PluginManager:
    """
    Manages multiple plugins and coordinates their execution.
    
    The manager ensures plugins are executed in the correct order
    based on their priority.
    """
    
    def __init__(self, plugins: List[OpPlugin]):
        """
        Initialize plugin manager.
        
        Args:
            plugins: List of plugins to manage
        """
        if not plugins:
            logger.warning("PluginManager created with no plugins")
        
        # Lower priority runs earlier. `priority` is defaulted rather than
        # required: a plugin that implements the protocol structurally, without
        # inheriting from OpPlugin, has no default to inherit.
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
        """
        Execute all plugins' before_execute in priority order.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            Combined context dict from all plugins.
            Each plugin's context is namespaced by plugin class name.
        """
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
        """
        Execute all plugins' after_execute in reverse priority order.
        
        Args:
            op_id: Operation ID
            op_name: Operation name
            output: Operation output
            context: Combined context from before_execute
        
        Returns:
            Final output (potentially modified by plugins)
        """
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
        """
        Wrap execution with all plugins (nested by priority).
        
        High priority plugins wrap low priority plugins.
        
        Example with 3 plugins (priorities 10, 30, 50):
            Plugin10.wrap_execute(
                Plugin30.wrap_execute(
                    Plugin50.wrap_execute(
                        op_callable
                    )
                )
            )
        
        Args:
            op_callable: The operation to execute
            context: Combined context from before_execute
        
        Returns:
            Operation output
        """
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
