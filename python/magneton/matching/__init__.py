"""
Graph matching module for finding equivalent subgraphs in dataflow DAGs.

This module provides utilities to compare two dataflow graphs and identify
pairs of subgraphs that are semantically equivalent (i.e., have equivalent
input/output tensor values).

Usage:
    >>> from magneton import Profiler, Config, DataflowConfig
    >>> from magneton.matching import SubgraphMatch, export_matches, print_matches
    >>> 
    >>> # Profile two models
    >>> config1 = Config(dataflow_config=DataflowConfig(record_dataflow=True))
    >>> with Profiler([ActivityType.CPU], model1, example_inputs, dataflow_config=config1.dataflow_config) as (prof1, compiled_model1):
    ...     compiled_model1(*inputs)
    >>> dag1 = prof1.dataflow_dag
    >>> 
    >>> # ... similar for model2 to get dag2 ...
    >>> 
    >>> # Find equivalent subgraphs (Phase 2 - algorithm not yet implemented)
    >>> # matches = match_graphs(dag1, dag2)
    >>> 
    >>> # Export results
    >>> # export_matches(matches, dag1, dag2, "matches.json")
    >>> # print_matches(matches, verbose=True)
"""

# Infrastructure (Phase 1)
from .subgraph import SubgraphMatch, export_matches, print_matches
from .dominator import DominatorTree

# Algorithm (Phase 2)
from .matcher import match_graphs, MatchConfig

__all__ = [
    # Phase 1 (infrastructure)
    "SubgraphMatch",
    "export_matches",
    "print_matches",
    "DominatorTree",
    # Phase 2 (algorithm)
    "match_graphs",
    "MatchConfig",
]

__version__ = "0.1.0"

