"""Graph matching module for finding equivalent subgraphs in dataflow DAGs."""

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

