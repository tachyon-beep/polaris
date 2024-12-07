"""
Graph traversal functionality.

This module is deprecated. Use polaris.core.graph.traversal instead.
"""

import warnings
from typing import List, Optional, Protocol
from ..models.edge import Edge
from .base import BaseGraph


class PathFinder(Protocol):
    """
    Protocol for path finding algorithms.

    @deprecated Use polaris.core.graph.traversal.PathFinder instead.
    """

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[Edge]]:
        """Find paths between nodes in the graph."""
        ...


warnings.warn(
    "The polaris.core.graph.traversal module is deprecated. "
    "Use polaris.core.graph.traversal package instead.",
    DeprecationWarning,
    stacklevel=2,
)
