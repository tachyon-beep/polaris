"""
Graph traversal functionality.

This module is deprecated. Use polaris.core.graph_paths for path finding functionality.
"""

from typing import List, Optional, Protocol
from ..models.edge import Edge
from .base import BaseGraph


class PathFinder(Protocol):
    """
    Protocol for path finding algorithms.

    @deprecated Use polaris.core.graph_paths.PathFinder instead.
    """

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[Edge]]:
        """Find paths between nodes in the graph."""
        ...
