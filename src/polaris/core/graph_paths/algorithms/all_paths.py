"""All paths finding implementation."""

from typing import Dict, List, Optional, Set, Iterator, Any
from time import time

from ...exceptions import GraphOperationError
from ..base import PathFinder
from ..models import PathResult, PerformanceMetrics
from ..types import WeightFunc, PathFilter
from ..utils import (
    create_path_result,
    get_edge_weight,
    MAX_QUEUE_SIZE,
    MemoryManager,
)
from ...models import Edge


class AllPathsFinder(PathFinder[PathResult]):
    """All paths finding implementation."""

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Optional[PathResult]:
        """Find a single path between nodes.

        For AllPathsFinder, this returns the first path found.
        """
        paths = self.find_paths(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=1,
            filter_func=filter_func,
            weight_func=weight_func,
            **kwargs,
        )
        try:
            return next(paths)
        except StopIteration:
            return None

    def find_paths(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Iterator[PathResult]:
        """Find all paths between nodes using DFS."""
        # Validate parameters
        if max_paths is not None and max_paths <= 0:
            raise ValueError("max_paths must be positive")

        # Validate nodes exist
        self.validate_nodes(start_node, end_node)

        # Special case: start_node equals end_node
        if start_node == end_node:
            yield create_path_result([], weight_func)
            return

        # Initialize search state
        current_path: List[Edge] = []
        path_nodes: Set[str] = set()  # Track nodes in current path only
        paths_found = 0

        def dfs(node: str) -> Iterator[PathResult]:
            """Depth-first search implementation."""
            nonlocal paths_found

            # Check memory usage
            self.memory_manager.check_memory()

            # Base case: reached max paths
            if max_paths is not None and paths_found >= max_paths:
                return

            # Base case: found end node
            if node == end_node:
                result = create_path_result(current_path.copy(), weight_func)
                if filter_func is None or filter_func(result.path):
                    paths_found += 1
                    yield result
                return

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(node):
                # Skip if neighbor is already in current path (avoid cycles)
                if neighbor in path_nodes:
                    continue

                # Skip if adding this edge would exceed max_length
                if max_length is not None and len(current_path) >= max_length:
                    continue

                edge = self.graph.get_edge(node, neighbor)
                if not edge:
                    continue

                # Add edge and neighbor to current path
                current_path.append(edge)
                path_nodes.add(neighbor)

                # Recursively explore from neighbor
                yield from dfs(neighbor)

                # Backtrack
                current_path.pop()
                path_nodes.remove(neighbor)

        # Start search from start node
        path_nodes.add(start_node)
        yield from dfs(start_node)
