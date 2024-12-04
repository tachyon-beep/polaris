"""
Implementation of depth-first search for finding all possible paths.

This module provides an implementation of depth-first search to find all possible
paths between two nodes in a graph. The implementation supports various constraints
and features:
- Maximum path length limits
- Maximum number of paths limits
- Custom path filtering
- Cycle detection
- Custom edge weights
- Performance monitoring

The algorithm uses an iterative approach rather than recursion to handle deep
paths efficiently and avoid stack overflow.

Example:
    >>> finder = AllPathsFinder(graph)
    >>> paths = finder.find_path(
    ...     "A", "B",
    ...     max_length=5,
    ...     filter_func=lambda edges: sum(e.weight for e in edges) < 10
    ... )
    >>> for path in paths:
    ...     print(f"Path length: {path.length}")
"""

from time import time
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from ....core.models import Edge
from ..base import PathFinder
from ..models import PathResult, PerformanceMetrics
from ..utils import WeightFunc, create_path_result, get_edge_weight

# Constants
DEFAULT_MAX_PATH_LENGTH = 50  # Default limit to prevent unbounded recursion
DEFAULT_MAX_PATHS = 1000  # Default limit for number of paths to return


class AllPathsFinder(PathFinder):
    """
    Implements depth-first search for finding all possible paths.

    This implementation uses an iterative depth-first search approach to find
    all possible paths between two nodes, with support for various constraints
    and features:
    - Path length limits
    - Maximum number of paths
    - Custom path filtering
    - Cycle detection
    - Custom edge weights

    The algorithm maintains visited nodes to prevent cycles and supports
    early termination through various constraints.

    Features:
        - Iterative DFS implementation
        - Configurable constraints
        - Cycle prevention
        - Custom path filtering
        - Performance monitoring

    Example:
        >>> finder = AllPathsFinder(graph)
        >>> # Find paths with custom filter
        >>> paths = finder.find_path(
        ...     "A", "B",
        ...     max_length=5,
        ...     filter_func=lambda edges: all(e.impact_score > 0.5 for e in edges)
        ... )
    """

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Iterator[PathResult]:
        """
        Find all paths between nodes using depth-first search.

        This method implements an iterative depth-first search to find all
        possible paths between two nodes that satisfy the given constraints.
        It uses a stack-based approach to avoid recursion depth limits and
        maintains a visited set to prevent cycles.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length (default: DEFAULT_MAX_PATH_LENGTH)
            max_paths: Maximum number of paths (default: DEFAULT_MAX_PATHS)
            filter_func: Optional function to filter paths
            weight_func: Optional function for edge weights

        Yields:
            PathResult objects for each valid path found

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            ValueError: If max_length or max_paths is not positive

        Example:
            >>> # Find paths with length and filter constraints
            >>> paths = finder.find_path(
            ...     "A", "B",
            ...     max_length=3,
            ...     filter_func=lambda edges: sum(e.weight for e in edges) < 5
            ... )
            >>> for path in paths:
            ...     print(f"Path: {' -> '.join(path.nodes)}")
        """
        metrics = PerformanceMetrics(operation="all_paths", start_time=time())
        metrics.nodes_explored = 0

        try:
            self.validate_nodes(start_node, end_node)

            # Use default limits
            if max_length is None:
                max_length = DEFAULT_MAX_PATH_LENGTH
            if max_paths is None:
                max_paths = DEFAULT_MAX_PATHS

            if max_length <= 0:
                raise ValueError(f"max_length must be positive, got {max_length}")
            if max_paths <= 0:
                raise ValueError(f"max_paths must be positive, got {max_paths}")

            paths_found = 0
            # Stack items: (current_node, path_edges, visited_nodes, total_weight)
            stack: List[Tuple[str, List[Edge], Set[str], float]] = [
                (start_node, [], {start_node}, 0.0)
            ]

            while stack and paths_found < max_paths:
                current, path_edges, visited, total_weight = stack.pop()
                metrics.nodes_explored += 1

                # Check if we've reached the target
                if current == end_node:
                    if filter_func is None or filter_func(path_edges):
                        result = create_path_result(path_edges, weight_func, self.graph)
                        metrics.path_length = len(path_edges)
                        yield result
                        paths_found += 1
                    continue

                # Skip if path is too long
                if len(path_edges) >= max_length:
                    continue

                # Process neighbors in reverse sorted order for consistent DFS
                neighbors = sorted(self.graph.get_neighbors(current), reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        edge = self.graph.get_edge(current, neighbor)
                        if edge is None:
                            continue

                        edge_weight = get_edge_weight(edge, weight_func)
                        new_total_weight = total_weight + edge_weight

                        # Add edge to path and neighbor to visited set
                        new_path_edges = path_edges + [edge]
                        new_visited = visited | {neighbor}
                        stack.append((neighbor, new_path_edges, new_visited, new_total_weight))

        finally:
            metrics.end_time = time()
