"""
Implementation of Dijkstra's algorithm for finding shortest paths.

This module provides an implementation of Dijkstra's algorithm for finding
the shortest path between two nodes in a graph. The implementation supports:
- Custom edge weight functions
- Path validation
- Performance monitoring
- Result caching

The algorithm uses a priority queue to efficiently find the shortest path,
considering either default unit weights or custom weight functions.

Example:
    >>> finder = ShortestPathFinder(graph)
    >>> result = finder.find_path(
    ...     "A", "B",
    ...     weight_func=lambda e: e.metadata.weight
    ... )
    >>> print(f"Path length: {result.length}")
"""

from collections import defaultdict
from heapq import heappop, heappush
from time import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from ....core.exceptions import GraphOperationError
from ....core.models import Edge
from ..base import PathFinder
from ..cache import PathCache
from ..models import PathResult, PerformanceMetrics
from ..utils import WeightFunc, create_path_result, get_edge_weight


class ShortestPathFinder(PathFinder):
    """
    Implements Dijkstra's algorithm for finding shortest paths.

    This implementation uses a priority queue to efficiently find the
    shortest path between two nodes in the graph, optionally using
    custom edge weights. The algorithm guarantees finding the shortest
    path in terms of total edge weight.

    Features:
        - Priority queue for efficient path finding
        - Support for custom weight functions
        - Result caching for improved performance
        - Path validation
        - Performance monitoring

    Example:
        >>> finder = ShortestPathFinder(graph)
        >>> # Find path with impact score as weight
        >>> result = finder.find_path(
        ...     "A", "B",
        ...     weight_func=lambda e: 1.0 / e.impact_score
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
    ) -> PathResult:
        """
        Find shortest path between nodes using Dijkstra's algorithm.

        This method implements Dijkstra's algorithm to find the shortest path
        between two nodes. It uses a priority queue to efficiently explore
        the graph, considering edge weights either from a custom weight
        function or using default unit weights.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Not used in shortest path
            max_paths: Not used in shortest path
            filter_func: Not used in shortest path
            weight_func: Optional function for edge weights

        Returns:
            PathResult containing the shortest path

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            GraphOperationError: If no path exists between the nodes

        Example:
            >>> # Find path minimizing metadata weight
            >>> result = finder.find_path(
            ...     "A", "B",
            ...     weight_func=lambda e: e.metadata.weight
            ... )
            >>> print(f"Total weight: {result.total_weight}")
        """
        metrics = PerformanceMetrics(operation="shortest_path", start_time=time())
        metrics.nodes_explored = 0

        try:
            # Check cache
            cache_key = PathCache.get_cache_key(
                start_node, end_node, "shortest", weight_func.__name__ if weight_func else None
            )
            cached_result = PathCache.get(cache_key)
            if cached_result is not None:
                metrics.cache_hit = True
                cached_result.validate(self.graph, weight_func)
                return cached_result

            self.validate_nodes(start_node, end_node)

            # Initialize Dijkstra's algorithm
            distances: Dict[str, float] = defaultdict(lambda: float("infinity"))
            distances[start_node] = 0
            previous: Dict[str, Optional[str]] = defaultdict(lambda: None)
            visited: Set[str] = set()
            pq: List[Tuple[float, str]] = [(0, start_node)]

            # Find shortest path
            while pq:
                current_distance, current_node = heappop(pq)
                metrics.nodes_explored += 1

                if current_node == end_node:
                    break

                if current_node in visited:
                    continue

                visited.add(current_node)

                for neighbor in self.graph.get_neighbors(current_node):
                    if neighbor in visited:
                        continue

                    edge = self.graph.get_edge(current_node, neighbor)
                    if edge is None:
                        continue

                    edge_weight = get_edge_weight(edge, weight_func)
                    new_distance = current_distance + edge_weight

                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        heappush(pq, (new_distance, neighbor))

            if distances[end_node] == float("infinity"):
                raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

            # Reconstruct path
            path: List[Edge] = []
            current = end_node
            while current != start_node:
                prev = previous[current]
                if prev is None:
                    break
                edge = self.graph.get_edge(prev, current)
                if edge is None:
                    raise GraphOperationError("Path reconstruction failed")
                path.append(edge)
                current = prev

            path.reverse()
            result = create_path_result(path, weight_func)
            result.validate(self.graph, weight_func)
            PathCache.put(cache_key, result)

            metrics.path_length = len(path)
            return result

        finally:
            metrics.end_time = time()
