"""Enhanced implementation of shortest path algorithms."""

from time import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from contextlib import contextmanager

from ...exceptions import GraphOperationError, NodeNotFoundError
from ...models import Edge
from ..base import PathFinder
from ..cache import PathCache
from ..models import PathResult, PathValidationError, PerformanceMetrics
from ..types import WeightFunc, PathFilter
from ..utils import (
    PriorityQueue,
    MemoryManager,
    PathState,
    create_path_result,
    get_edge_weight,
    validate_path,
    timer,
    is_better_cost,
    MAX_QUEUE_SIZE,
)


class ShortestPathFinder(PathFinder[PathResult]):
    """Enhanced shortest path implementation."""

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)

    @contextmanager
    def _search_context(self):
        """Context manager for search state."""
        try:
            yield
        finally:
            self.memory_manager.reset_peak_memory()

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
        """Find shortest path using the most appropriate algorithm."""
        metrics = PerformanceMetrics(operation="shortest_path", start_time=time())
        metrics.nodes_explored = 0

        with self._search_context():
            try:
                # Validate nodes
                self.validate_nodes(start_node, end_node)

                # Check cache
                cache_key = PathCache.get_cache_key(
                    start_node, end_node, "shortest", weight_func.__name__ if weight_func else None
                )
                cached = PathCache.get(cache_key)
                if cached:
                    metrics.cache_hit = True
                    # Validate cached path still meets constraints
                    try:
                        validate_path(cached.path, self.graph, weight_func, max_length)
                        return cached
                    except (PathValidationError, ValueError):
                        # Cached path no longer valid, continue with search
                        pass

                # Use Bellman-Ford as our default path finding algorithm
                result = self._bellman_ford(start_node, end_node, weight_func, max_length, metrics)

                # Apply filter if provided
                if filter_func and not filter_func(result.path):
                    return None

                # Validate and cache result
                validate_path(result.path, self.graph, weight_func, max_length)
                PathCache.put(cache_key, result)
                return result

            except ValueError as e:
                if "Edge weight must be finite number" in str(e):
                    raise ValueError("Path cost exceeded maximum value")
                raise
            finally:
                metrics.end_time = time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _bellman_ford(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
        metrics: PerformanceMetrics,
    ) -> PathResult:
        """Bellman-Ford algorithm implementation."""
        distances: Dict[str, float] = {start_node: 0.0}
        predecessors: Dict[str, Tuple[str, Edge]] = {}
        nodes_explored = 0

        # Initialize distances
        for node in self.graph.get_nodes():
            if node != start_node:
                distances[node] = float("inf")

        # Relax edges |V| - 1 times
        nodes = list(self.graph.get_nodes())
        n = len(nodes)

        # Track shortest path length found
        min_path_length = float("inf")

        for i in range(n - 1):
            self.memory_manager.check_memory()
            nodes_explored += len(nodes)
            relaxed = False

            for node in nodes:
                for neighbor in self.graph.get_neighbors(node):
                    edge = self.graph.get_edge(node, neighbor)
                    if not edge:
                        continue

                    try:
                        weight = get_edge_weight(edge, weight_func)
                        if distances[node] != float("inf") and is_better_cost(
                            distances[node] + weight, distances[neighbor]
                        ):
                            # Check path length before updating
                            path_length = len(self._reconstruct_path(predecessors, node)) + 1
                            if max_length is not None and path_length > max_length:
                                continue

                            distances[neighbor] = distances[node] + weight
                            predecessors[neighbor] = (node, edge)
                            relaxed = True

                            # Update shortest path length if this is the target
                            if neighbor == end_node:
                                min_path_length = min(min_path_length, path_length)

                    except ValueError as e:
                        if "Edge weight must be finite number" in str(e):
                            raise ValueError("Path cost exceeded maximum value")
                        raise ValueError(f"Invalid edge weight: {str(e)}")

            if not relaxed:
                break

        metrics.nodes_explored = nodes_explored

        # Check if we found a path within length constraints
        if end_node not in predecessors:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        if max_length is not None and min_path_length > max_length:
            raise GraphOperationError(f"No path of length <= {max_length} exists")

        path = self._reconstruct_path(predecessors, end_node)
        return create_path_result(path, weight_func)

    def _reconstruct_path(
        self, predecessors: Dict[str, Tuple[str, Edge]], end_node: str
    ) -> List[Edge]:
        """Helper to reconstruct path from predecessors."""
        path = []
        current = end_node
        while current in predecessors:
            prev, edge = predecessors[current]
            path.append(edge)
            current = prev
        path.reverse()
        return path
