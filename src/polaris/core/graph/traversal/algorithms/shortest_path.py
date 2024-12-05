"""Enhanced implementation of shortest path algorithms."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

from polaris.core.exceptions import GraphOperationError, NodeNotFoundError
from ..base import PathFinder
from ..cache import PathCache
from ..path_models import PathResult, PathValidationError, PerformanceMetrics
from ..types import PathFilter, WeightFunc
from ..utils import (
    MAX_QUEUE_SIZE,
    MemoryManager,
    PathState,
    PriorityQueue,
    create_path_result,
    get_edge_weight,
    is_better_cost,
    timer,
    validate_path,
)
from polaris.core.models import Edge


class ShortestPathFinder(PathFinder[PathResult]):
    """Enhanced shortest path implementation."""

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)

    def _get_timeout(self, graph_size: int) -> float:
        """Calculate appropriate timeout based on graph size."""
        # Base timeout of 30 seconds
        base_timeout = 30.0
        # Add 5 seconds for every 1000 nodes
        size_factor = graph_size / 1000
        return base_timeout + (size_factor * 5.0)

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
        metrics = PerformanceMetrics(operation="shortest_path", start_time=time.time())
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

                # Calculate appropriate timeout based on graph size
                graph_size = len(list(self.graph.get_nodes()))
                timeout = self._get_timeout(graph_size)

                # Use Bellman-Ford as our default path finding algorithm
                result = self._bellman_ford(
                    start_node, end_node, weight_func, max_length, metrics, timeout
                )

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
                metrics.end_time = time.time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _bellman_ford(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
        metrics: PerformanceMetrics,
        timeout: float,
    ) -> PathResult:
        """Bellman-Ford algorithm implementation with optimizations."""
        distances: Dict[str, float] = {start_node: 0.0}
        predecessors: Dict[str, Tuple[str, Edge]] = {}
        nodes_explored = 0
        start_time = time.time()
        max_iterations = 1000  # Maximum number of iterations to prevent infinite loops

        # Initialize distances
        for node in self.graph.get_nodes():
            if node != start_node:
                distances[node] = float("inf")

        # Relax edges |V| - 1 times
        nodes = list(self.graph.get_nodes())
        n = len(nodes)
        iteration = 0

        while iteration < min(n - 1, max_iterations):
            self.memory_manager.check_memory()

            # Check timeout
            if time.time() - start_time > timeout:
                raise GraphOperationError("Path finding timeout exceeded")

            nodes_explored += len(nodes)
            relaxed = False

            for node in nodes:
                # Check if we've found a path to the end node within length constraints
                if end_node in predecessors:
                    path_length = len(self._reconstruct_path(predecessors, end_node))
                    if max_length is None or path_length <= max_length:
                        # Early exit if we've found a valid path and no better path is possible
                        if all(
                            distances[neighbor] >= distances[end_node]
                            for neighbor in self.graph.get_neighbors(node)
                        ):
                            metrics.nodes_explored = nodes_explored
                            path = self._reconstruct_path(predecessors, end_node)
                            return create_path_result(path, weight_func)

                for neighbor in self.graph.get_neighbors(node):
                    edge = self.graph.get_edge(node, neighbor)
                    if not edge:
                        continue

                    try:
                        weight = get_edge_weight(edge, weight_func)
                        new_distance = distances[node] + weight
                        # Use is_better_cost for consistent comparison
                        if distances[node] != float("inf") and is_better_cost(
                            new_distance, distances[neighbor]
                        ):
                            # Check path length before updating
                            path_length = len(self._reconstruct_path(predecessors, node)) + 1
                            if max_length is not None and path_length > max_length:
                                continue

                            distances[neighbor] = new_distance
                            predecessors[neighbor] = (node, edge)
                            relaxed = True

                    except ValueError as e:
                        if "Edge weight must be finite number" in str(e):
                            raise ValueError("Path cost exceeded maximum value")
                        raise ValueError(f"Invalid edge weight: {str(e)}")

            if not relaxed:
                # No relaxations occurred, we can stop early
                break

            iteration += 1

        metrics.nodes_explored = nodes_explored

        # Check if we found a path within length constraints
        if end_node not in predecessors:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        path = self._reconstruct_path(predecessors, end_node)
        if max_length is not None and len(path) > max_length:
            raise GraphOperationError(f"No path of length <= {max_length} exists")

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
