"""
Enhanced implementation of shortest path algorithms.

This module provides optimized implementations of shortest path algorithms including:
- Dijkstra's algorithm with decrease-key operation
- Bellman-Ford for negative weights
- A* with landmarks (ALT)
- Memory-efficient path tracking
- Comprehensive validation
"""

from time import time
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union, Iterator

from ....core.exceptions import (
    GraphOperationError,
    NodeNotFoundError,
)
from ....core.models import Edge
from ..base import PathFinder, PathFilter
from ..cache import PathCache
from ..models import PathResult, PerformanceMetrics
from ..utils import (
    WeightFunc,
    PriorityQueue,
    MemoryManager,
    PathState,
    create_path_result,
    get_edge_weight,
    validate_path,
    timer,
)


class NegativeWeightError(GraphOperationError):
    """Raised when negative weights are detected in the graph."""

    pass


class ShortestPathFinder(PathFinder):
    """
    Enhanced shortest path implementation with multiple algorithms.

    Features:
    - Efficient decrease-key priority queue
    - Memory management and monitoring
    - Negative cycle detection
    - Path validation and metrics
    - A* with landmarks support
    """

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.landmarks: Dict[str, Dict[str, float]] = {}  # For A* with landmarks

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """
        Find shortest path using the most appropriate algorithm.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Optional maximum path length
            max_paths: Not used in shortest path (always returns single path)
            filter_func: Optional function to filter paths
            weight_func: Optional custom weight function
            **kwargs: Additional options:
                allow_negative: Whether to allow negative weights (default: False)
                use_astar: Whether to use A* with landmarks (default: False)
                validate: Whether to validate the result (default: True)

        Returns:
            PathResult containing the shortest path

        Raises:
            NodeNotFoundError: If nodes don't exist
            NegativeWeightError: If negative weights found
            MemoryError: If memory limit exceeded
        """
        # Extract additional options from kwargs
        allow_negative = kwargs.get("allow_negative", False)
        use_astar = kwargs.get("use_astar", False)
        validate = kwargs.get("validate", True)

        metrics = PerformanceMetrics(operation="shortest_path", start_time=time())

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
                if validate:
                    validate_path(cached.path, self.graph, weight_func)
                return cached

            # Check for negative weights
            if not allow_negative:
                for edge in self.graph.get_edges():
                    weight = get_edge_weight(edge, weight_func)
                    if weight < 0:
                        raise NegativeWeightError(
                            f"Negative weight {weight} found on edge "
                            f"{edge.from_entity} -> {edge.to_entity}"
                        )

            # Choose algorithm
            if allow_negative:
                result = self._bellman_ford(start_node, end_node, weight_func, max_length)
            elif use_astar and self.landmarks:
                result = self._astar(start_node, end_node, weight_func, max_length)
            else:
                result = self._dijkstra(start_node, end_node, weight_func, max_length)

            # Apply filter if provided
            if filter_func and not filter_func(result.path):
                raise GraphOperationError(
                    f"No path satisfying filter exists between {start_node} and {end_node}"
                )

            # Validate and cache result
            if validate:
                validate_path(result.path, self.graph, weight_func, max_length)
            PathCache.put(cache_key, result)
            return result

        finally:
            metrics.end_time = time()
            metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _dijkstra(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
    ) -> PathResult:
        """
        Enhanced Dijkstra's algorithm implementation.

        Uses:
        - Efficient decrease-key priority queue
        - Memory monitoring
        - Path state tracking
        """
        # Initialize data structures
        pq = PriorityQueue()
        pq.add_or_update(start_node, 0.0)

        distances: Dict[str, float] = {start_node: 0.0}
        states: Dict[str, PathState] = {start_node: PathState(start_node, None, None, 0, 0.0)}

        while not pq.empty():
            self.memory_manager.check_memory()

            current = pq.pop()
            if not current:
                break

            current_dist, current_node = current
            current_state = states[current_node]

            if current_node == end_node:
                path = current_state.get_path()
                return create_path_result(path, weight_func)

            if max_length and current_state.depth >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current_node):
                edge = self.graph.get_edge(current_node, neighbor)
                if not edge:
                    continue

                edge_weight = get_edge_weight(edge, weight_func)
                new_dist = current_dist + edge_weight

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    states[neighbor] = PathState(
                        neighbor, edge, current_state, current_state.depth + 1, new_dist
                    )
                    pq.add_or_update(neighbor, new_dist)

        raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

    def _bellman_ford(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
    ) -> PathResult:
        """
        Bellman-Ford algorithm for graphs with negative weights.

        Detects negative cycles and handles negative edge weights.
        """
        distances: Dict[str, float] = {start_node: 0.0}
        predecessors: Dict[str, Tuple[str, Edge]] = {}

        # Initialize distances
        for node in self.graph.get_nodes():
            if node != start_node:
                distances[node] = float("inf")

        # Relax edges |V| - 1 times
        nodes = list(self.graph.get_nodes())
        n = len(nodes)

        for i in range(n - 1):
            self.memory_manager.check_memory()
            relaxed = False

            for node in nodes:
                for neighbor in self.graph.get_neighbors(node):
                    edge = self.graph.get_edge(node, neighbor)
                    if not edge:
                        continue

                    weight = get_edge_weight(edge, weight_func)
                    if (
                        distances[node] != float("inf")
                        and distances[node] + weight < distances[neighbor]
                    ):
                        distances[neighbor] = distances[node] + weight
                        predecessors[neighbor] = (node, edge)
                        relaxed = True

            if not relaxed:
                break

        # Check for negative cycles
        for node in nodes:
            for neighbor in self.graph.get_neighbors(node):
                edge = self.graph.get_edge(node, neighbor)
                if not edge:
                    continue

                weight = get_edge_weight(edge, weight_func)
                if (
                    distances[node] != float("inf")
                    and distances[node] + weight < distances[neighbor]
                ):
                    raise NegativeWeightError("Negative cycle detected")

        # Reconstruct path
        if end_node not in predecessors:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        path = []
        current = end_node
        while current != start_node:
            prev, edge = predecessors[current]
            path.append(edge)
            current = prev

        path.reverse()
        return create_path_result(path, weight_func)

    def _astar(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
    ) -> PathResult:
        """
        A* algorithm with landmarks heuristic.

        Uses precomputed landmark distances for better heuristic.
        """

        def heuristic(node: str) -> float:
            """Compute admissible heuristic using landmarks."""
            if not self.landmarks:
                return 0.0

            h = 0.0
            for landmark, distances in self.landmarks.items():
                if node in distances and end_node in distances:
                    # Triangle inequality based lower bound
                    h = max(h, abs(distances[end_node] - distances[node]))
            return h

        pq = PriorityQueue()
        pq.add_or_update(start_node, heuristic(start_node))

        g_score: Dict[str, float] = {start_node: 0.0}
        states: Dict[str, PathState] = {start_node: PathState(start_node, None, None, 0, 0.0)}

        while not pq.empty():
            self.memory_manager.check_memory()

            current = pq.pop()
            if not current:
                break

            _, current_node = current
            current_state = states[current_node]

            if current_node == end_node:
                path = current_state.get_path()
                return create_path_result(path, weight_func)

            if max_length and current_state.depth >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current_node):
                edge = self.graph.get_edge(current_node, neighbor)
                if not edge:
                    continue

                edge_weight = get_edge_weight(edge, weight_func)
                tentative_g = g_score[current_node] + edge_weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    states[neighbor] = PathState(
                        neighbor, edge, current_state, current_state.depth + 1, tentative_g
                    )
                    pq.add_or_update(neighbor, f_score)

        raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

    def precompute_landmarks(self, num_landmarks: int = 16) -> None:
        """
        Precompute landmark distances for A* heuristic.

        Args:
            num_landmarks: Number of landmarks to select
        """
        # Select landmarks using degree centrality
        nodes = sorted(
            self.graph.get_nodes(),
            key=lambda n: len(list(self.graph.get_neighbors(n))),
            reverse=True,
        )
        landmarks = nodes[:num_landmarks]

        # Compute distances from landmarks
        for landmark in landmarks:
            distances = {}
            pq = PriorityQueue()
            pq.add_or_update(landmark, 0.0)

            while not pq.empty():
                current = pq.pop()
                if not current:
                    break

                dist, node = current
                if node in distances:
                    continue

                distances[node] = dist

                for neighbor in self.graph.get_neighbors(node):
                    edge = self.graph.get_edge(node, neighbor)
                    if not edge:
                        continue

                    weight = get_edge_weight(edge)
                    if neighbor not in distances:
                        pq.add_or_update(neighbor, dist + weight)

            self.landmarks[landmark] = distances
