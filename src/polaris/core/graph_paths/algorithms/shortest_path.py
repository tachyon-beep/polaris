"""
Enhanced implementation of shortest path algorithms.
"""

from time import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from contextlib import contextmanager

from ....core.exceptions import GraphOperationError, NodeNotFoundError
from ....core.models import Edge
from ..base import PathFinder, PathFilter, WeightFunc
from ..cache import PathCache
from ..models import PathResult, PerformanceMetrics
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NegativeWeightError(GraphOperationError):
    """Raised when negative weights are detected in the graph."""

    pass


class ShortestPathFinder(PathFinder[PathResult]):
    """Enhanced shortest path implementation."""

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.landmarks: Dict[str, Dict[str, float]] = {}  # For A* with landmarks

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
    ) -> PathResult:
        """Find shortest path using the most appropriate algorithm."""
        # Extract additional options from kwargs
        allow_negative = kwargs.get("allow_negative", False)
        use_astar = kwargs.get("use_astar", False)
        validate = kwargs.get("validate", True)

        metrics = PerformanceMetrics(operation="shortest_path", start_time=time())
        metrics.nodes_explored = 0  # Initialize counter

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
                    if validate:
                        validate_path(cached.path, self.graph, weight_func)
                    return cached

                # Check for negative weights
                if not allow_negative:
                    for edge in self.graph.get_edges():
                        try:
                            weight = get_edge_weight(edge, weight_func)
                            if weight < 0:
                                raise NegativeWeightError(
                                    f"Negative weight {weight} found on edge "
                                    f"{edge.from_entity} -> {edge.to_entity}"
                                )
                        except ValueError as e:
                            if "Edge weight must be finite number" in str(e):
                                raise ValueError("Path cost exceeded maximum value")
                            raise NegativeWeightError(str(e))

                # Choose algorithm
                if allow_negative:
                    result = self._bellman_ford(
                        start_node, end_node, weight_func, max_length, metrics
                    )
                elif use_astar and self.landmarks:
                    result = self._astar(start_node, end_node, weight_func, max_length, metrics)
                else:
                    result = self._dijkstra(start_node, end_node, weight_func, max_length, metrics)

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
        metrics: PerformanceMetrics,
    ) -> PathResult:
        """Dijkstra's algorithm implementation."""
        logger.debug(f"\nStarting Dijkstra's algorithm from {start_node} to {end_node}")
        if weight_func:
            logger.debug("Using custom weight function")

        pq = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
        pq.add_or_update(start_node, 0.0)

        distances: Dict[str, float] = {start_node: 0.0}
        states: Dict[str, PathState] = {start_node: PathState(start_node, None, None, 0, 0.0)}
        nodes_explored = 0  # Local counter

        while not pq.empty():
            self.memory_manager.check_memory()
            nodes_explored += 1

            current = pq.pop()
            if not current:
                break

            current_dist, current_node = current
            current_state = states[current_node]

            logger.debug(f"\nVisiting node {current_node} with distance {current_dist}")

            if current_node == end_node:
                metrics.nodes_explored = nodes_explored  # Update metrics at the end
                path = current_state.get_path()
                logger.debug(f"Found path to end node: {[e.to_entity for e in path]}")
                logger.debug(f"Path weights: {[get_edge_weight(e, weight_func) for e in path]}")
                logger.debug(f"Total distance: {current_dist}")
                return create_path_result(path, weight_func)

            if max_length and current_state.depth >= max_length:
                logger.debug(f"Skipping node {current_node} - max length reached")
                continue

            for neighbor in self.graph.get_neighbors(current_node):
                edge = self.graph.get_edge(current_node, neighbor)
                if not edge:
                    continue

                try:
                    edge_weight = get_edge_weight(edge, weight_func)
                    new_dist = current_dist + edge_weight

                    logger.debug(f"  Checking neighbor {neighbor}:")
                    logger.debug(f"    Edge weight: {edge_weight}")
                    logger.debug(f"    New distance: {new_dist}")
                    if neighbor in distances:
                        logger.debug(f"    Current best distance: {distances[neighbor]}")

                    if neighbor not in distances or is_better_cost(new_dist, distances[neighbor]):
                        logger.debug(f"    Updating distance to {neighbor}: {new_dist}")
                        distances[neighbor] = new_dist
                        states[neighbor] = PathState(
                            neighbor, edge, current_state, current_state.depth + 1, new_dist
                        )
                        pq.add_or_update(neighbor, new_dist)
                    else:
                        logger.debug(
                            f"    Keeping current distance to {neighbor}: {distances[neighbor]}"
                        )
                except (ValueError, OverflowError) as e:
                    # Log but continue with other neighbors
                    logger.debug(f"    Error processing edge: {str(e)}")
                    continue

        metrics.nodes_explored = nodes_explored  # Update metrics before raising error
        raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

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
        nodes_explored = 0  # Local counter

        # Initialize distances
        for node in self.graph.get_nodes():
            if node != start_node:
                distances[node] = float("inf")

        # Relax edges |V| - 1 times
        nodes = list(self.graph.get_nodes())
        n = len(nodes)

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
                            distances[neighbor] = distances[node] + weight
                            predecessors[neighbor] = (node, edge)
                            relaxed = True
                    except (ValueError, OverflowError) as e:
                        continue

            if not relaxed:
                break

        metrics.nodes_explored = nodes_explored  # Update metrics

        # Check for negative cycles
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
                        raise NegativeWeightError("Negative cycle detected")
                except ValueError as e:
                    continue

        # Reconstruct path
        if end_node not in predecessors:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        path = []
        current = end_node
        path_length = 0
        while current != start_node:
            if max_length and path_length >= max_length:
                raise GraphOperationError(f"No path of length <= {max_length} exists")
            prev, edge = predecessors[current]
            path.append(edge)
            current = prev
            path_length += 1

        path.reverse()
        return create_path_result(path, weight_func)

    def _astar(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
        metrics: PerformanceMetrics,
    ) -> PathResult:
        """A* algorithm implementation."""

        def heuristic(node: str) -> float:
            """Compute admissible heuristic using landmarks."""
            if not self.landmarks:
                return 0.0

            h = 0.0
            for landmark, distances in self.landmarks.items():
                if node in distances and end_node in distances:
                    h = max(h, abs(distances[end_node] - distances[node]))
            return h

        pq = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
        pq.add_or_update(start_node, heuristic(start_node))

        g_score: Dict[str, float] = {start_node: 0.0}
        states: Dict[str, PathState] = {start_node: PathState(start_node, None, None, 0, 0.0)}
        nodes_explored = 0  # Local counter

        while not pq.empty():
            self.memory_manager.check_memory()
            nodes_explored += 1

            current = pq.pop()
            if not current:
                break

            _, current_node = current
            current_state = states[current_node]

            if current_node == end_node:
                metrics.nodes_explored = nodes_explored  # Update metrics before returning
                path = current_state.get_path()
                return create_path_result(path, weight_func)

            if max_length and current_state.depth >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current_node):
                edge = self.graph.get_edge(current_node, neighbor)
                if not edge:
                    continue

                try:
                    edge_weight = get_edge_weight(edge, weight_func)
                    tentative_g = g_score[current_node] + edge_weight

                    if neighbor not in g_score or is_better_cost(tentative_g, g_score[neighbor]):
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor)
                        states[neighbor] = PathState(
                            neighbor, edge, current_state, current_state.depth + 1, tentative_g
                        )
                        pq.add_or_update(neighbor, f_score)
                except (ValueError, OverflowError) as e:
                    continue

        metrics.nodes_explored = nodes_explored  # Update metrics before raising error
        raise GraphOperationError(f"No path exists between {start_node} and {end_node}")
