"""
Path finding implementation for Contraction Hierarchies.

This module handles path finding using the preprocessed contraction hierarchy,
including bidirectional search and path reconstruction.
"""

from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING
import logging
import statistics
import threading
import time
from copy import deepcopy
from collections import defaultdict
from heapq import heappop, heappush

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import (
    WeightFunc,
    create_path_result,
    get_edge_weight,
    validate_path,
)
from polaris.core.models import Edge
from .models import ContractionState, Shortcut
from .storage import ContractionStorage
from .utils import unpack_shortcut

if TYPE_CHECKING:
    from polaris.core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        with self._lock:
            self.metrics[name].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of recorded metrics."""
        stats = {}
        for name, values in self.metrics.items():
            if values:  # Only compute stats if we have values
                stats[name] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                }
        return stats


class ContractionPathFinder:
    """
    Handles path finding using Contraction Hierarchies.

    Features:
    - Bidirectional search
    - Path reconstruction
    - Shortcut unpacking
    - Cycle prevention
    - Thread safety
    - Performance monitoring
    - Efficient caching
    """

    def __init__(
        self,
        graph: "Graph",
        state: ContractionState,
        storage: ContractionStorage,
    ):
        """
        Initialize path finder.

        Args:
            graph: Graph instance
            state: Preprocessed algorithm state
            storage: Storage manager
        """
        self.graph = graph
        self.state = state
        self.storage = storage
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Initialize caches and indices
        self._shortcut_index = self._build_shortcut_index()
        self._neighbor_cache = {}
        self._performance = PerformanceMonitor()
        self._reset_search_state()

    def _reset_search_state(self):
        """Reset all search-related state between path finding operations."""
        self._forward_distances = {}
        self._backward_distances = {}
        self._forward_visited = set()
        self._backward_visited = set()
        self._path_nodes = set()

    def _build_shortcut_index(self) -> Dict[str, List[Tuple[str, Edge]]]:
        """Build efficient index for shortcut lookup."""
        index = defaultdict(list)
        for (u, v), shortcut in self.state.shortcuts.items():
            index[u].append((v, shortcut.edge))
            index[v].append((u, shortcut.edge))  # Bidirectional lookup
        return dict(index)

    def _get_edges_to_process(self, node: str, is_forward: bool) -> Iterator[Tuple[str, Edge]]:
        """Get all edges efficiently using cached indices."""
        # Get regular edges
        neighbors = self.graph.get_neighbors(node, reverse=not is_forward)
        for neighbor in neighbors:
            edge = self.graph.get_edge(
                node if is_forward else neighbor, neighbor if is_forward else node
            )
            if edge:
                yield neighbor, edge

        # Get shortcuts efficiently using index
        if node in self._shortcut_index:
            for target, edge in self._shortcut_index[node]:
                if is_forward:
                    yield target, edge
                else:
                    yield target, self._create_reverse_edge(edge)

    def _is_path_valid(self, node: str, new_dist: float, distances: Dict[str, float]) -> bool:
        """
        Validate path to prevent cycles and ensure optimality.
        Returns True if path is valid and optimal.
        """
        # Check if we already have a better path
        if node in distances and distances[node] <= new_dist:
            return False

        # Check for cycles using path tracking
        if node in self._path_nodes:
            return False

        # Check level constraints
        if self._violates_level_constraints(node):
            return False

        return True

    def _violates_level_constraints(self, node: str) -> bool:
        """Check if adding this node would violate hierarchy constraints."""
        if node not in self.state.node_level:
            return False

        current_level = self.state.node_level[node]
        path_levels = [self.state.node_level.get(n, 0) for n in self._path_nodes]

        # Enforce up-then-down pattern in hierarchy
        if path_levels:
            max_level = max(path_levels)
            if current_level > max_level:  # Still going up
                return False
            min_level_after_max = min(l for l in path_levels[path_levels.index(max_level) :])
            if current_level > min_level_after_max:  # Violates down phase
                return True
        return False

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """
        Find shortest path using contraction hierarchies.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists
        """
        start_time = time.time()

        try:
            with self._lock:  # Thread safety for entire path finding operation
                self._reset_search_state()

                if start_node == end_node:
                    return PathResult(path=[], total_weight=0.0)

                validate = kwargs.get("validate", True)
                debug = kwargs.get("debug", True)

                # Try all possible paths if filter is provided
                if filter_func:
                    path = self._find_filtered_path(
                        start_node, end_node, filter_func, weight_func, validate, max_length, debug
                    )
                else:
                    # No filter, find shortest path
                    path = self._find_path(start_node, end_node, weight_func, debug)
                    if validate:
                        validate_path(path, self.graph, weight_func, max_length, allow_cycles=False)

                result = create_path_result(path, weight_func)
                return result

        finally:
            duration = time.time() - start_time
            self._performance.record_metric("path_finding_time", duration)

    def _find_filtered_path(
        self,
        start_node: str,
        end_node: str,
        filter_func: Callable[[List[Edge]], bool],
        weight_func: Optional[WeightFunc],
        validate: bool,
        max_length: Optional[int],
        debug: bool,
    ) -> List[Edge]:
        """Find path with filter function applied."""
        # First try direct path
        try:
            path = self._find_path(start_node, end_node, weight_func, debug)
            if filter_func(path):
                if validate:
                    validate_path(path, self.graph, weight_func, max_length)
                return path
        except GraphOperationError:
            pass

        # Try alternative paths through different nodes
        nodes = sorted(self.graph.get_nodes())  # Sort for deterministic behavior
        for node in nodes:
            if node == start_node or node == end_node:
                continue
            try:
                path1 = self._find_path(start_node, node, weight_func, debug)
                path2 = self._find_path(node, end_node, weight_func, debug)
                path = path1 + path2
                if filter_func(path):
                    if validate:
                        validate_path(path, self.graph, weight_func, max_length)
                    return path
            except GraphOperationError:
                continue

        raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

    def _find_path(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        debug: bool = False,
    ) -> List[Edge]:
        """Find shortest path without filter."""
        if debug:
            logger.debug(f"Starting bidirectional search from {start_node} to {end_node}")

        if not self.graph.has_node(start_node):
            raise GraphOperationError(f"Start node {start_node} not found in graph")
        if not self.graph.has_node(end_node):
            raise GraphOperationError(f"End node {end_node} not found in graph")

        # Initialize searches
        forward_pq = [(0.0, start_node)]
        backward_pq = [(0.0, end_node)]

        self._forward_distances[start_node] = 0.0
        self._backward_distances[end_node] = 0.0

        forward_predecessors: Dict[str, Tuple[str, Edge]] = {}
        backward_predecessors: Dict[str, Tuple[str, Edge]] = {}

        best_dist = float("inf")
        meeting_node = None

        iteration = 0
        max_iterations = 100000

        while forward_pq and backward_pq and iteration < max_iterations:
            iteration += 1

            if debug and iteration % 1000 == 0:
                logger.debug(f"Iteration {iteration}")

            # Process both directions alternately
            if forward_pq[0][0] < backward_pq[0][0]:
                best_dist, meeting_node = self._process_search_step(
                    True,
                    forward_pq,
                    self._forward_distances,
                    self._backward_distances,
                    forward_predecessors,
                    best_dist,
                    meeting_node,
                    weight_func,
                )
            else:
                best_dist, meeting_node = self._process_search_step(
                    False,
                    backward_pq,
                    self._backward_distances,
                    self._forward_distances,
                    backward_predecessors,
                    best_dist,
                    meeting_node,
                    weight_func,
                )

            # Early termination check
            if forward_pq and backward_pq:
                if forward_pq[0][0] + backward_pq[0][0] >= best_dist:
                    break

        if meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct and validate path
        path = self._reconstruct_path(
            start_node, end_node, meeting_node, forward_predecessors, backward_predecessors
        )

        return unpack_shortcut(path, self.graph)

    def _process_search_step(
        self,
        is_forward: bool,
        pq: List[Tuple[float, str]],
        distances: Dict[str, float],
        other_distances: Dict[str, float],
        predecessors: Dict[str, Tuple[str, Edge]],
        best_dist: float,
        meeting_node: Optional[str],
        weight_func: Optional[WeightFunc],
    ) -> Tuple[float, Optional[str]]:
        """Process one step of bidirectional search."""
        if not pq:
            return best_dist, meeting_node

        dist, node = heappop(pq)

        # Skip if we've found a better path
        if dist > best_dist:
            return best_dist, meeting_node

        # Check for meeting point
        if node in other_distances:
            total_dist = dist + other_distances[node]
            if total_dist < best_dist:
                best_dist = total_dist
                meeting_node = node

        # Expand node
        for neighbor, edge in self._get_edges_to_process(node, is_forward):
            edge_weight = get_edge_weight(edge, weight_func)
            new_dist = dist + edge_weight

            if self._is_path_valid(neighbor, new_dist, distances):
                distances[neighbor] = new_dist
                predecessors[neighbor] = (node, edge)
                heappush(pq, (new_dist, neighbor))
                self._path_nodes.add(neighbor)

        return best_dist, meeting_node

    def _create_reverse_edge(self, edge: Edge) -> Edge:
        """Create a reverse edge with copied metadata."""
        return Edge(
            from_entity=edge.to_entity,
            to_entity=edge.from_entity,
            relation_type=edge.relation_type,
            metadata=deepcopy(edge.metadata),
            impact_score=edge.impact_score,
            attributes=edge.attributes.copy(),
            context=edge.context,
            validation_status=edge.validation_status,
            custom_metrics=edge.custom_metrics.copy(),
        )

    def _reconstruct_path(
        self,
        start_node: str,
        end_node: str,
        meeting_node: str,
        forward_predecessors: Dict[str, Tuple[str, Edge]],
        backward_predecessors: Dict[str, Tuple[str, Edge]],
    ) -> List[Edge]:
        """Reconstruct path from predecessor maps."""
        path = []

        # Build forward path
        current = meeting_node
        while current != start_node:
            prev, edge = forward_predecessors[current]
            path.insert(0, edge)
            current = prev

        # Build backward path
        current = meeting_node
        while current != end_node:
            next_node, edge = backward_predecessors[current]
            forward_edge = Edge(
                from_entity=current,
                to_entity=next_node,
                relation_type=edge.relation_type,
                metadata=deepcopy(edge.metadata),
                impact_score=edge.impact_score,
                attributes=edge.attributes.copy(),
                context=edge.context,
                validation_status=edge.validation_status,
                custom_metrics=edge.custom_metrics.copy(),
            )
            path.append(forward_edge)
            current = next_node

        return path

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        return self._performance.get_statistics()
