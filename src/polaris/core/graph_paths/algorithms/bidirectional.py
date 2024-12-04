"""
Bidirectional search implementation.

This module provides an optimized bidirectional search implementation that
explores from both start and end nodes simultaneously.
"""

from time import time
from typing import Dict, List, Optional, Any, cast, Tuple
from contextlib import contextmanager

from ....core.exceptions import GraphOperationError
from ....core.models import Edge
from ..base import PathFinder, PathFilter, WeightFunc
from ..models import PathResult, PerformanceMetrics, PathValidationError
from ..utils import (
    PriorityQueue,
    MemoryManager,
    PathState,
    create_path_result,
    get_edge_weight,
    validate_path,
    is_better_cost,
)

import logging

logger = logging.getLogger(__name__)

# Constants
MAX_QUEUE_SIZE = 100000  # Maximum size for priority queues


class BidirectionalPathFinder(PathFinder[PathResult]):
    """
    Bidirectional search implementation.

    Features:
    - Simultaneous forward and backward search
    - Memory-efficient path state tracking
    - Early termination optimization
    - Path validation and metrics
    """

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

    def _expand_backward(
        self,
        node: str,
        visited: Dict[str, PathState],
        queue: PriorityQueue,
        weight_func: Optional[WeightFunc],
        best_cost: float,
    ) -> None:
        """Expand node in backward direction using incoming edges."""
        current_state = visited[node]

        # Get incoming edges using reverse neighbors
        for incoming_node in self.graph.get_neighbors(node, reverse=True):
            edge = self.graph.get_edge(incoming_node, node)
            if not edge:
                continue

            try:
                edge_weight = get_edge_weight(edge, weight_func)
                new_cost = current_state.total_weight + edge_weight

                # Only expand if new path is potentially better
                if is_better_cost(new_cost, best_cost) and (
                    incoming_node not in visited
                    or is_better_cost(new_cost, visited[incoming_node].total_weight)
                ):
                    visited[incoming_node] = PathState(
                        incoming_node, edge, current_state, current_state.depth + 1, new_cost
                    )
                    queue.add_or_update(incoming_node, new_cost)

            except (ValueError, OverflowError) as e:
                # Log but continue with other neighbors
                continue

    def _expand_forward(
        self,
        node: str,
        visited: Dict[str, PathState],
        queue: PriorityQueue,
        weight_func: Optional[WeightFunc],
        best_cost: float,
    ) -> None:
        """Expand node in forward direction using outgoing edges."""
        current_state = visited[node]

        # Get outgoing edges
        for neighbor in self.graph.get_neighbors(node):
            edge = self.graph.get_edge(node, neighbor)
            if not edge:
                continue

            try:
                edge_weight = get_edge_weight(edge, weight_func)
                new_cost = current_state.total_weight + edge_weight

                # Only expand if new path is potentially better
                if is_better_cost(new_cost, best_cost) and (
                    neighbor not in visited
                    or is_better_cost(new_cost, visited[neighbor].total_weight)
                ):
                    visited[neighbor] = PathState(
                        neighbor, edge, current_state, current_state.depth + 1, new_cost
                    )
                    queue.add_or_update(neighbor, new_cost)

            except (ValueError, OverflowError) as e:
                # Log but continue with other neighbors
                continue

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
        """Find shortest path using bidirectional search."""
        metrics = PerformanceMetrics(operation="bidirectional", start_time=time())
        metrics.nodes_explored = 0

        with self._search_context():
            try:
                self.validate_nodes(start_node, end_node)

                # Early exit for same node
                if start_node == end_node:
                    return create_path_result([], weight_func)

                # Handle max_length division safely
                forward_depth_limit = (max_length // 2) if max_length is not None else None
                backward_depth_limit = (
                    (max_length - cast(int, forward_depth_limit))
                    if max_length is not None
                    else None
                )

                forward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
                backward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
                forward_queue.add_or_update(start_node, 0.0)
                backward_queue.add_or_update(end_node, 0.0)

                forward_visited = {start_node: PathState(start_node, None, None, 0, 0.0)}
                backward_visited = {end_node: PathState(end_node, None, None, 0, 0.0)}

                best_path = None
                best_cost = float("inf")
                meeting_node = None

                while not (forward_queue.empty() or backward_queue.empty()):
                    self.memory_manager.check_memory()
                    metrics.nodes_explored += 1

                    # Get current minimum costs
                    forward_min = forward_queue.peek_priority()
                    backward_min = backward_queue.peek_priority()

                    # Strong termination condition
                    if forward_min + backward_min >= best_cost:
                        break

                    # Expand the direction with lower current cost
                    if forward_min <= backward_min:
                        result = forward_queue.pop()
                        if result is None:
                            continue
                        current_cost, current = result

                        # Skip if we've found a better path
                        if current_cost >= best_cost:
                            continue

                        # Check for intersection
                        if current in backward_visited:
                            total_cost = current_cost + backward_visited[current].total_weight
                            if is_better_cost(total_cost, best_cost):
                                forward_len = forward_visited[current].depth
                                backward_len = backward_visited[current].depth
                                if max_length is None or forward_len + backward_len <= max_length:
                                    best_cost = total_cost
                                    meeting_node = current

                        # Expand forward within depth limit
                        current_depth = forward_visited[current].depth
                        if forward_depth_limit is None or current_depth < forward_depth_limit:
                            self._expand_forward(
                                current,
                                forward_visited,
                                forward_queue,
                                weight_func,
                                best_cost,
                            )
                    else:
                        result = backward_queue.pop()
                        if result is None:
                            continue
                        current_cost, current = result

                        # Skip if we've found a better path
                        if current_cost >= best_cost:
                            continue

                        # Check for intersection
                        if current in forward_visited:
                            total_cost = current_cost + forward_visited[current].total_weight
                            if is_better_cost(total_cost, best_cost):
                                forward_len = forward_visited[current].depth
                                backward_len = backward_visited[current].depth
                                if max_length is None or forward_len + backward_len <= max_length:
                                    best_cost = total_cost
                                    meeting_node = current

                        # Expand backward within depth limit
                        current_depth = backward_visited[current].depth
                        if backward_depth_limit is None or current_depth < backward_depth_limit:
                            self._expand_backward(
                                current,
                                backward_visited,
                                backward_queue,
                                weight_func,
                                best_cost,
                            )

                if meeting_node is None:
                    raise GraphOperationError(
                        f"No path exists between '{start_node}' and '{end_node}'"
                    )

                # Reconstruct path
                forward_path = self._reconstruct_forward_path(meeting_node, forward_visited)
                backward_path = self._reconstruct_backward_path(meeting_node, backward_visited)
                best_path = forward_path + list(reversed(backward_path))

                # Validate final path
                if max_length is not None and len(best_path) > max_length:
                    raise GraphOperationError(f"No path of length <= {max_length} exists")

                result = create_path_result(best_path, weight_func)
                validate_path(best_path, self.graph, weight_func)
                return result

            finally:
                metrics.end_time = time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _reconstruct_forward_path(
        self, meeting_node: str, visited: Dict[str, PathState]
    ) -> List[Edge]:
        """Reconstruct the forward path."""
        path: List[Edge] = []
        state = visited.get(meeting_node)

        while state is not None and state.prev_state is not None:
            if state.prev_edge is not None:
                path.append(state.prev_edge)
            state = state.prev_state

        path.reverse()
        return path

    def _reconstruct_backward_path(
        self, meeting_node: str, visited: Dict[str, PathState]
    ) -> List[Edge]:
        """Reconstruct the backward path."""
        path: List[Edge] = []
        state = visited.get(meeting_node)

        while state is not None and state.prev_state is not None:
            if state.prev_edge is not None:
                path.append(state.prev_edge)
            state = state.prev_state

        return path
