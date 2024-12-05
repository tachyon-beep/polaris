"""Bidirectional search implementation."""

from typing import Dict, List, Optional, Set, Tuple, Any
from time import time

from ...exceptions import GraphOperationError
from ...models import Edge
from ..base import PathFinder
from ..models import PathResult, PerformanceMetrics
from ..types import WeightFunc, PathFilter
from ..utils import (
    PriorityQueue,
    MemoryManager,
    PathState,
    create_path_result,
    get_edge_weight,
    validate_path,
    is_better_cost,
    MAX_QUEUE_SIZE,
)


class BidirectionalFinder(PathFinder[PathResult]):
    """
    Bidirectional search implementation.

    This implementation supports both positive and negative weights through the
    allow_negative_weights decorator. When using negative weights (typically for
    maximization problems), the algorithm will find the path with the minimum
    total weight, which corresponds to the maximum value when weights are negated.

    Example:
        @allow_negative_weights
        def weight_func(edge: Edge) -> float:
            return -edge.impact_score  # Negate to find path with maximum impact

        # Find path with maximum total impact score
        path = PathFinding.bidirectional_search(graph, "A", "B", weight_func=weight_func)
    """

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
        """Find path using bidirectional search."""
        # Validate nodes exist
        self.validate_nodes(start_node, end_node)

        metrics = PerformanceMetrics(operation="bidirectional", start_time=time())
        metrics.nodes_explored = 0

        try:
            # Get max_depth from kwargs or max_length
            max_depth = kwargs.get("max_depth", max_length)
            if max_depth is not None:
                if not isinstance(max_depth, int):
                    raise TypeError("max_depth must be an integer")
                if max_depth <= 0:
                    raise ValueError("max_depth must be positive")
                if max_depth == 1:
                    # Special case: max_depth=1 means only direct edges allowed
                    edge = self.graph.get_edge(start_node, end_node)
                    if not edge:
                        raise GraphOperationError(
                            f"No path of length <= 1 exists between {start_node} and {end_node}"
                        )
                    return create_path_result([edge], weight_func)

            # Forward search state
            forward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
            forward_queue.add_or_update(start_node, 0.0)
            forward_distances: Dict[str, float] = {start_node: 0.0}
            forward_states: Dict[str, PathState] = {
                start_node: PathState(start_node, None, None, 0, 0.0)
            }
            forward_explored: Set[str] = set()

            # Backward search state
            backward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
            backward_queue.add_or_update(end_node, 0.0)
            backward_distances: Dict[str, float] = {end_node: 0.0}
            backward_states: Dict[str, PathState] = {
                end_node: PathState(end_node, None, None, 0, 0.0)
            }
            backward_explored: Set[str] = set()

            # Track best meeting point
            best_total_dist = float("inf")
            best_meeting_node = None
            best_forward_state = None
            best_backward_state = None

            while not (forward_queue.empty() and backward_queue.empty()):
                self.memory_manager.check_memory()
                metrics.nodes_explored += 1

                # Forward search step
                if not forward_queue.empty():
                    current = forward_queue.pop()
                    if not current:
                        continue
                    current_dist, current_node = current
                    if current_node in forward_explored:
                        continue
                    forward_explored.add(current_node)

                    current_state = forward_states[current_node]

                    # Check if we've found a better path
                    if current_node in backward_distances:
                        total_dist = current_dist + backward_distances[current_node]
                        if is_better_cost(total_dist, best_total_dist):
                            backward_state = backward_states[current_node]
                            total_depth = current_state.depth + backward_state.depth
                            if max_depth is None or total_depth <= max_depth:
                                # Check for cycles in combined path
                                forward_path = current_state.get_path()
                                backward_path = backward_state.get_path()
                                forward_nodes = (
                                    {edge.from_entity for edge in forward_path}
                                    | {forward_path[-1].to_entity}
                                    if forward_path
                                    else {start_node}
                                )
                                backward_nodes = (
                                    {edge.from_entity for edge in backward_path}
                                    | {backward_path[-1].to_entity}
                                    if backward_path
                                    else {end_node}
                                )
                                # Only check for overlap at meeting point
                                if len(forward_nodes & backward_nodes) <= 1:
                                    best_total_dist = total_dist
                                    best_meeting_node = current_node
                                    best_forward_state = current_state
                                    best_backward_state = backward_state

                    # Expand forward if within depth limit
                    if max_depth is None or current_state.depth < max_depth:
                        for neighbor in self.graph.get_neighbors(current_node):
                            # Skip if neighbor would create a cycle in forward path
                            if neighbor in current_state.get_visited():
                                continue

                            edge = self.graph.get_edge(current_node, neighbor)
                            if not edge:
                                continue

                            try:
                                edge_weight = get_edge_weight(edge, weight_func)
                                new_dist = current_dist + edge_weight
                                new_depth = current_state.depth + 1

                                if neighbor not in forward_distances or is_better_cost(
                                    new_dist, forward_distances[neighbor]
                                ):
                                    forward_distances[neighbor] = new_dist
                                    forward_states[neighbor] = PathState(
                                        neighbor, edge, current_state, new_depth, new_dist
                                    )
                                    forward_queue.add_or_update(neighbor, new_dist)
                            except ValueError as e:
                                if "Edge weight must be finite number" in str(e):
                                    raise ValueError("Path cost exceeded maximum value")
                                continue

                # Backward search step
                if not backward_queue.empty():
                    current = backward_queue.pop()
                    if not current:
                        continue
                    current_dist, current_node = current
                    if current_node in backward_explored:
                        continue
                    backward_explored.add(current_node)

                    current_state = backward_states[current_node]

                    # Check if we've found a better path
                    if current_node in forward_distances:
                        total_dist = current_dist + forward_distances[current_node]
                        if is_better_cost(total_dist, best_total_dist):
                            forward_state = forward_states[current_node]
                            total_depth = forward_state.depth + current_state.depth
                            if max_depth is None or total_depth <= max_depth:
                                # Check for cycles in combined path
                                forward_path = forward_state.get_path()
                                backward_path = current_state.get_path()
                                forward_nodes = (
                                    {edge.from_entity for edge in forward_path}
                                    | {forward_path[-1].to_entity}
                                    if forward_path
                                    else {start_node}
                                )
                                backward_nodes = (
                                    {edge.from_entity for edge in backward_path}
                                    | {backward_path[-1].to_entity}
                                    if backward_path
                                    else {end_node}
                                )
                                # Only check for overlap at meeting point
                                if len(forward_nodes & backward_nodes) <= 1:
                                    best_total_dist = total_dist
                                    best_meeting_node = current_node
                                    best_forward_state = forward_state
                                    best_backward_state = current_state

                    # Expand backward if within depth limit
                    if max_depth is None or current_state.depth < max_depth:
                        for neighbor in self.graph.get_neighbors(current_node, reverse=True):
                            # Skip if neighbor would create a cycle in backward path
                            if neighbor in current_state.get_visited():
                                continue

                            edge = self.graph.get_edge(
                                neighbor, current_node
                            )  # Note: reversed edge
                            if not edge:
                                continue

                            try:
                                edge_weight = get_edge_weight(edge, weight_func)
                                new_dist = current_dist + edge_weight
                                new_depth = current_state.depth + 1

                                if neighbor not in backward_distances or is_better_cost(
                                    new_dist, backward_distances[neighbor]
                                ):
                                    backward_distances[neighbor] = new_dist
                                    backward_states[neighbor] = PathState(
                                        neighbor, edge, current_state, new_depth, new_dist
                                    )
                                    backward_queue.add_or_update(neighbor, new_dist)
                            except ValueError as e:
                                if "Edge weight must be finite number" in str(e):
                                    raise ValueError("Path cost exceeded maximum value")
                                continue

            metrics.end_time = time()

            if (
                best_meeting_node is None
                or best_forward_state is None
                or best_backward_state is None
            ):
                raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

            # Reconstruct path
            forward_path = best_forward_state.get_path()
            backward_path = best_backward_state.get_path()
            backward_path.reverse()

            # Combine paths, removing duplicate meeting point edge if present
            if forward_path and backward_path and forward_path[-1] == backward_path[0]:
                backward_path = backward_path[1:]
            path = forward_path + backward_path

            # Check if path exceeds max_depth
            if max_depth is not None and len(path) > max_depth:
                raise GraphOperationError(
                    f"No path of length <= {max_depth} exists between {start_node} and {end_node}"
                )

            # Apply filter if provided
            if filter_func and not filter_func(path):
                return None

            return create_path_result(path, weight_func)

        finally:
            # Clean up resources
            self.memory_manager.reset_peak_memory()
