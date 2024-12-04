"""
Enhanced bidirectional search implementation.

This module provides an optimized bidirectional search implementation with:
- Efficient intersection detection
- Balanced expansion strategy
- Proper termination conditions
- Memory management
- Path state tracking
"""

from time import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

from ....core.exceptions import GraphOperationError, NodeNotFoundError
from ....core.models import Edge
from ..base import PathFinder, PathFilter
from ..models import PathResult, PerformanceMetrics
from ..utils import (
    WeightFunc,
    PriorityQueue,
    PathState,
    MemoryManager,
    create_path_result,
    get_edge_weight,
    validate_path,
    timer,
)


@dataclass(frozen=True)
class SearchFrontier:
    """
    Immutable search frontier state.

    Maintains priority queue and distance information for one direction
    of bidirectional search.
    """

    distances: Dict[str, float]
    predecessors: Dict[str, Tuple[str, Edge]]
    queue: PriorityQueue
    visited: Set[str]
    is_forward: bool

    @classmethod
    def create(cls, start_node: str, is_forward: bool) -> "SearchFrontier":
        """Create new frontier starting from given node."""
        queue = PriorityQueue()
        queue.add_or_update(start_node, 0.0)

        return cls(
            distances={start_node: 0.0},
            predecessors={},
            queue=queue,
            visited=set(),
            is_forward=is_forward,
        )


class BidirectionalPathFinder(PathFinder):
    """
    Enhanced bidirectional search implementation.

    Features:
    - Efficient priority queue-based expansion
    - Balanced expansion strategy
    - Proper termination conditions
    - Memory management
    - Path state tracking
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
    ) -> PathResult:
        """
        Find path using enhanced bidirectional search.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            max_paths: Not used in bidirectional search
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                mu: Balance parameter for expansion (default: 1.0)
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing the found path

        Raises:
            NodeNotFoundError: If nodes don't exist
            GraphOperationError: If no path exists
            MemoryError: If memory limit exceeded
        """
        metrics = PerformanceMetrics(operation="bidirectional_search", start_time=time())

        try:
            # Validate inputs
            self.validate_nodes(start_node, end_node)
            mu = kwargs.get("mu", 1.0)  # Balance parameter
            validate = kwargs.get("validate", True)

            # Initialize frontiers
            forward = SearchFrontier.create(start_node, True)
            backward = SearchFrontier.create(end_node, False)

            # Track best path
            best_path = None
            best_weight = float("inf")
            meeting_node = None

            while not forward.queue.empty() and not backward.queue.empty():
                self.memory_manager.check_memory()
                metrics.nodes_explored = len(forward.visited) + len(backward.visited)

                # Get minimum distances at frontiers
                forward_min = forward.queue.pop()
                backward_min = backward.queue.pop()

                # Break if either queue is exhausted
                if forward_min is None or backward_min is None:
                    break

                forward_dist, forward_node = forward_min
                backward_dist, backward_node = backward_min

                # Termination condition
                if forward_dist + backward_dist >= best_weight:
                    break

                # Re-add nodes to queues since we'll expand them
                forward.queue.add_or_update(forward_node, forward_dist)
                backward.queue.add_or_update(backward_node, backward_dist)

                # Expand forward or backward based on balance
                if forward_dist <= mu * backward_dist:
                    meeting_node = self._expand(forward, backward, weight_func, max_length)
                else:
                    meeting_node = self._expand(backward, forward, weight_func, max_length)

                # Update best path if intersection found
                if meeting_node:
                    path_weight = forward.distances.get(
                        meeting_node, float("inf")
                    ) + backward.distances.get(meeting_node, float("inf"))

                    if path_weight < best_weight:
                        best_weight = path_weight
                        best_path = self._reconstruct_path(meeting_node, forward, backward)

            if not best_path:
                raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

            # Apply filter if provided
            if filter_func and not filter_func(best_path):
                raise GraphOperationError(
                    f"No path satisfying filter exists between {start_node} and {end_node}"
                )

            # Create and validate result
            result = create_path_result(best_path, weight_func)
            if validate:
                validate_path(best_path, self.graph, weight_func, max_length)

            return result

        finally:
            metrics.end_time = time()
            metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _expand(
        self,
        current: SearchFrontier,
        opposite: SearchFrontier,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
    ) -> Optional[str]:
        """
        Expand one frontier and check for intersections.

        Args:
            current: Frontier to expand
            opposite: Opposite frontier
            weight_func: Optional weight function
            max_length: Maximum path length

        Returns:
            Meeting node if frontiers intersect, None otherwise
        """
        if current.queue.empty():
            return None

        min_item = current.queue.pop()
        if min_item is None:
            return None

        current_dist, current_node = min_item

        # Skip if we've found a better path
        if current_node in current.visited:
            return None

        current.visited.add(current_node)

        # Check for intersection
        if current_node in opposite.visited:
            return current_node

        # Get neighbors based on direction
        neighbors = self.graph.get_neighbors(current_node, reverse=not current.is_forward)

        # Process neighbors
        for neighbor in neighbors:
            if neighbor in current.visited:
                continue

            edge = self.graph.get_edge(
                current_node if current.is_forward else neighbor,
                neighbor if current.is_forward else current_node,
            )
            if not edge:
                continue

            edge_weight = get_edge_weight(edge, weight_func)
            new_distance = current_dist + edge_weight

            if neighbor not in current.distances or new_distance < current.distances[neighbor]:
                current.distances[neighbor] = new_distance
                current.predecessors[neighbor] = (current_node, edge)
                current.queue.add_or_update(neighbor, new_distance)

        return None

    def _reconstruct_path(
        self, meeting_node: str, forward: SearchFrontier, backward: SearchFrontier
    ) -> List[Edge]:
        """
        Reconstruct path from meeting point.

        Args:
            meeting_node: Node where frontiers intersect
            forward: Forward search frontier
            backward: Backward search frontier

        Returns:
            List of edges forming complete path
        """
        path = []

        # Build forward path
        current = meeting_node
        while current in forward.predecessors:
            prev, edge = forward.predecessors[current]
            path.append(edge)
            current = prev
        path.reverse()

        # Build backward path
        current = meeting_node
        while current in backward.predecessors:
            next_node, edge = backward.predecessors[current]
            path.append(edge)
            current = next_node

        return path
