"""
Implementation of bidirectional search for finding paths.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from time import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from ....core.exceptions import GraphOperationError
from ....core.models import Edge
from ..base import PathFinder
from ..cache import PathCache
from ..models import PathResult, PerformanceMetrics
from ..utils import WeightFunc, create_path_result, get_edge_weight


@dataclass
class SearchState:
    """State for each direction of bidirectional search."""

    queue: deque  # (node, depth, total_weight)
    visited: Set[str]
    parent: Dict[str, Optional[str]]
    edges: Dict[Tuple[str, str], Edge]
    weights: Dict[str, float]
    is_forward: bool


class BidirectionalPathFinder(PathFinder):
    """Bidirectional search implementation."""

    def _get_neighbors(self, node: str, is_forward: bool) -> Set[str]:
        """Get neighbors based on search direction."""
        return self.graph.get_neighbors(node, reverse=not is_forward)

    def _get_edge(self, from_node: str, to_node: str, is_forward: bool) -> Optional[Edge]:
        """Get edge based on search direction."""
        if is_forward:
            return self.graph.get_edge(from_node, to_node)
        return self.graph.get_edge(to_node, from_node)

    def _get_node_depth(self, node: str, state: SearchState) -> int:
        """Get the depth at which a node was discovered."""
        for n, d, _ in state.queue:
            if n == node:
                return d
        # If not in queue but visited, find its parent's depth + 1
        if node in state.visited:
            current = node
            depth = 0
            while current in state.parent and state.parent[current] is not None:
                current = state.parent[current]
                depth += 1
            return depth
        return 0

    def _expand_search(
        self,
        current: SearchState,
        opposite: SearchState,
        weight_func: Optional[WeightFunc] = None,
        max_depth: Optional[int] = None,
    ) -> Optional[str]:
        """Expand search in one direction."""
        if not current.queue:
            return None

        node, depth, total_weight = current.queue.popleft()

        # Don't explore beyond max_depth
        if max_depth is not None and depth >= max_depth:
            return None

        # Check if this node intersects with opposite frontier
        if node in opposite.visited:
            opp_depth = self._get_node_depth(node, opposite)
            total_depth = depth + opp_depth

            # For max_depth=1, require exactly one step from each direction
            if max_depth == 1:
                if total_depth != 2:  # One step from each direction
                    return None
            # For other depths, ensure total path length doesn't exceed max_depth
            elif max_depth is not None and total_depth > max_depth:
                return None

            return node

        # Explore neighbors
        neighbors = self._get_neighbors(node, current.is_forward)
        for neighbor in neighbors:
            if neighbor not in current.visited:
                edge = self._get_edge(node, neighbor, current.is_forward)
                if edge:
                    # Skip if adding this edge would exceed max_depth
                    if max_depth is not None and depth + 1 > max_depth:
                        continue

                    edge_weight = get_edge_weight(edge, weight_func)
                    new_weight = total_weight + edge_weight
                    current.queue.append((neighbor, depth + 1, new_weight))
                    current.visited.add(neighbor)
                    current.parent[neighbor] = node
                    current.edges[(node, neighbor)] = edge
                    current.weights[neighbor] = new_weight

        return None

    def _construct_path(
        self, intersection: str, forward: SearchState, backward: SearchState
    ) -> List[Edge]:
        """Construct the complete path from the intersection point."""
        path = []

        # Build forward path
        current = intersection
        while current in forward.parent:
            prev = forward.parent[current]
            if prev is not None:
                edge = forward.edges.get((prev, current))
                if edge:
                    path.append(edge)
            current = prev
        path.reverse()

        # Build backward path
        current = intersection
        while current in backward.parent:
            next_node = backward.parent[current]
            if next_node is not None:
                edge = self.graph.get_edge(current, next_node)
                if edge:
                    path.append(edge)
            current = next_node

        return path

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> PathResult:
        """Find path using bidirectional search."""
        metrics = PerformanceMetrics(operation="bidirectional_search", start_time=time())
        metrics.nodes_explored = 0

        try:
            # Check cache
            cache_key = PathCache.get_cache_key(
                start_node,
                end_node,
                "bidirectional",
                weight_func.__name__ if weight_func else None,
                max_length,
            )
            cached_result = PathCache.get(cache_key)
            if cached_result is not None:
                metrics.cache_hit = True
                return cached_result

            self.validate_nodes(start_node, end_node)

            # For max_length=1, we need a direct edge
            if max_length == 1:
                edge = self.graph.get_edge(start_node, end_node)
                if edge is None:
                    raise GraphOperationError(
                        f"No path exists between {start_node} and {end_node} within length {max_length}"
                    )
                return create_path_result([edge], weight_func, self.graph)

            # Initialize search states
            forward = SearchState(
                queue=deque([(start_node, 0, 0.0)]),
                visited={start_node},
                parent={start_node: None},
                edges={},
                weights={start_node: 0.0},
                is_forward=True,
            )

            backward = SearchState(
                queue=deque([(end_node, 0, 0.0)]),
                visited={end_node},
                parent={end_node: None},
                edges={},
                weights={end_node: 0.0},
                is_forward=False,
            )

            best_path: Optional[List[Edge]] = None
            best_weight = float("infinity")

            # Use max_length as max_depth for bidirectional search
            max_depth = max_length

            while forward.queue and backward.queue:
                metrics.nodes_explored += 1

                # Expand forward search
                intersection = self._expand_search(forward, backward, weight_func, max_depth)
                if intersection:
                    path = self._construct_path(intersection, forward, backward)
                    if max_length is not None and len(path) > max_length:
                        continue
                    total_weight = forward.weights[intersection] + backward.weights[intersection]
                    if total_weight < best_weight:
                        best_path = path
                        best_weight = total_weight

                # Expand backward search
                intersection = self._expand_search(backward, forward, weight_func, max_depth)
                if intersection:
                    path = self._construct_path(intersection, forward, backward)
                    if max_length is not None and len(path) > max_length:
                        continue
                    total_weight = forward.weights[intersection] + backward.weights[intersection]
                    if total_weight < best_weight:
                        best_path = path
                        best_weight = total_weight

                # Check if both queues are empty
                if not forward.queue or not backward.queue:
                    break

            if best_path is None:
                raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

            result = create_path_result(best_path, weight_func, self.graph)

            # Verify max_length constraint
            if max_length is not None and len(result) > max_length:
                raise GraphOperationError(
                    f"No path exists between {start_node} and {end_node} within length {max_length}"
                )

            PathCache.put(cache_key, result)
            metrics.path_length = len(best_path)
            return result

        finally:
            metrics.end_time = time()
