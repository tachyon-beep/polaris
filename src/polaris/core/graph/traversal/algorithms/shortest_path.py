"""Enhanced implementation of shortest path algorithms."""

import logging
import math
import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from array import array

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
        self._node_indices: Dict[str, int] = {}
        self._index_to_node: List[str] = []
        self._indices_lock = threading.Lock()
        self._initialize_node_indices()

    def _initialize_node_indices(self):
        """Initialize node index mappings for faster lookups."""
        with self._indices_lock:
            for i, node in enumerate(self.graph.get_nodes()):
                self._node_indices[node] = i
                self._index_to_node.append(node)

    def _get_node_index(self, node: str) -> int:
        """Thread-safe node index lookup with dynamic addition."""
        with self._indices_lock:
            if node not in self._node_indices:
                idx = len(self._node_indices)
                self._node_indices[node] = idx
                self._index_to_node.append(node)
            return self._node_indices[node]

    def _get_timeout(self, graph_size: int) -> float:
        """Calculate appropriate timeout based on graph size."""
        # More aggressive timeout scaling
        base_timeout = 15.0  # Reduced base timeout
        size_factor = (graph_size / 1000) ** 0.7  # More aggressive sublinear scaling
        return base_timeout + (size_factor * 5.0)

    def _estimate_distance(self, node_idx: int, target_idx: int) -> float:
        """Simple heuristic for A* optimization."""
        # Use node indices as a simple distance estimate
        # This provides a consistent but admissible heuristic
        return abs(node_idx - target_idx) * 0.1  # Scale factor to ensure admissibility

    def _get_queue_priority(self, queue: PriorityQueue) -> float:
        """Safely get priority from queue."""
        if queue.empty():
            return float("inf")
        peek_result = queue.peek()
        return peek_result[0] if peek_result is not None else float("inf")

    @contextmanager
    def _search_context(self):
        """Context manager for search operations."""
        try:
            yield
        finally:
            self.memory_manager.reset_peak_memory()

    def _safe_get_edge_weight(self, edge: Edge, weight_func: Optional[WeightFunc]) -> float:
        """Safely get edge weight, converting exceptions appropriately."""
        try:
            weight = get_edge_weight(edge, weight_func)
            if math.isinf(weight):
                raise ValueError("Path cost exceeded maximum value")
            return weight
        except ValueError as e:
            if "Edge weight must be finite number" in str(e):
                raise ValueError("Path cost exceeded maximum value")
            raise

    def _bidirectional_search(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        max_length: Optional[int],
        metrics: PerformanceMetrics,
        timeout: float,
    ) -> PathResult:
        """Optimized bidirectional search implementation."""
        start_time = time.time()
        nodes_explored = 0

        # Try direct path first
        if max_length is None or max_length >= 1:
            edge = self.graph.get_edge(start_node, end_node)
            if edge:
                try:
                    weight = self._safe_get_edge_weight(edge, weight_func)
                    return create_path_result([edge], weight_func)
                except ValueError:
                    # Continue with search if direct path has invalid weight
                    pass

        # Thread-safe index lookups
        start_idx = self._get_node_index(start_node)
        end_idx = self._get_node_index(end_node)

        # Use arrays for better memory efficiency
        array_size = max(len(self._node_indices), end_idx + 1)
        forward_distances = array("d", [float("inf")] * array_size)
        backward_distances = array("d", [float("inf")] * array_size)
        forward_lengths = array("H", [65535] * array_size)
        backward_lengths = array("H", [65535] * array_size)

        # Initialize distances
        forward_distances[start_idx] = 0.0
        backward_distances[end_idx] = 0.0
        forward_lengths[start_idx] = 0
        backward_lengths[end_idx] = 0

        # Priority queues with heuristic
        forward_queue = PriorityQueue()
        backward_queue = PriorityQueue()
        forward_queue.add_or_update(start_node, self._estimate_distance(start_idx, end_idx))
        backward_queue.add_or_update(end_node, self._estimate_distance(end_idx, start_idx))

        # Efficient queue membership tracking
        forward_in_queue = {start_node}
        backward_in_queue = {end_node}

        # Predecessor tracking with pre-allocated dictionaries
        forward_predecessors: Dict[str, Tuple[str, Edge]] = {}
        backward_predecessors: Dict[str, Tuple[str, Edge]] = {}

        best_meeting_node = None
        best_total_distance = float("inf")
        memory_check_counter = 0

        while not (forward_queue.empty() and backward_queue.empty()):
            # Periodic memory checks
            memory_check_counter += 1
            if memory_check_counter >= 1000:
                self.memory_manager.check_memory()
                memory_check_counter = 0

            # Timeout check
            if time.time() - start_time > timeout:
                raise GraphOperationError("Path finding timeout exceeded")

            # Choose direction based on queue priorities
            forward_priority = self._get_queue_priority(forward_queue)
            backward_priority = self._get_queue_priority(backward_queue)

            if not forward_queue.empty() and (
                backward_queue.empty() or forward_priority <= backward_priority
            ):
                # Forward search step
                result = forward_queue.pop()
                if result is None:
                    continue

                current_dist, current_node = result
                current_idx = self._get_node_index(current_node)
                forward_in_queue.remove(current_node)
                nodes_explored += 1

                # Early termination checks
                if current_dist >= best_total_distance:
                    continue

                current_length = forward_lengths[current_idx]
                if max_length is not None and current_length >= max_length:
                    continue

                # Update best path if possible
                backward_dist = backward_distances[current_idx]
                if backward_dist != float("inf"):
                    total_dist = current_dist + backward_dist
                    if total_dist < best_total_distance:
                        best_meeting_node = current_node
                        best_total_distance = total_dist

                # Process neighbors efficiently
                neighbors = list(self.graph.get_neighbors(current_node))
                for neighbor in neighbors:
                    edge = self.graph.get_edge(current_node, neighbor)
                    if not edge:
                        continue

                    try:
                        weight = self._safe_get_edge_weight(edge, weight_func)
                        neighbor_idx = self._get_node_index(neighbor)
                        new_distance = forward_distances[current_idx] + weight
                        new_length = current_length + 1

                        if (
                            max_length is not None and new_length > max_length
                        ) or new_distance >= best_total_distance:
                            continue

                        if new_distance < forward_distances[neighbor_idx]:
                            forward_distances[neighbor_idx] = new_distance
                            forward_lengths[neighbor_idx] = new_length
                            forward_predecessors[neighbor] = (current_node, edge)

                            # Add with heuristic estimate
                            if neighbor not in forward_in_queue:
                                estimate = new_distance + self._estimate_distance(
                                    neighbor_idx, end_idx
                                )
                                forward_queue.add_or_update(neighbor, estimate)
                                forward_in_queue.add(neighbor)

                    except ValueError as e:
                        if "Path cost exceeded maximum value" in str(e):
                            raise
                        continue

            elif not backward_queue.empty():
                # Backward search step
                result = backward_queue.pop()
                if result is None:
                    continue

                current_dist, current_node = result
                current_idx = self._get_node_index(current_node)
                backward_in_queue.remove(current_node)
                nodes_explored += 1

                if current_dist >= best_total_distance:
                    continue

                current_length = backward_lengths[current_idx]
                if max_length is not None and current_length >= max_length:
                    continue

                # Update best path if possible
                forward_dist = forward_distances[current_idx]
                if forward_dist != float("inf"):
                    total_dist = current_dist + forward_dist
                    if total_dist < best_total_distance:
                        best_meeting_node = current_node
                        best_total_distance = total_dist

                # Process neighbors efficiently
                neighbors = list(self.graph.get_neighbors(current_node, reverse=True))
                for neighbor in neighbors:
                    edge = self.graph.get_edge(neighbor, current_node)
                    if not edge:
                        continue

                    try:
                        weight = self._safe_get_edge_weight(edge, weight_func)
                        neighbor_idx = self._get_node_index(neighbor)
                        new_distance = backward_distances[current_idx] + weight
                        new_length = current_length + 1

                        if (
                            max_length is not None and new_length > max_length
                        ) or new_distance >= best_total_distance:
                            continue

                        if new_distance < backward_distances[neighbor_idx]:
                            backward_distances[neighbor_idx] = new_distance
                            backward_lengths[neighbor_idx] = new_length
                            backward_predecessors[neighbor] = (current_node, edge)

                            # Add with heuristic estimate
                            if neighbor not in backward_in_queue:
                                estimate = new_distance + self._estimate_distance(
                                    neighbor_idx, start_idx
                                )
                                backward_queue.add_or_update(neighbor, estimate)
                                backward_in_queue.add(neighbor)

                    except ValueError as e:
                        if "Path cost exceeded maximum value" in str(e):
                            raise
                        continue

            # Early termination check
            if (
                not forward_queue.empty()
                and not backward_queue.empty()
                and self._get_queue_priority(forward_queue)
                + self._get_queue_priority(backward_queue)
                >= best_total_distance
            ):
                break

        metrics.nodes_explored = nodes_explored

        if best_meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path efficiently
        path = []
        current = best_meeting_node

        # Forward path
        while current in forward_predecessors:
            prev, edge = forward_predecessors[current]
            path.insert(0, edge)
            current = prev

        # Backward path
        current = best_meeting_node
        while current in backward_predecessors:
            next_node, edge = backward_predecessors[current]
            path.append(edge)
            current = next_node

        if max_length is not None and len(path) > max_length:
            raise GraphOperationError(f"No path of length <= {max_length} exists")

        return create_path_result(path, weight_func)

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
        """Find shortest path using optimized bidirectional search."""
        metrics = PerformanceMetrics(operation="shortest_path", start_time=time.time())
        metrics.nodes_explored = 0

        with self._search_context():
            try:
                # Quick validation
                if start_node not in self._node_indices or end_node not in self._node_indices:
                    raise NodeNotFoundError(
                        f"{'Start' if start_node not in self._node_indices else 'End'} node not found"
                    )

                # Simple cache key
                cache_key = f"{start_node}:{end_node}:{max_length}"
                if cached := PathCache.get(cache_key):
                    metrics.cache_hit = True
                    try:
                        validate_path(cached.path, self.graph, weight_func, max_length)
                        return cached
                    except (PathValidationError, ValueError):
                        pass

                # Calculate timeout
                timeout = self._get_timeout(len(self._node_indices))

                # Use bidirectional search
                result = self._bidirectional_search(
                    start_node, end_node, weight_func, max_length, metrics, timeout
                )

                if filter_func and not filter_func(result.path):
                    return None

                PathCache.put(cache_key, result)
                return result

            except ValueError as e:
                if "Path cost exceeded maximum value" in str(e):
                    raise ValueError("Path cost exceeded maximum value")
                raise
            finally:
                metrics.end_time = time.time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def find_paths(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Iterator[PathResult]:
        """Find shortest path and return as iterator yielding single PathResult."""
        result = self.find_path(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            weight_func=weight_func,
            **kwargs,
        )
        if result is not None:
            yield result
