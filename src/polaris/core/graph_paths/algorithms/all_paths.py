"""
Enhanced implementation of all-paths algorithm with memory efficiency.

This module provides an optimized implementation for finding all paths between
nodes with features including:
- Memory-efficient path state tracking
- Early pruning strategies
- Path optimization
- Memory management
- Performance monitoring
"""

from time import time
from typing import Dict, Iterator, List, Optional, Set, Tuple, Any, cast
from contextlib import contextmanager

from ....core.exceptions import GraphOperationError, NodeNotFoundError
from ....core.models import Edge
from ..base import PathFinder, PathFilter, WeightFunc
from ..models import PathResult, PerformanceMetrics
from ..utils import (
    PathState,
    MemoryManager,
    create_path_result,
    get_edge_weight,
    validate_path,
    timer,
    is_better_cost,
)

# Constants
DEFAULT_MAX_PATH_LENGTH = 50
DEFAULT_MAX_PATHS = 1000
PRUNE_WEIGHT_FACTOR = 2.0  # Prune paths with weight > min_weight * factor


class AllPathsFinder(PathFinder[Iterator[PathResult]]):
    """
    Enhanced implementation for finding all paths with memory efficiency.

    Features:
    - Memory-efficient path state tracking
    - Early pruning strategies
    - Path optimization
    - Memory monitoring
    - Performance metrics
    """

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.min_path_weight = float("inf")
        self.paths_found = 0

    @contextmanager
    def _path_search_context(self):
        """Context manager for path search state."""
        try:
            yield
        finally:
            # Clean up state
            self.min_path_weight = float("inf")
            self.paths_found = 0

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Iterator[PathResult]:
        """
        Find all paths using memory-efficient implementation.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length (default: DEFAULT_MAX_PATH_LENGTH)
            max_paths: Maximum number of paths (default: DEFAULT_MAX_PATHS)
            filter_func: Optional function to filter paths
            weight_func: Optional function for edge weights
            **kwargs: Additional options:
                early_pruning: Enable aggressive pruning (default: True)
                validate: Validate paths (default: True)

        Yields:
            PathResult objects for each valid path found

        Raises:
            NodeNotFoundError: If nodes don't exist
            MemoryError: If memory limit exceeded
            GraphOperationError: If path finding fails
        """
        metrics = PerformanceMetrics(operation="all_paths", start_time=time())
        metrics.nodes_explored = 0  # Initialize counter

        with self._path_search_context():
            try:
                # Validate inputs
                self.validate_nodes(start_node, end_node)
                max_length = max_length or DEFAULT_MAX_PATH_LENGTH
                max_paths = max_paths or DEFAULT_MAX_PATHS
                early_pruning = kwargs.get("early_pruning", True)
                validate = kwargs.get("validate", True)

                if max_length <= 0:
                    raise ValueError(f"max_length must be positive, got {max_length}")
                if max_paths <= 0:
                    raise ValueError(f"max_paths must be positive, got {max_paths}")

                # Initialize search
                initial_state = PathState(start_node, None, None, 0, 0.0)
                stack = [(initial_state, {start_node}, 0.0)]  # state, path_visited, weight

                while stack and self.paths_found < max_paths:
                    self.memory_manager.check_memory()
                    metrics.nodes_explored += 1

                    current_state, path_visited, current_weight = stack.pop()

                    # Early pruning
                    if (
                        early_pruning
                        and current_weight > self.min_path_weight * PRUNE_WEIGHT_FACTOR
                    ):
                        continue

                    if current_state.node == end_node:
                        path = current_state.get_path()

                        # Apply filter if provided
                        if filter_func and not filter_func(path):
                            continue

                        # Update minimum path weight
                        path_weight = current_weight
                        if is_better_cost(path_weight, self.min_path_weight):
                            self.min_path_weight = path_weight

                        # Create and validate result
                        result = create_path_result(path, weight_func)
                        if validate:
                            validate_path(path, self.graph, weight_func, max_length)

                        self.paths_found += 1
                        yield result
                        continue

                    # Skip if path is too long
                    if current_state.depth >= max_length:
                        continue

                    # Process neighbors
                    neighbors = sorted(
                        self.graph.get_neighbors(current_state.node),
                        key=lambda n: self._estimate_cost(n, end_node),
                        reverse=True,
                    )

                    for neighbor in neighbors:
                        # Skip visited nodes in current path
                        if neighbor in path_visited:
                            continue

                        edge = self.graph.get_edge(current_state.node, neighbor)
                        if not edge:
                            continue

                        try:
                            edge_weight = get_edge_weight(edge, weight_func)
                            new_weight = current_weight + edge_weight

                            # Early pruning check
                            if (
                                early_pruning
                                and new_weight > self.min_path_weight * PRUNE_WEIGHT_FACTOR
                            ):
                                continue

                            # Create new state
                            new_state = PathState(
                                neighbor,
                                edge,
                                current_state,
                                current_state.depth + 1,
                                new_weight,
                            )

                            # Create new visited set for this path
                            new_path_visited = path_visited | {neighbor}
                            stack.append((new_state, new_path_visited, new_weight))

                        except (ValueError, OverflowError) as e:
                            # Log but continue with other neighbors
                            continue

            except Exception as e:
                # Ensure metrics are captured even on error
                metrics.end_time = time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)
                raise

            finally:
                # Always record metrics
                metrics.end_time = time()
                metrics.max_memory_used = int(self.memory_manager.peak_memory_mb * 1024 * 1024)

    def _estimate_cost(self, node: str, target: str) -> float:
        """
        Estimate cost to target for neighbor ordering.

        This is a simple heuristic that can be improved with
        domain-specific knowledge.
        """
        try:
            # Using degree and common neighbors as heuristics
            degree_cost = -len(list(self.graph.get_neighbors(node)))
            common_neighbors = len(
                set(self.graph.get_neighbors(node)) & set(self.graph.get_neighbors(target))
            )
            return degree_cost - (common_neighbors * 2)  # Prioritize nodes with common neighbors
        except Exception:
            return float("inf")  # Return high cost on error
