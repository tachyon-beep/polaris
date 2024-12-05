"""Implementation of algorithm to find all possible paths between nodes."""

from typing import Optional, Iterator, Any, List, Set
from time import time

from polaris.core.models import Edge
from polaris.core.exceptions import GraphOperationError
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult, PerformanceMetrics
from polaris.core.graph_paths.types import WeightFunc, PathFilter
from polaris.core.graph_paths.utils import create_path_result, get_edge_weight


class AllPathsFinder(PathFinder[PathResult]):
    """Finds all possible paths between two nodes."""

    def __init__(self, graph: Any):
        """Initialize finder with graph."""
        super().__init__(graph)

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
        """Find all paths between nodes.

        Uses depth-first search to find all possible paths between start_node and end_node.
        Paths are yielded as they are found.

        Args:
            start_node: Starting node
            end_node: Target node
            max_length: Maximum path length (optional)
            max_paths: Maximum number of paths to find (optional)
            filter_func: Function to filter paths (optional)
            weight_func: Function to compute edge weights (optional)
            **kwargs: Additional arguments

        Yields:
            PathResult objects containing valid paths

        Raises:
            GraphOperationError: If nodes don't exist or no path exists
            ValueError: If max_paths or max_length is not positive
        """
        self.validate_nodes(start_node, end_node)

        if max_paths is not None and max_paths <= 0:
            raise ValueError("max_paths must be positive")

        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")

        metrics = PerformanceMetrics(operation="all_paths", start_time=time())
        paths_found = 0

        try:
            visited: Set[str] = {start_node}
            current_path: List[Edge] = []

            for path in self._find_paths_recursive(
                start_node,
                end_node,
                visited,
                current_path,
                max_length,
                weight_func,
                metrics,
            ):
                if max_paths and paths_found >= max_paths:
                    break

                if filter_func and not filter_func(path):
                    continue

                paths_found += 1
                yield create_path_result(path, weight_func)

        except Exception as e:
            raise GraphOperationError(f"Error finding paths: {str(e)}") from e

        finally:
            metrics.end_time = time()

    def _find_paths_recursive(
        self,
        current: str,
        target: str,
        visited: Set[str],
        path: List[Edge],
        max_length: Optional[int],
        weight_func: Optional[WeightFunc],
        metrics: PerformanceMetrics,
    ) -> Iterator[List[Edge]]:
        """Recursive helper for finding all paths.

        Args:
            current: Current node being explored
            target: Target node to reach
            visited: Set of visited nodes
            path: Current path being built
            max_length: Maximum allowed path length
            weight_func: Function to compute edge weights
            metrics: Performance metrics to update

        Yields:
            Lists of edges representing valid paths
        """
        if metrics.nodes_explored is None:
            metrics.nodes_explored = 0
        metrics.nodes_explored += 1

        if current == target:
            yield path.copy()
            return

        # Check if adding another edge would exceed max_length
        if max_length is not None and len(path) >= max_length:
            return

        for neighbor in self.graph.get_neighbors(current):
            if neighbor in visited:
                continue

            edge = self.graph.get_edge(current, neighbor)
            if not edge:
                continue

            # Only proceed if adding this edge won't exceed max_length
            if max_length is not None and len(path) + 1 > max_length:
                continue

            visited.add(neighbor)
            path.append(edge)

            yield from self._find_paths_recursive(
                neighbor,
                target,
                visited,
                path,
                max_length,
                weight_func,
                metrics,
            )

            path.pop()
            visited.remove(neighbor)

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
        """Find a single path between nodes.

        Returns the first path found by find_paths().
        """
        try:
            return next(
                self.find_paths(
                    start_node,
                    end_node,
                    max_length=max_length,
                    max_paths=1,
                    filter_func=filter_func,
                    weight_func=weight_func,
                    **kwargs,
                )
            )
        except StopIteration:
            return None
