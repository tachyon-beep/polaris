"""Graph path finding functionality."""

from typing import List, Optional, Union, Type, Any, Iterator
from itertools import islice

from ..exceptions import GraphOperationError
from .algorithms.shortest_path import ShortestPathFinder
from .algorithms.bidirectional import BidirectionalFinder
from .algorithms.all_paths import AllPathsFinder
from .base import PathFinder
from .models import PathResult, PathValidationError
from .types import (
    PathType,
    WeightFunc,
    PathFilter,
    allow_negative_weights,
)
from .utils import calculate_path_weight, create_path_result, get_edge_weight
from .cache import PathCache

# Constants
DEFAULT_MAX_PATH_LENGTH = 100

# Re-export types
__all__ = [
    "PathType",
    "WeightFunc",
    "PathFilter",
    "PathResult",
    "PathValidationError",
    "allow_negative_weights",
    "DEFAULT_MAX_PATH_LENGTH",
]


class PathFinding:
    """Static interface for path finding operations."""

    @staticmethod
    def _validate_length(max_length: Optional[int]) -> None:
        """Validate max_length parameter."""
        if max_length is not None:
            if not isinstance(max_length, int):
                raise TypeError("max_length must be an integer")
            if max_length < 0:
                raise ValueError("max_length must be non-negative")

    @staticmethod
    def _get_edge_weight(edge: Any, weight_func: Optional[WeightFunc] = None) -> float:
        """Get weight of an edge using optional weight function."""
        return get_edge_weight(edge, weight_func)

    @staticmethod
    def _calculate_path_weight(path: List[Any], weight_func: Optional[WeightFunc] = None) -> float:
        """Calculate total weight of a path."""
        return calculate_path_weight(path, weight_func)

    @staticmethod
    def _create_path_result(
        path: List[Any], weight_func: Optional[WeightFunc] = None, graph: Optional[Any] = None
    ) -> PathResult:
        """Create path result from path."""
        return create_path_result(path, weight_func, graph)

    @staticmethod
    def get_cache_metrics() -> dict:
        """Get cache performance metrics."""
        return PathCache.get_metrics()

    @classmethod
    def shortest_path(
        cls,
        graph: Any,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """Find shortest path between nodes."""
        # Validate max_length before creating finder
        cls._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            raise GraphOperationError(
                f"No path of length 0 exists between {start_node} and {end_node}"
            )

        # Generate cache key
        weight_func_name = weight_func.__name__ if weight_func else None
        cache_key = PathCache.get_cache_key(
            start_node, end_node, PathType.SHORTEST.value, weight_func_name, max_length
        )

        # Check cache
        if cached_result := PathCache.get(cache_key):
            return cached_result

        # Special case: max_length=1 means only direct edges allowed
        if max_length == 1:
            edge = graph.get_edge(start_node, end_node)
            if not edge:
                raise GraphOperationError(
                    f"No path of length <= 1 exists between {start_node} and {end_node}"
                )
            result = create_path_result([edge], weight_func)
            PathCache.put(cache_key, result)
            return result

        finder = ShortestPathFinder(graph)
        result = finder.find_path(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            weight_func=weight_func,
            **kwargs,
        )
        if result is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Cache result
        PathCache.put(cache_key, result)
        return result

    @classmethod
    def bidirectional_search(
        cls,
        graph: Any,
        start_node: str,
        end_node: str,
        max_depth: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """Find path using bidirectional search."""
        # Validate max_depth before creating finder
        cls._validate_length(max_depth)

        # Special case: max_depth=0 means only direct paths
        if max_depth == 0:
            raise GraphOperationError(
                f"No path of length 0 exists between {start_node} and {end_node}"
            )

        # Generate cache key
        weight_func_name = weight_func.__name__ if weight_func else None
        cache_key = PathCache.get_cache_key(
            start_node, end_node, PathType.BIDIRECTIONAL.value, weight_func_name, max_depth
        )

        # Check cache
        if cached_result := PathCache.get(cache_key):
            return cached_result

        # Special case: max_depth=1 means only direct edges allowed
        if max_depth == 1:
            edge = graph.get_edge(start_node, end_node)
            if not edge:
                raise GraphOperationError(
                    f"No path of length <= 1 exists between {start_node} and {end_node}"
                )
            result = create_path_result([edge], weight_func)
            PathCache.put(cache_key, result)
            return result

        finder = BidirectionalFinder(graph)
        result = finder.find_path(
            start_node=start_node,
            end_node=end_node,
            max_length=max_depth,
            weight_func=weight_func,
            **kwargs,
        )
        if result is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Cache result
        PathCache.put(cache_key, result)
        return result

    @classmethod
    def all_paths(
        cls,
        graph: Any,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        **kwargs,
    ) -> Iterator[PathResult]:
        """Find all paths between nodes."""
        # Validate max_length before creating finder
        cls._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            return iter([])  # Return empty iterator for length 0

        # Note: We don't cache all_paths results since they're iterators
        finder = AllPathsFinder(graph)
        return finder.find_paths(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            **kwargs,
        )

    @classmethod
    def filtered_paths(
        cls,
        graph: Any,
        start_node: str,
        end_node: str,
        filter_func: PathFilter,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        **kwargs,
    ) -> Iterator[PathResult]:
        """Find paths that satisfy a filter function."""
        # Validate max_length before creating finder
        cls._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            return iter([])  # Return empty iterator for length 0

        # Note: We don't cache filtered_paths results since they're iterators
        finder = AllPathsFinder(graph)
        return finder.find_paths(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            **kwargs,
        )

    @classmethod
    def find_paths(
        cls,
        graph: Any,
        start_node: str,
        end_node: str,
        path_type: PathType = PathType.SHORTEST,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """Generic path finding interface."""
        # Validate max_length before proceeding
        cls._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            if path_type in (PathType.ALL, PathType.ALL_PATHS, PathType.FILTERED):
                return iter([])  # Return empty iterator for length 0
            else:
                raise GraphOperationError(
                    f"No path of length 0 exists between {start_node} and {end_node}"
                )

        if path_type in (PathType.ALL, PathType.ALL_PATHS, PathType.FILTERED):
            # Don't cache iterator results
            finder = AllPathsFinder(graph)
            return finder.find_paths(
                start_node=start_node,
                end_node=end_node,
                max_length=max_length,
                max_paths=max_paths,
                filter_func=filter_func,
                weight_func=weight_func,
                **kwargs,
            )
        else:
            # Generate cache key for single path results
            weight_func_name = weight_func.__name__ if weight_func else None
            cache_key = PathCache.get_cache_key(
                start_node,
                end_node,
                path_type.value,
                weight_func_name,
                max_length,
            )

            # Check cache
            if cached_result := PathCache.get(cache_key):
                return cached_result

            finder = (
                ShortestPathFinder(graph)
                if path_type == PathType.SHORTEST
                else BidirectionalFinder(graph)
            )
            result = finder.find_path(
                start_node=start_node,
                end_node=end_node,
                max_length=max_length,
                max_paths=max_paths,
                filter_func=filter_func,
                weight_func=weight_func,
                **kwargs,
            )
            if result is None:
                raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

            # Cache result
            PathCache.put(cache_key, result)
            return result
