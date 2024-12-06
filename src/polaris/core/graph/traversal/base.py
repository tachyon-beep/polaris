"""Base classes for graph traversal algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, List, Optional, TypeVar, cast

from polaris.core.exceptions import GraphOperationError, NodeNotFoundError
from .cache import PathCache
from .path_models import PathResult
from .types import PathFilter, PathType, WeightFunc
from .utils import calculate_path_weight, create_path_result, get_edge_weight

# Type variable for path finding results
T = TypeVar("T", bound=PathResult)


class PathFinder(ABC, Generic[T]):
    """Abstract base class for path finding algorithms."""

    def __init__(self, graph: Any):
        """Initialize finder with graph."""
        self.graph = graph

    @abstractmethod
    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Optional[T]:
        """Find path between nodes."""
        pass

    def find_paths(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Iterator[T]:
        """Find multiple paths between nodes.

        Default implementation yields single path from find_path.
        Subclasses may override this to provide more efficient implementations.
        """
        path = self.find_path(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            weight_func=weight_func,
            **kwargs,
        )
        if path is not None:
            yield path

    def validate_nodes(self, start_node: str, end_node: str) -> None:
        """Validate that nodes exist in graph."""
        if not self.graph.has_node(start_node):
            raise NodeNotFoundError(f"Start node '{start_node}' not found")
        if not self.graph.has_node(end_node):
            raise NodeNotFoundError(f"End node '{end_node}' not found")


class PathFinding(Generic[T]):
    """Interface for path finding operations."""

    def __init__(self):
        """Initialize path finding interface."""
        pass

    def _validate_length(self, max_length: Optional[int]) -> None:
        """Validate max_length parameter."""
        if max_length is not None:
            if not isinstance(max_length, int):
                raise TypeError("max_length must be an integer")
            if max_length < 0:
                raise ValueError("max_length must be non-negative")

    def _get_edge_weight(self, edge: Any, weight_func: Optional[WeightFunc] = None) -> float:
        """Get weight of an edge using optional weight function."""
        return get_edge_weight(edge, weight_func)

    def _calculate_path_weight(
        self, path: List[Any], weight_func: Optional[WeightFunc] = None
    ) -> float:
        """Calculate total weight of a path."""
        return calculate_path_weight(path, weight_func)

    def _create_path_result(
        self,
        path: List[Any],
        weight_func: Optional[WeightFunc] = None,
        graph: Optional[Any] = None,
    ) -> T:
        """Create path result from path."""
        return cast(T, create_path_result(path, weight_func))

    @staticmethod
    def get_cache_metrics() -> dict:
        """Get cache performance metrics."""
        return PathCache.get_metrics()

    def shortest_path(
        self,
        graph: Any,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> T:
        """Find shortest path between nodes."""
        from .algorithms.shortest_path import ShortestPathFinder

        # Validate max_length before creating finder
        self._validate_length(max_length)

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
            return cast(T, cached_result)

        # Special case: max_length=1 means only direct edges allowed
        if max_length == 1:
            edge = graph.get_edge(start_node, end_node)
            if not edge:
                raise GraphOperationError(
                    f"No path of length <= 1 exists between {start_node} and {end_node}"
                )
            result = self._create_path_result([edge], weight_func)
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
        return cast(T, result)

    def bidirectional_search(
        self,
        graph: Any,
        start_node: str,
        end_node: str,
        max_depth: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> T:
        """Find path using bidirectional search."""
        from .algorithms.bidirectional import BidirectionalFinder

        # Validate max_depth before creating finder
        self._validate_length(max_depth)

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
            return cast(T, cached_result)

        # Special case: max_depth=1 means only direct edges allowed
        if max_depth == 1:
            edge = graph.get_edge(start_node, end_node)
            if not edge:
                raise GraphOperationError(
                    f"No path of length <= 1 exists between {start_node} and {end_node}"
                )
            result = self._create_path_result([edge], weight_func)
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
        return cast(T, result)

    def all_paths(
        self,
        graph: Any,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        **kwargs,
    ) -> Iterator[T]:
        """Find all paths between nodes."""
        from .algorithms.all_paths import AllPathsFinder

        # Validate max_length before creating finder
        self._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            return iter([])  # Return empty iterator for length 0

        # Note: We don't cache all_paths results since they're iterators
        finder = AllPathsFinder(graph)
        paths = finder.find_paths(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            **kwargs,
        )
        return cast(Iterator[T], paths)

    def filtered_paths(
        self,
        graph: Any,
        start_node: str,
        end_node: str,
        filter_func: PathFilter,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        **kwargs,
    ) -> Iterator[T]:
        """Find paths that satisfy a filter function."""
        from .algorithms.all_paths import AllPathsFinder

        # Validate max_length before creating finder
        self._validate_length(max_length)

        # Special case: max_length=0 means only direct paths
        if max_length == 0:
            return iter([])  # Return empty iterator for length 0

        # Note: We don't cache filtered_paths results since they're iterators
        finder = AllPathsFinder(graph)
        paths = finder.find_paths(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            **kwargs,
        )
        return cast(Iterator[T], paths)

    def find_paths(
        self,
        graph: Any,
        start_node: str,
        end_node: str,
        path_type: PathType = PathType.SHORTEST,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> T | Iterator[T]:
        """Generic path finding interface."""
        from .algorithms.all_paths import AllPathsFinder
        from .algorithms.shortest_path import ShortestPathFinder
        from .algorithms.bidirectional import BidirectionalFinder

        # Validate max_length before proceeding
        self._validate_length(max_length)

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
            paths = finder.find_paths(
                start_node=start_node,
                end_node=end_node,
                max_length=max_length,
                max_paths=max_paths,
                filter_func=filter_func,
                weight_func=weight_func,
                **kwargs,
            )
            return cast(Iterator[T], paths)
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
                return cast(T, cached_result)

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
            return cast(T, result)
