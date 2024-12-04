"""
Graph path finding functionality.

This module provides path finding algorithms and utilities for working with
graph paths, including:
- Shortest path finding
- All paths enumeration
- Bidirectional search
- Path validation and caching
"""

from typing import Callable, Iterator, List, Optional, Union

from ..graph import Graph
from ..models import Edge
from .algorithms.all_paths import AllPathsFinder
from .algorithms.bidirectional import BidirectionalPathFinder
from .algorithms.shortest_path import ShortestPathFinder
from .base import PathFinder
from .cache import PathCache, PATH_CACHE_SIZE, PATH_CACHE_TTL
from .models import PathResult, PathValidationError
from .utils import WeightFunc, calculate_path_weight, create_path_result, get_edge_weight

# Constants
DEFAULT_MAX_PATH_LENGTH = 50


class PathType:
    """Path finding algorithm types."""

    SHORTEST = "shortest"
    ALL = "all"
    FILTERED = "filtered"
    BIDIRECTIONAL = "bidirectional"


class PathFinding:
    """
    Static interface for graph path finding operations.

    This class provides a unified interface to various path finding algorithms,
    handling algorithm selection and common operations like caching.
    """

    @staticmethod
    def _get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
        """Get edge weight using weight function or default."""
        return get_edge_weight(edge, weight_func)

    @staticmethod
    def _calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc] = None) -> float:
        """Calculate total path weight."""
        return calculate_path_weight(path, weight_func)

    @staticmethod
    def _create_path_result(
        path: List[Edge], weight_func: Optional[WeightFunc] = None, graph: Optional[Graph] = None
    ) -> PathResult:
        """Create PathResult instance."""
        return create_path_result(path, weight_func, graph)

    @classmethod
    def shortest_path(
        cls,
        graph: Graph,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc] = None,
    ) -> PathResult:
        """Find shortest path between nodes."""
        finder = ShortestPathFinder(graph)
        return finder.find_path(start_node, end_node, weight_func=weight_func)

    @classmethod
    def all_paths(
        cls,
        graph: Graph,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Iterator[PathResult]:
        """Find all paths between nodes."""
        finder = AllPathsFinder(graph)
        return finder.find_path(
            start_node,
            end_node,
            max_length=max_length,
            max_paths=max_paths,
            filter_func=filter_func,
            weight_func=weight_func,
        )

    @classmethod
    def bidirectional_search(
        cls,
        graph: Graph,
        start_node: str,
        end_node: str,
        max_depth: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> PathResult:
        """Find path using bidirectional search."""
        finder = BidirectionalPathFinder(graph)
        return finder.find_path(start_node, end_node, max_length=max_depth, weight_func=weight_func)

    @classmethod
    def find_paths(
        cls,
        graph: Graph,
        start_node: str,
        end_node: str,
        path_type: str = PathType.SHORTEST,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """
        Find paths between nodes using specified algorithm.

        Args:
            graph: Graph to search in
            start_node: Starting node ID
            end_node: Target node ID
            path_type: Algorithm to use (default: shortest)
            max_length: Maximum path length
            max_paths: Maximum number of paths
            filter_func: Optional path filter function
            weight_func: Optional edge weight function

        Returns:
            Either a single PathResult or an iterator of PathResults
        """
        if path_type == PathType.SHORTEST:
            return cls.shortest_path(graph, start_node, end_node, weight_func=weight_func)
        elif path_type == PathType.BIDIRECTIONAL:
            return cls.bidirectional_search(
                graph, start_node, end_node, max_depth=max_length, weight_func=weight_func
            )
        elif path_type in (PathType.ALL, PathType.FILTERED):
            return cls.all_paths(
                graph,
                start_node,
                end_node,
                max_length=max_length,
                max_paths=max_paths,
                filter_func=filter_func,
                weight_func=weight_func,
            )
        else:
            raise ValueError(f"Unknown path type: {path_type}")

    @classmethod
    def get_cache_metrics(cls) -> dict:
        """Get path cache metrics."""
        return PathCache.get_metrics()
