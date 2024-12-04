"""
Base class for path finding algorithms.

This module provides the abstract base class that defines the interface for all
path finding algorithm implementations. It ensures consistent behavior across
different algorithms while allowing for specialized implementations.

The PathFinder class provides:
- Common interface for all path finding algorithms
- Node validation functionality
- Type hints for algorithm parameters
- Documentation of expected behavior

Example:
    >>> class CustomPathFinder(PathFinder):
    ...     def find_path(self, start_node, end_node, **kwargs):
    ...         self.validate_nodes(start_node, end_node)
    ...         # Implement custom path finding logic
    ...         return path_result
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Union

from ..exceptions import NodeNotFoundError
from ..graph import Graph
from ..models import Edge
from .models import PathResult
from .utils import WeightFunc

# Type aliases
PathFilter = Callable[[List[Edge]], bool]


class PathFinder(ABC):
    """
    Base class for path finding algorithms.

    This abstract class defines the interface that all path finding
    implementations must follow. It provides common functionality like
    node validation while allowing specific algorithms to implement
    their own path finding logic.

    The class enforces a consistent interface across different path finding
    strategies while providing flexibility in how paths are found and
    what constraints are applied.

    Attributes:
        graph: The graph instance to find paths in

    Example:
        >>> class ShortestPathFinder(PathFinder):
        ...     def find_path(self, start_node, end_node, **kwargs):
        ...         self.validate_nodes(start_node, end_node)
        ...         # Implement Dijkstra's algorithm
        ...         return path_result
    """

    def __init__(self, graph: Graph):
        """
        Initialize path finder with a graph.

        Args:
            graph: The graph instance to find paths in
        """
        self.graph = graph

    def validate_nodes(self, start_node: str, end_node: str) -> None:
        """
        Validate that both nodes exist in the graph.

        This method ensures that both the start and end nodes exist in the graph
        before attempting to find a path between them.

        Args:
            start_node: Starting node ID
            end_node: Target node ID

        Raises:
            NodeNotFoundError: If either node doesn't exist in the graph

        Example:
            >>> finder.validate_nodes("A", "B")  # Validates nodes exist
        """
        if not self.graph.has_node(start_node):
            raise NodeNotFoundError(f"Start node '{start_node}' not found in the graph")
        if not self.graph.has_node(end_node):
            raise NodeNotFoundError(f"End node '{end_node}' not found in the graph")

    @abstractmethod
    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """
        Find path between nodes.

        This abstract method must be implemented by concrete path finding
        algorithms to provide specific path finding logic. The implementation
        should handle all provided parameters appropriately for the specific
        algorithm.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length (optional)
            max_paths: Maximum number of paths to return (optional)
            filter_func: Optional function to filter paths
            weight_func: Optional function for edge weights

        Returns:
            Either a single PathResult or an iterator of PathResults

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            GraphOperationError: If no path exists between the nodes
            ValueError: If provided constraints are invalid

        Example:
            >>> # Find path with custom weight function
            >>> result = finder.find_path(
            ...     "A", "B",
            ...     weight_func=lambda e: e.metadata.weight
            ... )
        """
        pass
