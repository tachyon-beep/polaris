"""
Base class for path finding algorithms.

This module provides the abstract base class that defines the interface for all
path finding algorithm implementations. It ensures consistent behavior across
different algorithms while allowing for specialized implementations.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Iterator, Callable, List

from ..exceptions import NodeNotFoundError
from ..graph import Graph
from ..models import Edge
from .models import PathResult

# Type aliases
WeightFunc = Callable[[Edge], float]
PathFilter = Callable[[List[Edge]], bool]

# Type variable for path finding return types
P = TypeVar("P", PathResult, Iterator[PathResult], covariant=True)


class PathFinder(ABC, Generic[P]):
    """
    Base class for path finding algorithms.

    This abstract class defines the interface that all path finding
    implementations must follow. It provides common functionality like
    node validation while allowing specific algorithms to implement
    their own path finding logic.

    Type Parameters:
        P: The return type of find_path (either PathResult or Iterator[PathResult])

    Attributes:
        graph: The graph instance to find paths in
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

        Args:
            start_node: Starting node ID
            end_node: Target node ID

        Raises:
            NodeNotFoundError: If either node doesn't exist in the graph
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
    ) -> P:
        """
        Find path between nodes.

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
        """
        pass
