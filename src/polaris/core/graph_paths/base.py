from abc import ABC, abstractmethod
from typing import Optional, Iterator, Any, Generic, TypeVar

from polaris.core.exceptions import NodeNotFoundError
from polaris.core.graph_paths.types import WeightFunc, PathFilter
from polaris.core.graph_paths.models import PathResult

# Type variable for path finding results
T = TypeVar("T", bound=PathResult)


class PathFinder[T](ABC):
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
