"""
Type definitions for path finding functionality.

This module provides type aliases and protocols used across the path finding
implementation to ensure type safety and clarity.
"""

from typing import Callable, List, TypeVar, Union, Iterator, Protocol
from ..models import Edge
from .models import PathResult

# Type aliases
WeightFunc = Callable[[Edge], float]
PathFilter = Callable[[List[Edge]], bool]

# Type variable for path finding return types
P = TypeVar("P", PathResult, Iterator[PathResult], covariant=True)


class PathFinderProtocol(Protocol[P]):
    """Protocol defining the interface for path finders."""

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: int | None = None,
        max_paths: int | None = None,
        filter_func: PathFilter | None = None,
        weight_func: WeightFunc | None = None,
    ) -> P: ...
