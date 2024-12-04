"""
Utility functions for path finding operations.

This module provides helper functions used across different path finding
algorithms, including:
- Path result creation
- Weight calculation
- Path validation
"""

from typing import List, Optional, TypeVar, Callable, Union

from ..graph import Graph
from ..models import Edge
from .models import PathResult

# Type alias for weight functions
WeightFunc = Callable[[Edge], float]


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
    """
    Get weight of an edge using optional weight function.

    Args:
        edge: Edge to get weight for
        weight_func: Optional function to calculate custom weight

    Returns:
        Weight of the edge

    Raises:
        ValueError: If weight function returns non-positive value

    Example:
        >>> weight = get_edge_weight(edge, lambda e: e.metadata.weight)
    """
    if weight_func is None:
        return 1.0

    weight = weight_func(edge)
    if weight <= 0:
        raise ValueError("Edge weight must be positive")
    return weight


def calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc] = None) -> float:
    """
    Calculate total weight of a path.

    Args:
        path: List of edges forming the path
        weight_func: Optional function to calculate custom weights

    Returns:
        Total weight of the path

    Example:
        >>> total = calculate_path_weight(path, lambda e: e.metadata.weight)
    """
    if not path:
        return 0.0

    if weight_func is None:
        # Default to weight of 1.0 per edge
        return float(len(path))

    return sum(weight_func(edge) for edge in path)


def create_path_result(
    path: List[Edge], weight_func: Optional[WeightFunc], graph: Optional[Graph] = None
) -> PathResult:
    """
    Create PathResult from path.

    Creates a PathResult object containing the path and its properties,
    calculating total weight using provided weight function.

    Args:
        path: List of edges forming the path
        weight_func: Optional function to calculate custom weights
        graph: Optional graph instance for validation

    Returns:
        PathResult object containing the path and its properties

    Example:
        >>> result = create_path_result(path, lambda e: e.metadata.weight, graph)
        >>> print(f"Path length: {len(result)}")
        >>> print(f"Total weight: {result.total_weight}")
    """
    total_weight = calculate_path_weight(path, weight_func)
    return PathResult(path=path, total_weight=total_weight)
