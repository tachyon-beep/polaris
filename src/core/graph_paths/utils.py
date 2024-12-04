"""
Utility functions for graph path finding.

This module provides common utility functions used across different path finding
algorithms, including:
- Edge weight calculation with custom weight functions
- Path weight calculation
- Path result creation

These utilities ensure consistent behavior across all path finding implementations
while providing flexibility through custom weight functions.

Example:
    >>> weight = get_edge_weight(edge, lambda e: e.metadata.weight)
    >>> path_weight = calculate_path_weight(edges, weight_func)
    >>> result = create_path_result(edges, weight_func, graph)
"""

from typing import Callable, List, Optional

from ..graph import Graph
from ..models import Edge
from .models import PathResult

# Type aliases
Weight = float
WeightFunc = Callable[[Edge], Weight]


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc]) -> Weight:
    """
    Calculate edge weight using optional weight function.

    This function applies a custom weight function to an edge or uses
    a default weight of 1.0 if no function is provided.

    Args:
        edge: The edge to calculate weight for
        weight_func: Optional function to calculate custom weight

    Returns:
        The calculated edge weight

    Raises:
        ValueError: If weight_func returns zero or negative value

    Example:
        >>> weight = get_edge_weight(edge, lambda e: e.metadata.confidence)
        >>> print(f"Edge weight: {weight}")
    """
    weight = weight_func(edge) if weight_func is not None else 1.0
    if weight <= 0:
        raise ValueError(
            f"Edge weight must be positive. Got {weight} for "
            f"edge {edge.from_entity}->{edge.to_entity}"
        )
    return weight


def calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc]) -> float:
    """
    Calculate total path weight.

    Computes the total weight of a path by summing the weights of all edges,
    using either custom weight function or default weights.

    Args:
        path: List of edges forming the path
        weight_func: Optional function to calculate custom weights

    Returns:
        Total weight of the path

    Example:
        >>> total = calculate_path_weight(path, lambda e: e.impact_score)
        >>> print(f"Total path weight: {total}")
    """
    return sum(get_edge_weight(edge, weight_func) for edge in path)


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
        >>> print(f"Path length: {result.length}")
        >>> print(f"Total weight: {result.total_weight}")
    """
    total_weight = calculate_path_weight(path, weight_func)
    result = PathResult(path=path, total_weight=total_weight, length=len(path))
    # Don't validate here - let the caller handle validation
    return result
