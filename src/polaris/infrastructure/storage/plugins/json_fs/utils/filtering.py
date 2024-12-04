"""Utilities for filtering nodes and edges."""

from typing import Any, Dict

from polaris.core.models import Edge, Node
from polaris.infrastructure.storage.plugins.utils import get_nested_attr


def node_matches_filters(node: Node, filters: Dict[str, Any]) -> bool:
    """
    Check if a node matches all given filters.

    Args:
        node: Node to check
        filters: Dictionary of attribute paths and values to match

    Returns:
        True if node matches all filters, False otherwise
    """
    return all(get_nested_attr(node, key) == value for key, value in filters.items())


def edge_matches_filters(edge: Edge, filters: Dict[str, Any]) -> bool:
    """
    Check if an edge matches all given filters.

    Supports both simple attribute matching and $or conditions.

    Args:
        edge: Edge to check
        filters: Dictionary of filters to apply

    Returns:
        True if edge matches filters, False otherwise
    """
    if "$or" in filters:
        return any(
            all(get_nested_attr(edge, key) == value for key, value in or_filter.items())
            for or_filter in filters["$or"]
        )

    return all(get_nested_attr(edge, key) == value for key, value in filters.items())
