"""
Validation utilities for storage plugins.

This module provides functions for validating data and accessing nested attributes
safely across storage plugin implementations.
"""

from typing import Any


def get_nested_attr(obj: Any, attr_path: str) -> Any:
    """
    Safely get nested attribute value from an object.

    Traverses a dot-separated path to access nested attributes of an object,
    returning None if any part of the path is invalid.

    Args:
        obj: Object to get attribute from
        attr_path: Dot-separated path to attribute (e.g., "metadata.created_at")

    Returns:
        Attribute value if found, None if path is invalid

    Example:
        >>> obj = Node(metadata=NodeMetadata(created_at=datetime.now()))
        >>> get_nested_attr(obj, "metadata.created_at")
        datetime.datetime(...)
    """
    curr = obj
    for attr in attr_path.split("."):
        if not hasattr(curr, attr):
            return None
        curr = getattr(curr, attr)
    return curr


def validate_pagination(offset: int, limit: int) -> None:
    """
    Validate pagination parameters.

    Ensures that pagination parameters are within valid ranges:
    - offset must be non-negative
    - limit must be positive

    Args:
        offset: Number of items to skip
        limit: Maximum number of items to return

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> validate_pagination(0, 10)  # Valid
        >>> validate_pagination(-1, 10)  # Raises ValueError
        >>> validate_pagination(0, 0)   # Raises ValueError
    """
    if offset < 0:
        raise ValueError("Offset must be non-negative")
    if limit < 1:
        raise ValueError("Limit must be positive")
