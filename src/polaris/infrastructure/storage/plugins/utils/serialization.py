"""
Serialization utilities for storage plugins.

This module provides functions for handling serialization and deserialization
of objects, with special handling for datetime objects.
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict


def serialize_datetime(obj: Any) -> Any:
    """
    Handle datetime serialization for JSON.

    Args:
        obj: Object to serialize

    Returns:
        ISO format string if obj is datetime, otherwise unchanged obj
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def deserialize_datetime(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle datetime deserialization from JSON.

    Recursively processes dictionary to convert ISO format strings to datetime
    objects for keys that are expected to contain dates.

    Args:
        obj: Dictionary potentially containing datetime strings

    Returns:
        Dictionary with datetime strings converted to datetime objects
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ["created_at", "last_modified"] and isinstance(value, str):
                obj[key] = datetime.fromisoformat(value)
            elif isinstance(value, dict):
                obj[key] = deserialize_datetime(value)
    return obj


def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert object to dictionary with datetime handling.

    Args:
        obj: Object to convert to dictionary

    Returns:
        Dictionary representation with datetime values serialized
    """
    return asdict(obj, dict_factory=lambda x: {k: serialize_datetime(v) for k, v in x})
