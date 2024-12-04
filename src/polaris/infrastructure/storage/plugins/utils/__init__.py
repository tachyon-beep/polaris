"""
Utility functions for storage plugins.

This package provides common utility functions used across different storage plugins:
- serialize_datetime: Handle datetime serialization for JSON
- deserialize_datetime: Handle datetime deserialization from JSON
- get_nested_attr: Safely get nested attribute values
- validate_pagination: Validate pagination parameters
- to_dict: Convert objects to dictionaries with datetime handling
"""

from .serialization import deserialize_datetime, serialize_datetime, to_dict
from .validation import get_nested_attr, validate_pagination

__all__ = [
    "serialize_datetime",
    "deserialize_datetime",
    "get_nested_attr",
    "validate_pagination",
    "to_dict",
]
