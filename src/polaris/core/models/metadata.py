"""
Metadata utilities for the knowledge graph system.

This module provides shared metadata functionality and utilities used
across different types of entities in the knowledge graph system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .base import validate_dataclass, validate_date_order


@runtime_checkable
class MetadataProvider(Protocol):
    """Protocol defining the interface for metadata-enabled entities."""

    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        ...

    @property
    def last_modified(self) -> datetime:
        """Get last modification timestamp."""
        ...

    @property
    def source(self) -> str:
        """Get entity source."""
        ...


@validate_dataclass
@dataclass
class BaseMetadata:
    """
    Base metadata class providing common metadata functionality.

    This class serves as a foundation for specific metadata types
    in the knowledge graph system.

    Attributes:
        created_at (datetime): Timestamp of entity creation
        last_modified (datetime): Timestamp of last modification
        source (str): Origin or source system
        custom_attributes (Dict[str, Any]): Additional custom attributes
    """

    created_at: datetime
    last_modified: datetime
    source: str
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate base metadata after initialization."""
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")
        validate_date_order(self.created_at, self.last_modified)

    def update_modified(self) -> None:
        """Update the last_modified timestamp to current time."""
        self.last_modified = datetime.now()

    def add_custom_attribute(self, key: str, value: Any) -> None:
        """
        Add a custom attribute and update last_modified.

        Args:
            key (str): Attribute key
            value (Any): Attribute value
        """
        self.custom_attributes[key] = value
        self.update_modified()

    def remove_custom_attribute(self, key: str) -> None:
        """
        Remove a custom attribute and update last_modified.

        Args:
            key (str): Attribute key to remove
        """
        if key in self.custom_attributes:
            del self.custom_attributes[key]
            self.update_modified()
