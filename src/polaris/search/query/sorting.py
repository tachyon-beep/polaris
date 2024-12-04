"""
Sorting models for search queries in Polaris.

This module provides sorting functionality for search results, enabling:
- Field-based sorting
- Direction control (ascending/descending)
- Storage-compatible sort criteria generation

The sorting capabilities support all query types (basic, semantic, and graph)
and can be applied to any field in the search results.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ...core.exceptions import QueryError


@dataclass
class SearchSort:
    """
    Sorting criteria for search results.

    This class defines how search results should be ordered, specifying
    both the field to sort by and the sort direction.

    Attributes:
        field (str): The field to sort results by
        direction (str): Sort direction ("asc" for ascending, "desc" for descending)

    Example uses:
        - Sort by creation date: SearchSort(field="created_at", direction="desc")
        - Sort by name: SearchSort(field="name", direction="asc")
        - Sort by relevance: SearchSort(field="score", direction="desc")
        - Sort by size: SearchSort(field="size", direction="asc")

    The sort criteria can be applied to:
        - Basic fields (name, date, type)
        - Metadata fields (created_at, updated_at)
        - Computed fields (score, relevance)
        - Nested fields (metadata.confidence)
    """

    field: str
    direction: str = "asc"  # "asc" or "desc"

    def validate(self) -> None:
        """
        Validate sort configuration.

        This method ensures the sort direction is valid ("asc" or "desc").
        Additional validation could be added for field names or specific
        sort requirements.

        Raises:
            QueryError: If the sort direction is invalid
        """
        if self.direction not in {"asc", "desc"}:
            raise QueryError(f"Invalid sort direction: {self.direction}")

    def to_storage_sort(self) -> Dict[str, Any]:
        """
        Convert sort criteria to storage-compatible format.

        This method transforms the sort configuration into a format that
        can be understood by the storage backend.

        Returns:
            Dictionary containing storage-formatted sort criteria where:
            - 1 represents ascending order
            - -1 represents descending order

        Example:
            SearchSort(field="name", direction="asc").to_storage_sort()
            returns: {"name": 1}

            SearchSort(field="created_at", direction="desc").to_storage_sort()
            returns: {"created_at": -1}
        """
        return {self.field: 1 if self.direction == "asc" else -1}
