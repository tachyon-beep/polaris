"""
Base search query model for Polaris.

This module defines the foundational search query model that serves as the base
for all specialized query types (semantic, graph, etc.). It provides core query
functionality including:
- Text-based search
- Filtering
- Sorting
- Pagination
- Metadata inclusion
- Confidence thresholds
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.exceptions import QueryError
from .filters import SearchFilter
from .sorting import SearchSort


@dataclass
class SearchQuery:
    """
    Base search query model providing core search functionality.

    This class defines the common attributes and behaviors shared by all query types.
    It handles basic search operations like filtering, sorting, and pagination while
    providing extension points for specialized query types.

    Attributes:
        query_text (Optional[str]): The main search text/keywords
        filters (List[SearchFilter]): List of filters to apply to search results
        sort (Optional[SearchSort]): Sorting criteria for results
        page (int): Current page number (1-based)
        page_size (int): Number of results per page
        include_metadata (bool): Whether to include metadata in results
        min_confidence (float): Minimum confidence threshold for results
        search_fields (Optional[List[str]]): Specific fields to search in
    """

    query_text: Optional[str] = None
    filters: List[SearchFilter] = field(default_factory=list)
    sort: Optional[SearchSort] = None
    page: int = 1
    page_size: int = 50
    include_metadata: bool = True
    min_confidence: float = 0.0
    search_fields: Optional[List[str]] = None

    def validate(self) -> None:
        """
        Validate search query configuration.

        This method ensures that all query parameters are valid and within
        acceptable ranges. It checks:
        - Page number is positive
        - Page size is positive
        - Confidence threshold is between 0 and 1
        - All filters are valid
        - Sort configuration is valid

        Raises:
            QueryError: If any validation check fails
        """
        if self.page < 1:
            raise QueryError("Page number must be positive")
        if self.page_size < 1:
            raise QueryError("Page size must be positive")
        if self.min_confidence < 0 or self.min_confidence > 1:
            raise QueryError("Confidence must be between 0 and 1")
        for filter in self.filters:
            filter.validate()
        if self.sort:
            self.sort.validate()

    def to_storage_query(self) -> Dict[str, Any]:
        """
        Convert query to storage-compatible format.

        This method transforms the query into a format that can be understood
        by the storage backend. It handles:
        - Filter conversion
        - Confidence threshold application
        - Text search configuration
        - Field-specific search settings

        Returns:
            Dictionary containing storage-formatted query parameters suitable
            for execution against the storage backend
        """
        query = {}

        # Convert filters
        if self.filters:
            filter_conditions = [f.to_storage_filter() for f in self.filters]
            if len(filter_conditions) == 1:
                query.update(filter_conditions[0])
            else:
                query["$and"] = filter_conditions

        # Add confidence filter if specified
        if self.min_confidence > 0:
            query["metadata.metrics.confidence"] = {"$gte": self.min_confidence}

        # Add text search if specified
        if self.query_text:
            if self.search_fields:
                # Search in specific fields
                text_conditions = [
                    {field: {"$regex": self.query_text, "$options": "i"}}
                    for field in self.search_fields
                ]
                query["$or"] = text_conditions
            else:
                # Search in default fields
                query["$text"] = {"$search": self.query_text}

        return query
