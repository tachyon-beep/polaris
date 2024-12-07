"""
Search results container and formatting for Polaris.

This module provides the container class for search results and handles result
formatting. It supports various types of search results including:
- Basic search results
- Graph search results with relationships
- Semantic search results with similarity scores
- Faceted search results with aggregations
- Search suggestions for query refinement
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .base import SearchQuery
from .graph import GraphQuery
from .semantic import SemanticQuery


@dataclass
class SearchResults:
    """
    Container for search results across all query types.

    This class provides a standardized format for returning search results,
    regardless of the query type (basic, graph, or semantic). It includes
    both the result items and metadata about the search operation.

    Attributes:
        items (List[Dict[str, Any]]): List of matched items with their data
        total (int): Total number of matches (before pagination)
        page (int): Current page number
        page_size (int): Number of items per page
        execution_time (float): Time taken to execute the search (in seconds)
        query (Union[SearchQuery, GraphQuery, SemanticQuery]): Original query object
        facets (Optional[Dict[str, Any]]): Faceted aggregations of results
        suggestions (Optional[List[str]]): Search suggestions for query refinement
        aggregations (Optional[Dict[str, Any]]): Statistical aggregations of results

    The results container supports various features:
    - Pagination metadata
    - Execution timing
    - Faceted search results
    - Query suggestions
    - Statistical aggregations

    Example structure of items:
        [
            {
                "id": "entity123",
                "name": "Example Entity",
                "type": "document",
                "metadata": {...},
                "similarity_score": 0.95,  # For semantic search
                "relationships": [...],    # For graph search
            },
            ...
        ]

    Example structure of facets:
        {
            "entity_type": {
                "document": 45,
                "person": 23,
                "organization": 12
            },
            "domain": {
                "finance": 30,
                "technology": 50
            }
        }

    Example structure of aggregations:
        {
            "avg_confidence": 0.85,
            "date_histogram": {
                "2023-01": 45,
                "2023-02": 62
            }
        }
    """

    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    execution_time: float
    query: Union[SearchQuery, GraphQuery, SemanticQuery]
    facets: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    aggregations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search results to dictionary format.

        This method transforms the results into a serializable dictionary format
        suitable for API responses or storage. It includes:
        - Result items
        - Pagination information
        - Execution metadata
        - Optional facets and aggregations

        Returns:
            Dictionary containing formatted search results and metadata

        Example:
            {
                "items": [...],
                "total": 100,
                "page": 1,
                "page_size": 10,
                "total_pages": 10,
                "execution_time": 0.125,
                "facets": {...},
                "suggestions": [...],
                "aggregations": {...}
            }
        """
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": (self.total + self.page_size - 1) // self.page_size,
            "execution_time": self.execution_time,
            "facets": self.facets,
            "suggestions": self.suggestions,
            "aggregations": self.aggregations,
        }
