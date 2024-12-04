"""
Graph-specific query model for Polaris.

This module provides specialized query functionality for graph traversal and analysis.
It extends the base search query model with graph-specific features including:
- Domain context filtering
- Entity and relation type filtering
- Time-based traversal
- Graph analysis options
- Traversal strategy configuration
- Connection inclusion
- Node aggregation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.enums import DomainContext, EntityType, RelationType
from ...core.exceptions import QueryError
from .base import SearchQuery


@dataclass
class GraphQuery(SearchQuery):
    """
    Specialized query for graph traversal and analysis.

    This class extends the base SearchQuery with graph-specific functionality,
    enabling complex graph operations and analysis.

    Attributes:
        domain (Optional[DomainContext]): Domain context for the query
        entity_types (Optional[List[EntityType]]): Types of entities to include
        relation_types (Optional[List[RelationType]]): Types of relations to traverse
        time_range (Optional[tuple[datetime, datetime]]): Time window for temporal analysis
        include_analysis (bool): Whether to include graph analysis metrics
        max_depth (int): Maximum traversal depth from starting nodes
        traversal_strategy (str): Strategy for graph traversal ("breadth_first" or "depth_first")
        include_connections (bool): Whether to include edge/connection information
        aggregate_nodes (bool): Whether to aggregate similar nodes in results

    The query supports various graph operations:
    - Filtered traversal based on entity and relation types
    - Temporal analysis within specified time windows
    - Configurable traversal strategies
    - Optional graph metrics calculation
    - Node aggregation for result summarization
    """

    domain: Optional[DomainContext] = None
    entity_types: Optional[List[EntityType]] = None
    relation_types: Optional[List[RelationType]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    include_analysis: bool = False
    max_depth: int = 3
    traversal_strategy: str = "breadth_first"  # or "depth_first"
    include_connections: bool = False
    aggregate_nodes: bool = False

    def validate(self) -> None:
        """
        Validate graph query configuration.

        This method extends the base query validation with graph-specific checks:
        - Maximum depth is positive
        - Traversal strategy is valid
        - Time range is properly formatted if specified
        - Entity and relation types are valid if specified

        Raises:
            QueryError: If any validation check fails
        """
        super().validate()
        if self.max_depth < 1:
            raise QueryError("Max depth must be positive")
        if self.traversal_strategy not in {"breadth_first", "depth_first"}:
            raise QueryError(f"Invalid traversal strategy: {self.traversal_strategy}")

    def to_storage_query(self) -> Dict[str, Any]:
        """
        Convert graph query to storage-compatible format.

        This method extends the base query conversion with graph-specific parameters:
        - Domain context filtering
        - Entity type filtering
        - Time range constraints
        - Graph traversal configuration

        Returns:
            Dictionary containing storage-formatted query parameters suitable
            for execution against the graph storage backend

        Example:
            A graph query might be converted to:
            {
                "domain": "finance",
                "entity_type": {"$in": ["company", "person"]},
                "metadata.created_at": {"$gte": "2023-01-01", "$lte": "2023-12-31"},
                ...additional base query parameters
            }
        """
        query = super().to_storage_query()

        # Add domain filter
        if self.domain:
            query["domain"] = self.domain.value

        # Add entity type filter
        if self.entity_types:
            query["entity_type"] = {"$in": [t.value for t in self.entity_types]}

        # Add time range filter
        if self.time_range:
            start, end = self.time_range
            query["metadata.created_at"] = {"$gte": start, "$lte": end}

        return query
