"""
Edge models for the knowledge graph system.

This module defines the models representing connections between nodes
in the knowledge graph, including edges and their associated metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .base import (
    validate_dataclass,
    validate_date_order,
    validate_metric_range,
    validate_custom_metrics,
)
from ..enums import RelationType


@validate_dataclass
@dataclass
class EdgeMetadata:
    """
    Metadata associated with an edge.

    Stores administrative and qualitative information about connections
    between nodes in the knowledge graph.

    Attributes:
        created_at (datetime): Timestamp of edge creation
        last_modified (datetime): Timestamp of last modification
        confidence (float): Confidence score (0.0 to 1.0)
        source (str): Origin of the edge
        bidirectional (bool): Whether edge is bidirectional
        temporal (bool): Whether edge has temporal aspects
        weight (float): Edge strength or importance
        custom_attributes (Dict[str, Any]): Additional custom attributes
    """

    created_at: datetime
    last_modified: datetime
    confidence: float
    source: str
    bidirectional: bool = False
    temporal: bool = False
    weight: float = 1.0
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")
        validate_date_order(self.created_at, self.last_modified)
        validate_metric_range("confidence", self.confidence)
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


@validate_dataclass
@dataclass
class Edge:
    """
    Base edge model representing a connection in the knowledge graph.

    Edges represent connections or relationships between nodes,
    forming the connections of the knowledge graph.

    Attributes:
        from_entity (str): Source node ID
        to_entity (str): Target node ID
        relation_type (RelationType): Type of relationship
        metadata (EdgeMetadata): Administrative metadata
        impact_score (float): Measure of connection impact (0.0 to 1.0)
        attributes (Dict[str, Any]): Custom attributes
        context (Optional[str]): Additional context
        validation_status (str): Validation state
        custom_metrics (Dict[str, Tuple[float, float, float]]): Custom metrics with ranges
            Each tuple contains (value, min_range, max_range)
    """

    from_entity: str
    to_entity: str
    relation_type: RelationType
    metadata: EdgeMetadata
    impact_score: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    validation_status: str = "unverified"
    custom_metrics: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge after initialization."""
        if not self.from_entity.strip():
            raise ValueError("source node must be a non-empty string")
        if not self.to_entity.strip():
            raise ValueError("target node must be a non-empty string")
        if not isinstance(self.relation_type, RelationType):
            raise TypeError("relation_type must be a RelationType enum")
        validate_metric_range("impact_score", self.impact_score)
        if self.validation_status not in {"unverified", "verified", "invalid"}:
            raise ValueError("validation_status must be one of: unverified, verified, invalid")

        # Validate custom metrics ranges
        validate_custom_metrics(self.custom_metrics)

    def add_custom_metric(
        self, name: str, value: float, min_range: float = 0.0, max_range: float = 1.0
    ) -> None:
        """
        Add a custom metric with specified range.

        Args:
            name (str): Name of the metric
            value (float): Metric value
            min_range (float): Minimum allowed value (default: 0.0)
            max_range (float): Maximum allowed value (default: 1.0)

        Raises:
            ValueError: If the value is outside the specified range
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Custom metric {name} must be a numeric value")
        validate_metric_range(name, value, min_range, max_range)
        self.custom_metrics[name] = (value, min_range, max_range)
        self.metadata.last_modified = datetime.now()

    def update_validation_status(self, new_status: str) -> None:
        """
        Update validation status and last_modified timestamp.

        Args:
            new_status (str): New validation status

        Raises:
            ValueError: If status is not one of: unverified, verified, invalid
        """
        if new_status not in {"unverified", "verified", "invalid"}:
            raise ValueError("validation_status must be one of: unverified, verified, invalid")
        self.validation_status = new_status
        self.metadata.last_modified = datetime.now()
