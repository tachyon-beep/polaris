"""
Core domain models for the knowledge graph system.

This module defines the fundamental data structures that represent nodes
and edges in the knowledge graph. It includes:
- Node models for representing vertices in the graph
- Edge models for representing connections between nodes
- Metadata models for tracking node and edge properties
- Metrics models for storing analytical data

The models are implemented using dataclasses for:
- Type safety and validation
- Automatic initialization
- Clear structure definition
- Easy serialization/deserialization
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.validation import validate_dataclass
from .enums import EntityType, RelationType


@validate_dataclass
@dataclass
class NodeMetrics:
    """
    Metrics associated with a node.

    Stores various numerical metrics that describe the characteristics
    and quality attributes of a node.

    Attributes:
        complexity (float): Measure of node's structural complexity (0.0 to 1.0)
        reliability (float): Measure of node's reliability score (0.0 to 1.0)
        coverage (float): Measure of documentation/test coverage (0.0 to 1.0)
        impact (float): Measure of node's system-wide impact (0.0 to 1.0)
        confidence (float): Confidence score in node's correctness (0.0 to 1.0)
        custom_metrics (Dict[str, Tuple[float, float, float]]): Custom metrics with their ranges
            Each tuple contains (value, min_range, max_range)
    """

    complexity: float = 0.0
    reliability: float = 0.0
    coverage: float = 0.0
    impact: float = 0.0
    confidence: float = 0.0
    custom_metrics: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric values after initialization."""
        # Validate standard metrics are in range
        standard_metrics = {
            "complexity": self.complexity,
            "reliability": self.reliability,
            "coverage": self.coverage,
            "impact": self.impact,
            "confidence": self.confidence,
        }

        for name, value in standard_metrics.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        # Validate custom metrics ranges
        for metric_name, (value, min_range, max_range) in self.custom_metrics.items():
            if min_range >= max_range:
                raise ValueError(f"Invalid range for {metric_name}: min must be less than max")
            if not min_range <= value <= max_range:
                raise ValueError(
                    f"Custom metric {metric_name} value {value} must be "
                    f"between {min_range} and {max_range}"
                )


@validate_dataclass
@dataclass
class NodeMetadata:
    """
    Metadata associated with a node.

    Stores administrative and tracking information about a node,
    including temporal data, authorship, and classification.

    Attributes:
        created_at (datetime): Timestamp of node creation
        last_modified (datetime): Timestamp of last modification
        version (int): Node version number
        author (str): Creator or owner of the node
        source (str): Origin or source system of the node
        tags (List[str]): List of classification tags
        status (str): Current status (e.g., "active", "deprecated")
        metrics (NodeMetrics): Associated quality metrics
        custom_attributes (Dict[str, Any]): Additional custom attributes
    """

    created_at: datetime
    last_modified: datetime
    version: int
    author: str
    source: str
    tags: List[str] = field(default_factory=list)
    status: str = "active"
    metrics: NodeMetrics = field(default_factory=NodeMetrics)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.author.strip():
            raise ValueError("author must be a non-empty string")
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if self.last_modified < self.created_at:
            raise ValueError("last modified cannot be before created_at")
        if self.version < 1:
            raise ValueError("version must be a positive integer")


@validate_dataclass
@dataclass
class Node:
    """
    Base node model representing a vertex in the knowledge graph.

    Nodes are the primary vertices in the knowledge graph, representing
    discrete concepts, components, or resources in the system.

    Attributes:
        name (str): Unique identifier for the node
        entity_type (EntityType): Classification of entity
        observations (List[str]): List of recorded observations
        metadata (NodeMetadata): Administrative metadata
        attributes (Dict[str, Any]): Custom attributes
        dependencies (List[str]): Referenced node dependencies
        documentation (Optional[str]): Detailed documentation
        code_reference (Optional[str]): Reference to source code
        data_schema (Optional[Dict]): Data structure definition
        metrics (Optional[Dict[str, float]]): Custom metrics
        validation_rules (Optional[List[str]]): Validation rules
        examples (List[str]): Usage examples
    """

    name: str
    entity_type: EntityType
    observations: List[str]
    metadata: NodeMetadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    documentation: Optional[str] = None
    code_reference: Optional[str] = None
    data_schema: Optional[Dict] = None
    metrics: Optional[Dict[str, float]] = None
    validation_rules: Optional[List[str]] = None
    examples: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate node after initialization."""
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType enum")
        # Validate metrics if present
        if self.metrics is not None:
            if not all(isinstance(v, (int, float)) for v in self.metrics.values()):
                raise ValueError("all metric values must be numeric")


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
        if self.last_modified < self.created_at:
            raise ValueError("last modified cannot be before created_at")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
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
        if not 0 <= self.impact_score <= 1:
            raise ValueError("impact_score must be between 0 and 1")
        if self.validation_status not in {"unverified", "verified", "invalid"}:
            raise ValueError("validation_status must be one of: unverified, verified, invalid")

        # Validate custom metrics ranges
        for metric_name, (value, min_range, max_range) in self.custom_metrics.items():
            if min_range >= max_range:
                raise ValueError(f"Invalid range for {metric_name}: min must be less than max")
            if not min_range <= value <= max_range:
                raise ValueError(
                    f"Custom metric {metric_name} value {value} must "
                    f"be between {min_range} and {max_range}"
                )

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
        if not min_range <= value <= max_range:
            raise ValueError(
                f"Value {value} for {name} must be between {min_range} and {max_range}"
            )
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
