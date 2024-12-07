"""
Node models for the knowledge graph system.

This module defines the models representing vertices in the graph,
including nodes and their associated metadata and metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    validate_dataclass,
    validate_date_order,
    validate_metric_range,
    validate_custom_metrics,
)
from ..enums import EntityType


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
            validate_metric_range(name, value)

        # Validate custom metrics ranges
        validate_custom_metrics(self.custom_metrics)


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
        validate_date_order(self.created_at, self.last_modified)
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
