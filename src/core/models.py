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

from .enums import EntityType, RelationType


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

    Example:
        >>> metrics = NodeMetrics(
        ...     complexity=0.7,
        ...     reliability=0.9,
        ...     coverage=0.8,
        ...     impact=0.6,
        ...     confidence=0.95,
        ...     custom_metrics={"performance": (0.85, 0.0, 1.0)}
        ... )
    """

    complexity: float = 0.0
    reliability: float = 0.0
    coverage: float = 0.0
    impact: float = 0.0
    confidence: float = 0.0
    custom_metrics: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric values after initialization."""
        # Validate standard metrics
        standard_metrics = {
            "complexity": self.complexity,
            "reliability": self.reliability,
            "coverage": self.coverage,
            "impact": self.impact,
            "confidence": self.confidence,
        }

        for name, value in standard_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a numeric value")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        # Validate custom metrics
        for metric_name, (value, min_range, max_range) in self.custom_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Custom metric {metric_name} must be a numeric value")
            if not isinstance(min_range, (int, float)) or not isinstance(max_range, (int, float)):
                raise ValueError(f"Range values for {metric_name} must be numeric")
            if min_range >= max_range:
                raise ValueError(f"Invalid range for {metric_name}: min must be less than max")
            if not min_range <= value <= max_range:
                raise ValueError(
                    f"Custom metric {metric_name} value {value} must be "
                    "between {min_range} and {max_range}"
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

    Example:
        >>> metadata = NodeMetadata(
        ...     created_at=datetime.now(),
        ...     last_modified=datetime.now(),
        ...     version=1,
        ...     author="john.doe",
        ...     source="code_analysis",
        ...     tags=["core", "critical"],
        ...     status="active",
        ...     custom_attributes={"priority": "high"}
        ... )
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
        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime")
        if not isinstance(self.last_modified, datetime):
            raise ValueError("last_modified must be a datetime")
        if self.last_modified < self.created_at:
            raise ValueError("last modified cannot be before created_at")
        if not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.author, str) or not self.author.strip():
            raise ValueError("author must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.tags, list):
            raise TypeError("tags must be a list")
        if not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("all tags must be strings")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        if not isinstance(self.custom_attributes, dict):
            raise TypeError("custom_attributes must be a dictionary")

    def add_tag(self, tag: str) -> None:
        """
        Add a tag if it doesn't already exist.

        Args:
            tag (str): Tag to add

        Raises:
            ValueError: If tag is not a non-empty string
        """
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("tag must be a non-empty string")
        if tag not in self.tags:
            self.tags.append(tag)

    def update_status(self, new_status: str) -> None:
        """
        Update the status and last_modified timestamp.

        Args:
            new_status (str): New status value

        Raises:
            ValueError: If status is not a non-empty string
        """
        if not isinstance(new_status, str) or not new_status.strip():
            raise ValueError("status must be a non-empty string")
        self.status = new_status
        self.last_modified = datetime.now()


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

    Example:
        >>> node = Node(
        ...     name="UserService",
        ...     entity_type=EntityType.CODE_CLASS,
        ...     observations=["Handles user authentication"],
        ...     metadata=NodeMetadata(...),
        ...     documentation="Service for user management",
        ...     code_reference="src/services/user.py"
        ... )
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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType enum")
        if not isinstance(self.observations, list):
            raise TypeError("observations must be a list")
        if not all(isinstance(obs, str) for obs in self.observations):
            raise ValueError("all observations must be strings")
        if self.documentation is not None and not isinstance(self.documentation, str):
            raise TypeError("documentation must be a string or None")
        if self.code_reference is not None and not isinstance(self.code_reference, str):
            raise TypeError("code_reference must be a string or None")
        if self.data_schema is not None and not isinstance(self.data_schema, dict):
            raise TypeError("data_schema must be a dictionary or None")
        if self.metrics is not None:
            if not isinstance(self.metrics, dict):
                raise TypeError("metrics must be a dictionary or None")
            if not all(isinstance(v, (int, float)) for v in self.metrics.values()):
                raise ValueError("all metric values must be numeric")
        if self.validation_rules is not None:
            if not isinstance(self.validation_rules, list):
                raise TypeError("validation_rules must be a list or None")
            if not all(isinstance(rule, str) for rule in self.validation_rules):
                raise ValueError("all validation rules must be strings")
        if not isinstance(self.examples, list):
            raise TypeError("examples must be a list")
        if not all(isinstance(example, str) for example in self.examples):
            raise ValueError("all examples must be strings")

    def add_observation(self, observation: str) -> None:
        """
        Add an observation if it doesn't already exist.

        Args:
            observation (str): Observation to add

        Raises:
            ValueError: If observation is not a non-empty string
        """
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("observation must be a non-empty string")
        if observation not in self.observations:
            self.observations.append(observation)
            self.metadata.last_modified = datetime.now()

    def add_dependency(self, dependency: str) -> None:
        """
        Add a dependency if it doesn't already exist.

        Args:
            dependency (str): Dependency node name to add

        Raises:
            ValueError: If dependency is not a non-empty string
        """
        if not isinstance(dependency, str) or not dependency.strip():
            raise ValueError("dependency must be a non-empty string")
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)
            self.metadata.last_modified = datetime.now()


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

    Example:
        >>> metadata = EdgeMetadata(
        ...     created_at=datetime.now(),
        ...     last_modified=datetime.now(),
        ...     confidence=0.95,
        ...     source="static_analysis",
        ...     bidirectional=True,
        ...     weight=0.8,
        ...     custom_attributes={"priority": "high"}
        ... )
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
        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime")
        if not isinstance(self.last_modified, datetime):
            raise ValueError("last_modified must be a datetime")
        if self.last_modified < self.created_at:
            raise ValueError("last modified cannot be before created_at")
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a numeric value")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be a boolean")
        if not isinstance(self.temporal, bool):
            raise TypeError("temporal must be a boolean")
        if not isinstance(self.weight, (int, float)):
            raise TypeError("weight must be a numeric value")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if not isinstance(self.custom_attributes, dict):
            raise TypeError("custom_attributes must be a dictionary")

    def update_confidence(self, new_confidence: float) -> None:
        """
        Update confidence score and last_modified timestamp.

        Args:
            new_confidence (float): New confidence score

        Raises:
            ValueError: If confidence is not between 0 and 1
        """
        if not isinstance(new_confidence, (int, float)):
            raise ValueError("confidence must be a numeric value")
        if not 0 <= new_confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = new_confidence
        self.last_modified = datetime.now()


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

    Example:
        >>> edge = Edge(
        ...     from_entity="UserService",
        ...     to_entity="DatabaseService",
        ...     relation_type=RelationType.DEPENDS_ON,
        ...     metadata=EdgeMetadata(...),
        ...     impact_score=0.8,
        ...     context="User data storage",
        ...     custom_metrics={"latency": (0.95, 0.0, 2.0)}
        ... )

    Note:
        - Edges can be directional or bidirectional
        - Impact score ranges from 0.0 to 1.0
        - Validation status can be: "unverified", "verified", "invalid"
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
        if not isinstance(self.from_entity, str) or not self.from_entity.strip():
            raise ValueError("source node must be a non-empty string")
        if not isinstance(self.to_entity, str) or not self.to_entity.strip():
            raise ValueError("target node must be a non-empty string")
        if not isinstance(self.relation_type, RelationType):
            raise TypeError("relation_type must be a RelationType enum")
        if not isinstance(self.impact_score, (int, float)):
            raise ValueError("impact_score must be a numeric value")
        if not 0 <= self.impact_score <= 1:
            raise ValueError("impact_score must be between 0 and 1")
        if self.context is not None and not isinstance(self.context, str):
            raise TypeError("context must be a string or None")
        if not isinstance(self.validation_status, str) or not self.validation_status.strip():
            raise ValueError("validation_status must be a non-empty string")
        if self.validation_status not in {"unverified", "verified", "invalid"}:
            raise ValueError("validation_status must be one of: unverified, verified, invalid")

        # Validate custom metrics
        for metric_name, (value, min_range, max_range) in self.custom_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Custom metric {metric_name} must be a numeric value")
            if not isinstance(min_range, (int, float)) or not isinstance(max_range, (int, float)):
                raise ValueError(f"Range values for {metric_name} must be numeric")
            if min_range >= max_range:
                raise ValueError(f"Invalid range for {metric_name}: min must be less than max")
            if not min_range <= value <= max_range:
                raise ValueError(
                    f"Custom metric {metric_name} value {value} must "
                    "be between {min_range} and {max_range}"
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
