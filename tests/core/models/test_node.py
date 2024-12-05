"""
Tests for node models.
"""

from datetime import datetime, timedelta
from typing import cast

import pytest

from polaris.core.enums import EntityType
from polaris.core.models import Node, NodeMetadata, NodeMetrics


@pytest.fixture
def sample_node_metadata():
    """Fixture providing sample node metadata."""
    return NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test.author",
        source="test_source",
        tags=["test"],
        status="active",
        metrics=NodeMetrics(
            complexity=0.5, reliability=0.8, coverage=0.7, impact=0.6, confidence=0.9
        ),
    )


def test_node_metrics_creation():
    """Test creation of node metrics with valid values."""
    metrics = NodeMetrics(complexity=0.5, reliability=0.8, coverage=0.7, impact=0.6, confidence=0.9)

    assert metrics.complexity == pytest.approx(0.5)
    assert metrics.reliability == pytest.approx(0.8)
    assert metrics.coverage == pytest.approx(0.7)
    assert metrics.impact == pytest.approx(0.6)
    assert metrics.confidence == pytest.approx(0.9)


def test_node_creation(sample_node_metadata):
    """Test basic node creation and properties."""
    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=["Test observation"],
        metadata=sample_node_metadata,
        documentation="Test documentation",
    )

    assert node.name == "test_node"
    assert node.entity_type == EntityType.CODE_MODULE
    assert len(node.observations) == 1
    assert node.observations[0] == "Test observation"
    assert node.documentation == "Test documentation"
    assert node.metadata.author == "test.author"
    assert node.metadata.metrics.complexity == pytest.approx(0.5)


def test_node_metrics_validation():
    """Test validation of node metrics with invalid values."""
    with pytest.raises(ValueError):
        NodeMetrics(
            complexity=1.5,  # Should be between 0 and 1
            reliability=0.8,
            coverage=0.7,
            impact=0.6,
            confidence=0.9,
        )


def test_node_with_custom_attributes(sample_node_metadata):
    """Test node creation with custom attributes."""
    custom_attributes = {"priority": "high", "category": "service", "team": "backend"}

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=["Test observation"],
        metadata=sample_node_metadata,
        attributes=custom_attributes,
    )

    assert node.attributes["priority"] == "high"
    assert node.attributes["category"] == "service"
    assert node.attributes["team"] == "backend"


def test_node_metadata_timestamp_validation():
    """Test validation of node metadata timestamps."""
    now = datetime.now()
    earlier = now - timedelta(days=1)

    # Valid case: created_at before last_modified
    metadata = NodeMetadata(
        created_at=earlier,
        last_modified=now,
        version=1,
        author="test.author",
        source="test_source",
    )
    assert metadata.created_at == earlier
    assert metadata.last_modified == now

    # Invalid case: last_modified before created_at
    with pytest.raises(ValueError):
        NodeMetadata(
            created_at=now,
            last_modified=earlier,
            version=1,
            author="test.author",
            source="test_source",
        )


def test_node_metadata_version_validation():
    """Test validation of node metadata version."""
    now = datetime.now()

    # Invalid case: version less than 1
    with pytest.raises(ValueError):
        NodeMetadata(
            created_at=now,
            last_modified=now,
            version=0,  # Invalid version
            author="test.author",
            source="test_source",
        )

    # Invalid case: non-integer version
    with pytest.raises(TypeError):
        NodeMetadata(
            created_at=now,
            last_modified=now,
            version=cast(int, 1.5),  # Type error with float
            author="test.author",
            source="test_source",
        )


def test_node_with_empty_collections(sample_node_metadata):
    """Test node creation with empty collections."""
    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=[],  # Empty observations
        metadata=sample_node_metadata,
        dependencies=[],  # Empty dependencies
        validation_rules=[],  # Empty validation rules
        examples=[],  # Empty examples
        attributes={},  # Empty attributes
    )

    assert isinstance(node.validation_rules, list)
    assert isinstance(node.examples, list)
    assert len(node.observations) == 0
    assert len(node.dependencies) == 0
    assert len(node.validation_rules) == 0
    assert len(node.examples) == 0
    assert len(node.attributes) == 0


def test_node_with_special_characters(sample_node_metadata):
    """Test node creation with special characters in fields."""
    node = Node(
        name="test-node@2.0",  # Special characters in name
        entity_type=EntityType.CODE_MODULE,
        observations=["Test observation!@#$%"],  # Special characters in observation
        metadata=sample_node_metadata,
        documentation="Documentation with üñîçødé",  # Unicode characters
        code_reference="/path/to/file-v2.0.py",  # Special characters in reference
    )

    assert node.name == "test-node@2.0"
    assert node.observations[0] == "Test observation!@#$%"
    assert node.documentation == "Documentation with üñîçødé"
    assert node.code_reference == "/path/to/file-v2.0.py"


def test_node_with_data_schema(sample_node_metadata):
    """Test node with data schema validation."""
    data_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
        },
        "required": ["id", "name"],
    }

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=[],
        metadata=sample_node_metadata,
        data_schema=data_schema,
    )

    assert node.data_schema is not None
    assert node.data_schema["type"] == "object"
    assert "id" in node.data_schema["properties"]
    assert "name" in node.data_schema["properties"]


def test_node_with_dependencies(sample_node_metadata):
    """Test node with dependencies."""
    dependencies = ["dep1", "dep2", "dep3"]

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=[],
        metadata=sample_node_metadata,
        dependencies=dependencies,
    )

    assert isinstance(node.dependencies, list)
    assert len(node.dependencies) == 3
    assert all(dep in node.dependencies for dep in dependencies)


def test_node_with_validation_rules(sample_node_metadata):
    """Test node with validation rules."""
    validation_rules = [
        "must_have_documentation",
        "must_have_tests",
        "must_follow_naming_convention",
    ]

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=[],
        metadata=sample_node_metadata,
        validation_rules=validation_rules,
    )

    assert isinstance(node.validation_rules, list)
    assert len(node.validation_rules) == 3
    assert all(rule in node.validation_rules for rule in validation_rules)


def test_node_metrics_with_boundary_values():
    """Test node metrics with boundary values."""
    # Test minimum valid values
    metrics = NodeMetrics(
        complexity=0.0,
        reliability=0.0,
        coverage=0.0,
        impact=0.0,
        confidence=0.0,
    )
    standard_metrics = {"complexity", "reliability", "coverage", "impact", "confidence"}
    assert all(getattr(metrics, field) == pytest.approx(0.0) for field in standard_metrics)

    # Test maximum valid values
    metrics = NodeMetrics(
        complexity=1.0,
        reliability=1.0,
        coverage=1.0,
        impact=1.0,
        confidence=1.0,
    )
    assert all(getattr(metrics, field) == pytest.approx(1.0) for field in standard_metrics)
