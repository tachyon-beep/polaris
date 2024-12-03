"""
Tests for custom exceptions and validation.
"""

from datetime import datetime
from typing import Any, cast

import pytest

from src.core.enums import EntityType, RelationType
from src.core.exceptions import GraphOperationError, ValidationError
from src.core.graph import Graph
from src.core.models import Edge, EdgeMetadata, Node, NodeMetadata, NodeMetrics


@pytest.fixture
def valid_node_metadata():
    """Fixture providing valid node metadata."""
    return NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test.author",
        source="test_source",
        metrics=NodeMetrics(
            complexity=0.0,
            reliability=0.0,
            coverage=0.0,
            impact=0.0,
            confidence=0.0,
        ),
    )


@pytest.fixture
def valid_edge_metadata():
    """Fixture providing valid edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )


def test_validation_error_message():
    """Test validation error message formatting."""
    error = ValidationError("test message")
    assert str(error) == "Validation Error: test message"


def test_graph_operation_error_message():
    """Test graph operation error message formatting."""
    error = GraphOperationError("test message")
    assert str(error) == "Graph Operation Error: test message"


def test_node_validation_empty_name(valid_node_metadata):
    """Test node validation with empty name."""
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        Node(
            name="",  # Invalid: empty name
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=valid_node_metadata,
        )


def test_node_validation_invalid_type(valid_node_metadata):
    """Test node validation with invalid entity type."""
    with pytest.raises(TypeError, match="entity_type must be an EntityType enum"):
        Node(
            name="test_node",
            entity_type=cast(EntityType, "invalid_type"),  # Invalid: not an enum
            observations=[],
            metadata=valid_node_metadata,
        )


def test_edge_validation_empty_source(valid_edge_metadata):
    """Test edge validation with empty source node."""
    with pytest.raises(ValueError, match="source node must be a non-empty string"):
        Edge(
            from_entity="",  # Invalid: empty source
            to_entity="node2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=valid_edge_metadata,
            impact_score=0.8,
        )


def test_edge_validation_empty_target(valid_edge_metadata):
    """Test edge validation with empty target node."""
    with pytest.raises(ValueError, match="target node must be a non-empty string"):
        Edge(
            from_entity="node1",
            to_entity="",  # Invalid: empty target
            relation_type=RelationType.DEPENDS_ON,
            metadata=valid_edge_metadata,
            impact_score=0.8,
        )


def test_edge_validation_invalid_type(valid_edge_metadata):
    """Test edge validation with invalid relation type."""
    with pytest.raises(TypeError, match="relation_type must be a RelationType enum"):
        Edge(
            from_entity="node1",
            to_entity="node2",
            relation_type=cast(RelationType, "invalid_type"),  # Invalid: not an enum
            metadata=valid_edge_metadata,
            impact_score=0.8,
        )


def test_edge_validation_invalid_impact(valid_edge_metadata):
    """Test edge validation with invalid impact score."""
    with pytest.raises(ValueError, match="impact_score must be between 0 and 1"):
        Edge(
            from_entity="node1",
            to_entity="node2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=valid_edge_metadata,
            impact_score=1.5,  # Invalid: greater than 1
        )


def test_multiple_validation_errors(valid_node_metadata):
    """Test multiple validation errors in a single object."""
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        Node(
            name="",  # Invalid: empty name
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=valid_node_metadata,
            metrics={"complexity": float("inf")},  # Invalid: infinite value
            attributes={"": "invalid"},  # Invalid: empty key
        )


def test_edge_empty_nodes_validation(valid_edge_metadata):
    """Test validation of edge with empty node references."""
    with pytest.raises(ValueError, match="source node must be a non-empty string"):
        Edge(
            from_entity="",  # Invalid: empty source node
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=valid_edge_metadata,
            impact_score=0.8,
        )


def test_node_metadata_validation():
    """Test validation of node metadata."""
    now = datetime.now()

    with pytest.raises(ValueError, match="author must be a non-empty string"):
        NodeMetadata(
            created_at=now,
            last_modified=now,
            version=1,
            author="",  # Invalid: empty author
            source="test_source",
        )


def test_graph_operation_error(valid_edge_metadata):
    """Test graph operation errors."""
    graph = Graph(
        edges=[
            Edge(
                from_entity="A",
                to_entity="B",
                relation_type=RelationType.DEPENDS_ON,
                metadata=valid_edge_metadata,
                impact_score=0.8,
            )
        ]
    )

    # Test operation on non-existent node
    edge = graph.get_edge("NonExistent", "B")
    assert edge is None  # Should return None for non-existent nodes


def test_validation_error_inheritance():
    """Test that ValidationError inherits from Exception."""
    error = ValidationError("test")
    assert isinstance(error, Exception)


def test_graph_operation_error_inheritance():
    """Test that GraphOperationError inherits from Exception."""
    error = GraphOperationError("test")
    assert isinstance(error, Exception)
