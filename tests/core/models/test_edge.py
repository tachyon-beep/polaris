"""
Tests for edge models.
"""

from datetime import datetime
from typing import cast

import pytest

from polaris.core.enums import RelationType
from polaris.core.models import Edge, EdgeMetadata


@pytest.fixture
def sample_edge_metadata():
    """Fixture providing sample edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
        bidirectional=False,
        temporal=False,
        weight=1.0,
    )


def test_edge_creation(sample_edge_metadata):
    """Test basic edge creation and properties."""
    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        context="Test context",
        impact_score=0.8,
    )

    assert edge.from_entity == "node1"
    assert edge.to_entity == "node2"
    assert edge.relation_type == RelationType.DEPENDS_ON
    assert edge.context == "Test context"
    assert edge.impact_score == pytest.approx(0.8)
    assert edge.metadata.confidence == pytest.approx(0.9)
    assert not edge.metadata.bidirectional
    assert edge.metadata.weight == pytest.approx(1.0)


def test_edge_metadata_validation():
    """Test validation of edge metadata."""
    with pytest.raises(ValueError):
        EdgeMetadata(
            created_at=datetime.now(),
            last_modified=datetime.now(),
            confidence=1.5,  # Should be between 0 and 1
            source="test_source",
        )


def test_edge_with_custom_attributes(sample_edge_metadata):
    """Test edge creation with custom attributes."""
    custom_attributes = {"strength": "strong", "reviewed": True, "reviewer": "john.doe"}

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        attributes=custom_attributes,
        impact_score=0.8,
    )

    assert edge.attributes["strength"] == "strong"
    assert edge.attributes["reviewed"] is True
    assert edge.attributes["reviewer"] == "john.doe"


def test_edge_validation():
    """Test edge validation."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )

    # Test empty source node
    with pytest.raises(ValueError):
        Edge(
            from_entity="",  # Empty source
            to_entity="node2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=metadata,
            impact_score=0.8,
        )

    # Test empty target node
    with pytest.raises(ValueError):
        Edge(
            from_entity="node1",
            to_entity="",  # Empty target
            relation_type=RelationType.DEPENDS_ON,
            metadata=metadata,
            impact_score=0.8,
        )

    # Test invalid impact score
    with pytest.raises(ValueError):
        Edge(
            from_entity="node1",
            to_entity="node2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=metadata,
            impact_score=1.5,  # Invalid score
        )


def test_edge_metadata_weight_validation():
    """Test edge metadata weight validation."""
    now = datetime.now()

    # Test negative weight
    with pytest.raises(ValueError):
        EdgeMetadata(
            created_at=now,
            last_modified=now,
            confidence=0.9,
            source="test_source",
            weight=-1.0,  # Invalid negative weight
        )

    # Test non-numeric weight
    with pytest.raises(TypeError):
        EdgeMetadata(
            created_at=now,
            last_modified=now,
            confidence=0.9,
            source="test_source",
            weight=cast(float, "high"),  # Type error with string
        )


def test_edge_with_boundary_values(sample_edge_metadata):
    """Test edge creation with boundary values."""
    # Test minimum valid values
    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.0,  # Minimum valid impact score
    )
    assert edge.impact_score == pytest.approx(0.0)

    # Test maximum valid values
    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=1.0,  # Maximum valid impact score
    )
    assert edge.impact_score == pytest.approx(1.0)


def test_edge_custom_metrics_management():
    """Test edge custom metrics management."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=metadata,
        impact_score=0.8,
    )

    # Add valid custom metric
    edge.add_custom_metric("test_metric", 0.5)
    assert "test_metric" in edge.custom_metrics
    assert edge.custom_metrics["test_metric"][0] == 0.5

    # Add custom metric with custom range
    edge.add_custom_metric("custom_range", 5.0, 0.0, 10.0)
    assert "custom_range" in edge.custom_metrics
    assert edge.custom_metrics["custom_range"][0] == 5.0

    # Test invalid metric value type
    with pytest.raises(ValueError):
        edge.add_custom_metric("invalid_type", "not_a_number")

    # Test value outside custom range
    with pytest.raises(ValueError):
        edge.add_custom_metric("out_of_range", 15.0, 0.0, 10.0)

    # Verify last_modified was updated
    assert edge.metadata.last_modified > metadata.created_at


def test_edge_validation_status_management():
    """Test edge validation status management."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.DEPENDS_ON,
        metadata=metadata,
        impact_score=0.8,
    )

    # Test valid status updates
    valid_statuses = ["unverified", "verified", "invalid"]
    for status in valid_statuses:
        edge.update_validation_status(status)
        assert edge.validation_status == status
        # Verify last_modified was updated
        assert edge.metadata.last_modified > metadata.created_at

    # Test invalid status
    with pytest.raises(ValueError):
        edge.update_validation_status("unknown_status")

    # Test empty status
    with pytest.raises(ValueError):
        edge.update_validation_status("")
