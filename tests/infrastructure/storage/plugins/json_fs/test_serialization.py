"""Tests for JSON serialization utilities."""

import json
from datetime import datetime

import pytest

from src.core.enums import EntityType, RelationType
from src.core.models import Edge, EdgeMetadata, Node, NodeMetadata
from src.infrastructure.storage.plugins.json_fs.utils.serialization import (
    PolarisJSONEncoder,
    deserialize_datetime,
    deserialize_edge,
    deserialize_enum,
    deserialize_node,
    serialize_edge,
    serialize_node,
)


def test_polaris_json_encoder_enum():
    """Test JSON encoding of enum values."""
    data = {"type": EntityType.CODE_MODULE}
    encoded = json.dumps(data, cls=PolarisJSONEncoder)
    assert json.loads(encoded)["type"] == "code_module"


def test_polaris_json_encoder_datetime():
    """Test JSON encoding of datetime values."""
    now = datetime.now()
    data = {"timestamp": now}
    encoded = json.dumps(data, cls=PolarisJSONEncoder)
    assert json.loads(encoded)["timestamp"] == now.isoformat()


def test_deserialize_datetime():
    """Test datetime deserialization."""
    now = datetime.now()
    iso_str = now.isoformat()
    deserialized = deserialize_datetime(iso_str)
    assert deserialized == now


def test_deserialize_enum():
    """Test enum deserialization."""
    value = "code_module"
    deserialized = deserialize_enum(value, EntityType)
    assert deserialized == EntityType.CODE_MODULE


def test_node_serialization():
    """Test complete node serialization/deserialization cycle."""
    metadata = NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test_author",
        source="test_source",
    )

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=["test observation"],
        metadata=metadata,
        dependencies=["dep1", "dep2"],
    )

    serialized = serialize_node(node)
    deserialized = deserialize_node(serialized)

    assert deserialized.name == node.name
    assert deserialized.entity_type == node.entity_type
    assert deserialized.observations == node.observations
    assert deserialized.dependencies == node.dependencies
    assert deserialized.metadata.created_at == node.metadata.created_at
    assert deserialized.metadata.last_modified == node.metadata.last_modified
    assert deserialized.metadata.version == node.metadata.version
    assert deserialized.metadata.author == node.metadata.author
    assert deserialized.metadata.source == node.metadata.source


def test_edge_serialization():
    """Test complete edge serialization/deserialization cycle."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.8,
        source="test_source",
    )

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.RELATES_TO,
        metadata=metadata,
        impact_score=0.8,
    )

    serialized = serialize_edge(edge)
    deserialized = deserialize_edge(serialized)

    assert deserialized.from_entity == edge.from_entity
    assert deserialized.to_entity == edge.to_entity
    assert deserialized.relation_type == edge.relation_type
    assert deserialized.impact_score == edge.impact_score
    assert deserialized.metadata.created_at == edge.metadata.created_at
    assert deserialized.metadata.last_modified == edge.metadata.last_modified
    assert deserialized.metadata.confidence == edge.metadata.confidence
    assert deserialized.metadata.source == edge.metadata.source


def test_node_serialization_without_optional_fields():
    """Test node serialization when optional fields are missing."""
    metadata = NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test_author",
        source="test_source",
    )

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=["test observation"],
        metadata=metadata,
    )

    serialized = serialize_node(node)
    deserialized = deserialize_node(serialized)

    assert deserialized.name == node.name
    assert deserialized.entity_type == node.entity_type
    assert deserialized.observations == node.observations


def test_edge_serialization_without_optional_fields():
    """Test edge serialization when optional fields are missing."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.8,
        source="test_source",
    )

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.RELATES_TO,
        metadata=metadata,
        impact_score=0.8,
    )

    serialized = serialize_edge(edge)
    deserialized = deserialize_edge(serialized)

    assert deserialized.from_entity == edge.from_entity
    assert deserialized.to_entity == edge.to_entity
    assert deserialized.relation_type == edge.relation_type
    assert deserialized.impact_score == edge.impact_score


def test_invalid_enum_value():
    """Test handling of invalid enum values."""
    with pytest.raises(ValueError):
        deserialize_enum("INVALID", EntityType)


def test_invalid_datetime():
    """Test handling of invalid datetime strings."""
    with pytest.raises(ValueError):
        deserialize_datetime("invalid-datetime")


def test_node_serialization_with_empty_fields():
    """Test node serialization with empty list fields."""
    metadata = NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test_author",
        source="test_source",
    )

    node = Node(
        name="test_node",
        entity_type=EntityType.CODE_MODULE,
        observations=[],
        metadata=metadata,
        dependencies=[],
    )

    serialized = serialize_node(node)
    deserialized = deserialize_node(serialized)

    assert deserialized.name == node.name
    assert deserialized.entity_type == node.entity_type
    assert deserialized.observations == []
    assert deserialized.dependencies == []


def test_edge_serialization_with_null_fields():
    """Test edge serialization with null optional fields."""
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.8,
        source="test_source",
    )

    edge = Edge(
        from_entity="node1",
        to_entity="node2",
        relation_type=RelationType.RELATES_TO,
        metadata=metadata,
        impact_score=0.8,
        context=None,
    )

    serialized = serialize_edge(edge)
    deserialized = deserialize_edge(serialized)

    assert deserialized.from_entity == edge.from_entity
    assert deserialized.to_entity == edge.to_entity
    assert deserialized.relation_type == edge.relation_type
    assert deserialized.impact_score == edge.impact_score
    assert deserialized.context == edge.context
