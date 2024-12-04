"""JSON serialization utilities for Polaris types."""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Type, TypeVar

from src.core.enums import EntityType, RelationType
from src.core.models import Edge, EdgeMetadata, Node, NodeMetadata, NodeMetrics

T = TypeVar("T")


class PolarisJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Polaris types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (EntityType, RelationType)):
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, (Node, Edge, NodeMetadata, EdgeMetadata, NodeMetrics)):
            return self.encode_dataclass(o)
        return super().default(o)

    def encode_dataclass(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary."""
        result = {}
        for field in obj.__dataclass_fields__:  # type: ignore
            value = getattr(obj, field)
            if value is not None:
                result[field] = value
        return result


def deserialize_datetime(value: str) -> datetime:
    """Convert ISO format string to datetime."""
    return datetime.fromisoformat(value)


def deserialize_enum(value: str, enum_class: Type[Enum]) -> Enum:
    """Convert string to enum value."""
    return enum_class(value)


def serialize_node(node: Node) -> Dict[str, Any]:
    """Convert Node to JSON-serializable dictionary."""
    return json.loads(json.dumps(node, cls=PolarisJSONEncoder))


def deserialize_node(data: Dict[str, Any]) -> Node:
    """Convert dictionary to Node instance."""
    # Handle entity_type enum
    if isinstance(data.get("entity_type"), str):
        data["entity_type"] = EntityType(data["entity_type"])

    # Handle metadata
    if "metadata" in data:
        meta_data = data["metadata"]
        if "created_at" in meta_data:
            meta_data["created_at"] = deserialize_datetime(meta_data["created_at"])
        if "last_modified" in meta_data:
            meta_data["last_modified"] = deserialize_datetime(meta_data["last_modified"])
        if "metrics" in meta_data:
            meta_data["metrics"] = NodeMetrics(**meta_data["metrics"])
        data["metadata"] = NodeMetadata(**meta_data)

    return Node(**data)


def serialize_edge(edge: Edge) -> Dict[str, Any]:
    """Convert Edge to JSON-serializable dictionary."""
    return json.loads(json.dumps(edge, cls=PolarisJSONEncoder))


def deserialize_edge(data: Dict[str, Any]) -> Edge:
    """Convert dictionary to Edge instance."""
    # Handle relation_type enum
    if isinstance(data.get("relation_type"), str):
        data["relation_type"] = RelationType(data["relation_type"])

    # Handle metadata
    if "metadata" in data:
        meta_data = data["metadata"]
        if "created_at" in meta_data:
            meta_data["created_at"] = deserialize_datetime(meta_data["created_at"])
        if "last_modified" in meta_data:
            meta_data["last_modified"] = deserialize_datetime(meta_data["last_modified"])
        data["metadata"] = EdgeMetadata(**meta_data)

    return Edge(**data)
