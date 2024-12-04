"""Utilities for converting between database rows and model objects."""

import json
from dataclasses import asdict
from typing import Any, Tuple

from ......core.enums import EntityType, RelationType
from ......core.models import Edge, EdgeMetadata, Node, NodeMetadata
from ...utils import deserialize_datetime
from ..utils.persistence import row_to_tuple


def node_to_row(node: Node) -> Tuple:
    """
    Convert node to database row.

    Args:
        node: Node to convert

    Returns:
        Tuple of values matching table schema
    """
    return (
        node.name,
        node.entity_type.value,
        json.dumps(node.observations),
        json.dumps(node.attributes),
        json.dumps(asdict(node.metadata)),
        json.dumps(node.dependencies),
        node.documentation,
        node.code_reference,
        json.dumps(node.data_schema) if node.data_schema else None,
        json.dumps(node.metrics) if node.metrics else None,
        json.dumps(node.validation_rules) if node.validation_rules else None,
        json.dumps(node.examples),
    )


def row_to_node(row: Any) -> Node:
    """
    Convert database row to node.

    Args:
        row: Database row (tuple or Row object)

    Returns:
        Node instance
    """
    row_tuple = row_to_tuple(row)
    (
        name,
        entity_type,
        observations,
        attributes,
        metadata,
        dependencies,
        documentation,
        code_reference,
        data_schema,
        metrics,
        validation_rules,
        examples,
    ) = row_tuple
    return Node(
        name=name,
        entity_type=EntityType(entity_type),
        observations=json.loads(observations),
        metadata=NodeMetadata(**deserialize_datetime(json.loads(metadata))),
        attributes=json.loads(attributes),
        dependencies=json.loads(dependencies),
        documentation=documentation,
        code_reference=code_reference,
        data_schema=json.loads(data_schema) if data_schema else None,
        metrics=json.loads(metrics) if metrics else None,
        validation_rules=json.loads(validation_rules) if validation_rules else None,
        examples=json.loads(examples),
    )


def edge_to_row(edge: Edge) -> Tuple:
    """
    Convert edge to database row.

    Args:
        edge: Edge to convert

    Returns:
        Tuple of values matching table schema
    """
    return (
        edge.from_entity,
        edge.to_entity,
        edge.relation_type.value,
        json.dumps(edge.attributes),
        json.dumps(asdict(edge.metadata)),
        edge.context,
        edge.impact_score,
        edge.validation_status,
    )


def row_to_edge(row: Any) -> Edge:
    """
    Convert database row to edge.

    Args:
        row: Database row (tuple or Row object)

    Returns:
        Edge instance
    """
    row_tuple = row_to_tuple(row)
    (
        from_node,
        to_node,
        relation_type,
        attributes,
        metadata,
        context,
        impact_score,
        validation_status,
    ) = row_tuple
    return Edge(
        from_entity=from_node,
        to_entity=to_node,
        relation_type=RelationType(relation_type),
        metadata=EdgeMetadata(**deserialize_datetime(json.loads(metadata))),
        attributes=json.loads(attributes),
        context=context,
        impact_score=impact_score,
        validation_status=validation_status,
    )
