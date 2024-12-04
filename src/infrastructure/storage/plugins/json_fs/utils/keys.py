"""Utilities for generating and managing storage keys."""

from ......core.enums import RelationType


def get_edge_key(from_entity: str, to_entity: str, relation_type: RelationType) -> str:
    """
    Generate a unique key for an edge.

    Args:
        from_entity: Source node name
        to_entity: Target node name
        relation_type: Type of relationship

    Returns:
        Unique string key for edge

    Raises:
        ValueError: If relation type is invalid
    """
    if not isinstance(relation_type, RelationType):
        raise ValueError(f"Invalid relation type: {relation_type}")
    return f"{from_entity}:{relation_type.value}:{to_entity}"
