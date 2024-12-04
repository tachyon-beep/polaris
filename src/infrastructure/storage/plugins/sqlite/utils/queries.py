"""Utilities for building SQL queries."""

from typing import Any, Dict, List, Tuple

from ......core.enums import EntityType, RelationType


def build_node_filter_query(filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build SQL query and parameters for node filters.

    Args:
        filters: Dictionary of filter criteria

    Returns:
        Tuple of (query string, parameter list)
    """
    if not filters:
        return "SELECT * FROM nodes", []

    conditions = []
    params = []
    for key, value in filters.items():
        if key == "entity_type":
            conditions.append("entity_type = ?")
            params.append(value.value if isinstance(value, EntityType) else value)
        # Add more filter conditions as needed

    if conditions:
        where_clause = " AND ".join(conditions)
        return f"SELECT * FROM nodes WHERE {where_clause}", params
    return "SELECT * FROM nodes", []


def build_edge_filter_query(filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build SQL query and parameters for edge filters.

    Args:
        filters: Dictionary of filter criteria

    Returns:
        Tuple of (query string, parameter list)
    """
    if not filters:
        return "SELECT * FROM edges", []

    conditions = []
    params = []
    for key, value in filters.items():
        if key == "relation_type":
            conditions.append("relation_type = ?")
            params.append(value.value if isinstance(value, RelationType) else value)
        elif key in ["from_node", "to_node"]:
            conditions.append(f"{key} = ?")
            params.append(value)
        # Add more filter conditions as needed

    if conditions:
        where_clause = " AND ".join(conditions)
        return f"SELECT * FROM edges WHERE {where_clause}", params
    return "SELECT * FROM edges", []


def add_pagination(query: str, limit: int, offset: int) -> Tuple[str, List[Any]]:
    """
    Add pagination to a SQL query.

    Args:
        query: Base SQL query
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Tuple of (query string with pagination, parameter list)
    """
    return f"{query} LIMIT ? OFFSET ?", [limit, offset]
