"""
Shared fixtures for Contraction Hierarchies tests.
"""

import pytest
from datetime import datetime

from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType


@pytest.fixture(scope="function")
def base_metadata() -> EdgeMetadata:
    """Create base metadata for test edges."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )


def create_edge(
    from_node: str,
    to_node: str,
    weight: float,
    base_metadata: EdgeMetadata,
) -> Edge:
    """Helper function to create an edge with given parameters."""
    return Edge(
        from_entity=from_node,
        to_entity=to_node,
        relation_type=RelationType.CONNECTS_TO,
        metadata=EdgeMetadata(
            created_at=base_metadata.created_at,
            last_modified=base_metadata.last_modified,
            confidence=base_metadata.confidence,
            source=base_metadata.source,
            weight=weight,
        ),
        impact_score=0.5,
    )


@pytest.fixture(scope="function")
def simple_graph(base_metadata: EdgeMetadata) -> Graph:
    """Create a simple test graph with basic structure."""
    g = Graph(edges=[])

    # Add test edges
    edges = [
        create_edge("A", "B", 1.0, base_metadata),
        create_edge("B", "C", 2.0, base_metadata),
        create_edge("A", "C", 5.0, base_metadata),
    ]

    for edge in edges:
        g.add_edge(edge)

    return g


@pytest.fixture(scope="function")
def complex_graph(base_metadata: EdgeMetadata) -> Graph:
    """Create a complex test graph with cycles and multiple paths."""
    g = Graph(edges=[])

    # Define edges for a complex graph with cycles
    edges_data = [
        ("A", "B", 1.0),
        ("B", "C", 2.0),
        ("C", "D", 1.0),
        ("D", "E", 3.0),
        ("A", "C", 5.0),
        ("B", "D", 2.0),
        ("C", "E", 4.0),
        ("E", "A", 6.0),  # Creates a cycle
        ("D", "B", 2.0),  # Creates another cycle
        ("E", "C", 3.0),  # Creates more complex paths
    ]

    # Add edges to graph
    for from_node, to_node, weight in edges_data:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        g.add_edge(edge)

    return g
