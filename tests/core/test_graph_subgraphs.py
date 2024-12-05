"""
Tests for graph subgraph extraction functionality.
"""

from datetime import datetime

import pytest

from polaris.core.enums import EntityType, RelationType
from polaris.core.exceptions import NodeNotFoundError
from polaris.core.graph import Graph
from polaris.core.graph_subgraphs import SubgraphExtraction
from polaris.core.models import Edge, EdgeMetadata, Node, NodeMetadata, NodeMetrics


@pytest.fixture
def sample_edge_metadata():
    """Fixture providing sample edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )


@pytest.fixture
def sample_node_metadata():
    """Fixture providing sample node metadata."""
    return NodeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        version=1,
        author="test_author",
        source="test_source",
        metrics=NodeMetrics(),
    )


@pytest.fixture
def sample_graph(sample_edge_metadata, sample_node_metadata):
    """
    Fixture providing a test graph with the following structure:
    A -> B -> D
    |    |
    v    v
    C <- E
    """
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.CALLS,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="E",
            relation_type=RelationType.CALLS,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="E",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
    ]
    nodes = [
        Node(
            name="A",
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="B",
            entity_type=EntityType.CODE_FUNCTION,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="C",
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="D",
            entity_type=EntityType.CODE_FUNCTION,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="E",
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=sample_node_metadata,
        ),
    ]
    return Graph(edges=edges), nodes


def test_extract_neighborhood_radius_0(sample_graph):
    """Test neighborhood extraction with radius 0."""
    graph, _ = sample_graph
    nodes, edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=0)

    assert nodes == {"A"}  # Only the center node
    assert len(edges) == 0  # No edges


def test_extract_neighborhood_radius_1(sample_graph):
    """Test neighborhood extraction with radius 1."""
    graph, _ = sample_graph
    nodes, edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=1)

    assert nodes == {"A", "B", "C"}  # Center node and direct neighbors
    assert len(edges) == 2  # A->B and A->C


def test_extract_neighborhood_radius_2(sample_graph):
    """Test neighborhood extraction with radius 2."""
    graph, _ = sample_graph
    nodes, edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=2)

    assert nodes == {"A", "B", "C", "D", "E"}  # All nodes within 2 steps
    assert len(edges) == 5  # All edges in the graph


def test_extract_neighborhood_with_node_filter(sample_graph):
    """Test neighborhood extraction with node filtering."""
    graph, _ = sample_graph

    def node_filter(node: str) -> bool:
        return node not in {"D", "E"}  # Exclude D and E

    nodes, edges = SubgraphExtraction.extract_neighborhood(
        graph, "A", radius=2, node_filter=node_filter
    )

    assert nodes == {"A", "B", "C"}
    assert all(edge.to_entity not in {"D", "E"} for edge in edges)


def test_extract_neighborhood_with_edge_filter(sample_graph):
    """Test neighborhood extraction with edge filtering."""
    graph, _ = sample_graph

    def edge_filter(edge: Edge) -> bool:
        return edge.relation_type == RelationType.DEPENDS_ON

    _, edges = SubgraphExtraction.extract_neighborhood(
        graph, "A", radius=2, edge_filter=edge_filter
    )

    assert all(edge.relation_type == RelationType.DEPENDS_ON for edge in edges)


def test_extract_by_type_entity_filter(sample_graph):
    """Test subgraph extraction filtered by entity type."""
    graph, nodes = sample_graph

    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph, nodes, entity_types={EntityType.CODE_MODULE.name}
    )

    assert len(filtered_nodes) == 3  # A, C, E are CODE_MODULE
    assert all(node.entity_type == EntityType.CODE_MODULE for node in filtered_nodes)
    # Edges between CODE_MODULE nodes should be included
    assert len(filtered_edges) == 2  # A->C and E->C
    assert {(e.from_entity, e.to_entity) for e in filtered_edges} == {
        ("A", "C"),
        ("E", "C"),
    }


def test_extract_by_type_relation_filter(sample_graph):
    """Test subgraph extraction filtered by relation type."""
    graph, nodes = sample_graph

    _, filtered_edges = SubgraphExtraction.extract_by_type(
        graph, nodes, relation_types={RelationType.CALLS.name}
    )

    assert len(filtered_edges) == 2  # Only CALLS edges
    assert all(edge.relation_type == RelationType.CALLS for edge in filtered_edges)


def test_extract_by_type_both_filters(sample_graph):
    """Test subgraph extraction filtered by both entity and relation types."""
    graph, nodes = sample_graph

    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph,
        nodes,
        entity_types={EntityType.CODE_MODULE.name},
        relation_types={RelationType.CALLS.name},
    )

    assert all(node.entity_type == EntityType.CODE_MODULE for node in filtered_nodes)
    assert all(edge.relation_type == RelationType.CALLS for edge in filtered_edges)
    # Only CALLS edges between CODE_MODULE nodes
    assert len(filtered_edges) == 1  # A->C is a CALLS edge between CODE_MODULE nodes
    assert filtered_edges[0].from_entity == "A"
    assert filtered_edges[0].to_entity == "C"


def test_extract_neighborhood_empty_graph():
    """Test neighborhood extraction on an empty graph."""
    graph = Graph(edges=[])
    with pytest.raises(NodeNotFoundError):
        SubgraphExtraction.extract_neighborhood(graph, "A")


def test_extract_by_type_empty_graph():
    """Test type-based extraction on an empty graph."""
    graph = Graph(edges=[])
    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph, [], entity_types={EntityType.CODE_MODULE.name}
    )

    assert len(filtered_nodes) == 0
    assert len(filtered_edges) == 0


def test_extract_neighborhood_nonexistent_center():
    """Test neighborhood extraction with nonexistent center node."""
    graph = Graph(edges=[])
    with pytest.raises(NodeNotFoundError):
        SubgraphExtraction.extract_neighborhood(graph, "NonExistent")


def test_extract_by_type_no_filters(sample_graph):
    """Test type-based extraction with no filters returns complete graph."""
    graph, nodes = sample_graph
    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(graph, nodes)

    assert len(filtered_nodes) == len(nodes)  # All nodes included
    assert len(filtered_edges) == 5  # All edges in their directed form


def test_neighborhood_edge_key_generation():
    """Test _get_neighborhood_edge_key method."""
    # Test with ordered nodes
    key1 = SubgraphExtraction._get_neighborhood_edge_key("A", "B")
    assert key1 == ("A", "B")

    # Test with reverse order nodes
    key2 = SubgraphExtraction._get_neighborhood_edge_key("B", "A")
    assert key2 == ("A", "B")  # Should be the same as key1

    # Test with same node
    key3 = SubgraphExtraction._get_neighborhood_edge_key("A", "A")
    assert key3 == ("A", "A")


def test_is_valid_neighborhood_edge(sample_edge_metadata):
    """Test _is_valid_neighborhood_edge method."""
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.5,
    )
    nodes = {"A", "B", "C"}

    # Test valid edge
    assert SubgraphExtraction._is_valid_neighborhood_edge(edge, "B", nodes, None)

    # Test edge with nodes in set
    assert SubgraphExtraction._is_valid_neighborhood_edge(edge, "A", nodes, None)

    # Test with edge filter
    def edge_filter(e: Edge) -> bool:
        return e.relation_type == RelationType.CALLS

    assert not SubgraphExtraction._is_valid_neighborhood_edge(edge, "B", nodes, edge_filter)


def test_extract_neighborhood_bidirectional(sample_edge_metadata):
    """Test neighborhood extraction with bidirectional edges."""
    # Create a graph with bidirectional edges: A <-> B <-> C
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
    ]
    graph = Graph(edges=edges)

    # Extract neighborhood from A with radius 1
    nodes, edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=1)
    assert nodes == {"A", "B"}
    assert len(edges) == 2  # Both directions of A-B edge

    # Extract neighborhood from A with radius 2
    nodes, edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=2)
    assert nodes == {"A", "B", "C"}
    assert len(edges) == 4  # All bidirectional edges


def test_edge_key_generation(sample_edge_metadata):
    """Test _get_edge_key method."""
    edge1 = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.5,
    )
    edge2 = Edge(
        from_entity="B",
        to_entity="A",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.5,
    )

    # Test different edges with same relation
    key1 = SubgraphExtraction._get_edge_key(edge1)
    key2 = SubgraphExtraction._get_edge_key(edge2)
    assert key1 != key2  # Directed edges should have different keys

    # Test same edge
    key3 = SubgraphExtraction._get_edge_key(edge1)
    assert key1 == key3  # Same edge should have same key


def test_extract_by_type_invalid_types(sample_graph):
    """Test type-based extraction with invalid types."""
    graph, nodes = sample_graph

    # Test with non-existent entity type
    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph, nodes, entity_types={"INVALID_TYPE"}
    )
    assert len(filtered_nodes) == 0
    assert len(filtered_edges) == 0

    # Test with non-existent relation type
    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph, nodes, relation_types={"INVALID_RELATION"}
    )
    assert len(filtered_nodes) == len(nodes)  # All nodes included
    assert len(filtered_edges) == 0  # No edges match invalid relation


def test_complex_filtering(sample_edge_metadata, sample_node_metadata):
    """Test complex filtering scenarios."""
    # Create a more complex graph with various entity and relation types
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="A",
            relation_type=RelationType.CALLS,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="C",  # Self-loop
            relation_type=RelationType.CALLS,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
    ]
    nodes = [
        Node(
            name="A",
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="B",
            entity_type=EntityType.CODE_FUNCTION,
            observations=[],
            metadata=sample_node_metadata,
        ),
        Node(
            name="C",
            entity_type=EntityType.CODE_MODULE,
            observations=[],
            metadata=sample_node_metadata,
        ),
    ]
    graph = Graph(edges=edges)

    # Test filtering with multiple constraints
    filtered_nodes, filtered_edges = SubgraphExtraction.extract_by_type(
        graph,
        nodes,
        entity_types={EntityType.CODE_MODULE.name},
        relation_types={RelationType.CALLS.name},
    )

    assert len(filtered_nodes) == 2  # A and C
    assert len(filtered_edges) == 1  # Only C->C self-loop
    assert filtered_edges[0].from_entity == "C"
    assert filtered_edges[0].to_entity == "C"


def test_edge_deduplication(sample_edge_metadata):
    """Test edge deduplication behavior."""
    # Create a graph with potential duplicate edges
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="A",
            to_entity="B",  # Duplicate edge
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="A",  # Reverse direction
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
    ]
    graph = Graph(edges=edges)

    # Test neighborhood extraction
    nodes, extracted_edges = SubgraphExtraction.extract_neighborhood(graph, "A", radius=1)
    assert len(nodes) == 2  # A and B
    assert len(extracted_edges) == 2  # A->B and B->A, no duplicates
