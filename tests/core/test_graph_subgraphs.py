"""Tests for subgraph extraction operations."""

from polaris.core.graph_operations.subgraphs import SubgraphExtractor
from polaris.core.models import Edge, EdgeMetadata, Node, NodeMetadata
from polaris.core.enums import RelationType


def test_extract_subgraph(sample_edge_metadata):
    """Test basic subgraph extraction."""
    edges = [
        Edge("A", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("B", "C", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("C", "D", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("D", "E", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
    ]
    extractor = SubgraphExtractor(edges)

    # Extract middle section
    subgraph = extractor.extract_subgraph({"B", "C", "D"})
    assert len(subgraph) == 2  # Should contain B->C and C->D

    # Verify edge endpoints
    edge_pairs = {(e.from_entity, e.to_entity) for e in subgraph}
    assert ("B", "C") in edge_pairs
    assert ("C", "D") in edge_pairs
    assert ("A", "B") not in edge_pairs
    assert ("D", "E") not in edge_pairs


def test_extract_neighborhood(sample_edge_metadata):
    """Test neighborhood extraction."""
    edges = [
        Edge("A", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("B", "C", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("C", "D", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("D", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),  # Cycle
    ]
    extractor = SubgraphExtractor(edges)

    # Extract 1-hop neighborhood of B
    neighborhood = extractor.extract_neighborhood("B", radius=1)
    edge_pairs = {(e.from_entity, e.to_entity) for e in neighborhood}

    assert ("A", "B") in edge_pairs  # Incoming edge
    assert ("B", "C") in edge_pairs  # Outgoing edge
    assert ("D", "B") in edge_pairs  # Cycle edge
    assert ("C", "D") not in edge_pairs  # Beyond 1-hop


def test_extract_path_context(sample_edge_metadata):
    """Test path context extraction."""
    edges = [
        Edge("A", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("B", "C", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("C", "D", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("D", "E", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("X", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("C", "Y", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
    ]
    extractor = SubgraphExtractor(edges)

    # Extract context around path B->C->D
    context = extractor.extract_path_context(["B", "C", "D"], context_size=1)
    edge_pairs = {(e.from_entity, e.to_entity) for e in context}

    # Should include path edges and immediate context
    assert ("B", "C") in edge_pairs
    assert ("C", "D") in edge_pairs
    assert ("X", "B") in edge_pairs
    assert ("C", "Y") in edge_pairs
    assert ("A", "B") in edge_pairs
    assert ("D", "E") in edge_pairs


def test_extract_between(sample_edge_metadata):
    """Test extraction of paths between node sets."""
    edges = [
        Edge("A", "B", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("B", "C", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("C", "D", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("B", "X", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
        Edge("X", "D", RelationType.DEPENDS_ON, sample_edge_metadata, impact_score=0.8),
    ]
    extractor = SubgraphExtractor(edges)

    # Extract all paths from {A,B} to {D}
    between = extractor.extract_between({"A", "B"}, {"D"})
    edge_pairs = {(e.from_entity, e.to_entity) for e in between}

    # Should include both paths to D
    assert ("A", "B") in edge_pairs
    assert ("B", "C") in edge_pairs
    assert ("C", "D") in edge_pairs
    assert ("B", "X") in edge_pairs
    assert ("X", "D") in edge_pairs
