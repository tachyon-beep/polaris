"""
Data models for Contraction Hierarchies algorithm.

This module defines the core data structures used by the Contraction Hierarchies
implementation, including shortcuts and algorithm state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set, Tuple

from polaris.core.enums import RelationType
from polaris.core.models import Edge, EdgeMetadata

# Use CONNECTS_TO temporarily for shortcuts
SHORTCUT_TYPE = RelationType.CONNECTS_TO


@dataclass(frozen=True)
class Shortcut:
    """
    Represents a shortcut edge in the contraction hierarchy.

    Attributes:
        edge: The shortcut edge itself
        via_node: Node that was contracted to create this shortcut
        lower_edge: First edge in the shortcut path
        upper_edge: Second edge in the shortcut path
    """

    edge: Edge
    via_node: str
    lower_edge: Edge
    upper_edge: Edge

    @classmethod
    def create(
        cls,
        from_node: str,
        to_node: str,
        via_node: str,
        lower_edge: Edge,
        upper_edge: Edge,
    ) -> "Shortcut":
        """
        Create a new shortcut edge.

        Args:
            from_node: Source node
            to_node: Target node
            via_node: Contracted node
            lower_edge: First edge in path
            upper_edge: Second edge in path

        Returns:
            New Shortcut instance
        """
        # Calculate shortcut properties
        shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight
        impact_score = min(lower_edge.impact_score, upper_edge.impact_score)
        now = datetime.now()

        # Create shortcut edge
        shortcut_edge = Edge(
            from_entity=from_node,
            to_entity=to_node,
            relation_type=SHORTCUT_TYPE,
            metadata=EdgeMetadata(
                created_at=now,
                last_modified=now,
                confidence=min(
                    lower_edge.metadata.confidence,
                    upper_edge.metadata.confidence,
                ),
                source="contraction_hierarchies",
                weight=shortcut_weight,
            ),
            impact_score=impact_score,
            context=f"Shortcut via {via_node}",
        )

        return cls(
            edge=shortcut_edge,
            via_node=via_node,
            lower_edge=lower_edge,
            upper_edge=upper_edge,
        )


@dataclass
class ContractionState:
    """
    State maintained by the Contraction Hierarchies algorithm.

    Attributes:
        node_level: Map of node ID to contraction level
        shortcuts: Map of (from_node, to_node) to Shortcut
        contracted_neighbors: Map of node ID to set of contracted neighbors
    """

    node_level: Dict[str, int] = field(default_factory=dict)
    shortcuts: Dict[Tuple[str, str], Shortcut] = field(default_factory=dict)
    contracted_neighbors: Dict[str, Set[str]] = field(default_factory=lambda: {})

    def add_shortcut(self, shortcut: Shortcut) -> None:
        """
        Add a shortcut to the state.

        Args:
            shortcut: Shortcut to add
        """
        key = (shortcut.edge.from_entity, shortcut.edge.to_entity)
        self.shortcuts[key] = shortcut

        # Update contracted neighbors
        if shortcut.edge.from_entity not in self.contracted_neighbors:
            self.contracted_neighbors[shortcut.edge.from_entity] = set()
        self.contracted_neighbors[shortcut.edge.from_entity].add(shortcut.edge.to_entity)

    def get_shortcut(self, from_node: str, to_node: str) -> Optional[Shortcut]:
        """
        Get shortcut between two nodes if it exists.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Shortcut if it exists, None otherwise
        """
        return self.shortcuts.get((from_node, to_node))

    def set_node_level(self, node: str, level: int) -> None:
        """
        Set contraction level for a node.

        Args:
            node: Node ID
            level: Contraction level
        """
        self.node_level[node] = level

    def get_node_level(self, node: str) -> int:
        """
        Get contraction level of a node.

        Args:
            node: Node ID

        Returns:
            Contraction level (0 if not contracted)
        """
        return self.node_level.get(node, 0)

    def get_contracted_neighbors(self, node: str) -> Set[str]:
        """
        Get contracted neighbors of a node.

        Args:
            node: Node ID

        Returns:
            Set of contracted neighbor node IDs
        """
        return self.contracted_neighbors.get(node, set())
