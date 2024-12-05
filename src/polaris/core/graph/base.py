"""
Core graph data structure with efficient adjacency list representation.

This module provides the fundamental BaseGraph class that represents the knowledge graph
structure using an adjacency list representation. The graph is directed and supports
efficient neighbor lookups and edge queries between nodes.

The implementation is pure, focusing only on graph operations without side concerns
like events, caching, or transactions which are handled by separate modules.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

from ..exceptions import EdgeNotFoundError, NodeNotFoundError
from ..models.edge import Edge


@dataclass
class BaseGraph:
    """
    Pure graph data structure implementation using adjacency list representation.

    This class provides the core graph operations without side concerns like
    events, caching, or transactions. It focuses purely on maintaining the
    graph structure and providing efficient access patterns.

    Attributes:
        _adjacency (Dict[str, Dict[str, Edge]]): Adjacency list representation
        _reverse_index (Dict[str, Set[str]]): Reverse lookup index
        _nodes (Set[str]): Set of all nodes
        _edge_count (int): Total number of edges
    """

    _adjacency: Dict[str, Dict[str, Edge]] = field(default_factory=lambda: defaultdict(dict))
    _reverse_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _nodes: Set[str] = field(default_factory=set)
    _edge_count: int = 0

    def add_edge(self, edge: Edge) -> None:
        """
        Add a single edge to the graph.

        Args:
            edge (Edge): The edge to add
        """
        # Add both nodes to the node set
        self._nodes.add(edge.from_entity)
        self._nodes.add(edge.to_entity)

        # Add the edge to the adjacency list and reverse index
        if edge.to_entity not in self._adjacency[edge.from_entity]:
            self._edge_count += 1
        self._adjacency[edge.from_entity][edge.to_entity] = edge
        self._reverse_index[edge.to_entity].add(edge.from_entity)

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """
        Add multiple edges to the graph efficiently.

        Args:
            edges (List[Edge]): List of edges to add
        """
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """
        Remove an edge from the graph.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            Optional[Edge]: The removed edge if it existed, None otherwise

        Raises:
            EdgeNotFoundError: If the edge doesn't exist
        """
        if not self.has_edge(from_node, to_node):
            raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")

        edge = self._adjacency[from_node][to_node]
        del self._adjacency[from_node][to_node]
        self._reverse_index[to_node].remove(from_node)
        self._edge_count -= 1

        # Clean up empty adjacency entries
        if not self._adjacency[from_node]:
            del self._adjacency[from_node]
        if not self._reverse_index[to_node]:
            del self._reverse_index[to_node]

        return edge

    def remove_edges_batch(self, edges: List[Tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph efficiently.

        Args:
            edges (List[Tuple[str, str]]): List of (from_node, to_node) pairs
        """
        for from_node, to_node in edges:
            self.remove_edge(from_node, to_node)

    def get_neighbors(self, node: str, reverse: bool = False) -> Set[str]:
        """
        Get all neighbors of a node.

        Args:
            node (str): The node to get neighbors for
            reverse (bool): If True, get incoming neighbors instead of outgoing

        Returns:
            Set[str]: Set of neighbor node IDs
        """
        if reverse:
            return self._reverse_index.get(node, set()).copy()
        return set(self._adjacency.get(node, {}).keys())

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """
        Get the edge between two nodes if it exists.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            Optional[Edge]: The edge if it exists, None otherwise
        """
        return self._adjacency.get(from_node, {}).get(to_node)

    def get_edge_safe(self, from_node: str, to_node: str) -> Edge:
        """
        Get the edge between two nodes, raising an error if it doesn't exist.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            Edge: The edge between the nodes

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If the edge doesn't exist
        """
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Source node '{from_node}' not found in the graph")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Target node '{to_node}' not found in the graph")

        edge = self.get_edge(from_node, to_node)
        if edge is None:
            raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")
        return edge

    def get_incoming_edges(self, node: str) -> Set[Edge]:
        """
        Get all incoming edges for a node.

        Args:
            node (str): The node to get incoming edges for

        Returns:
            Set[Edge]: Set of incoming edges
        """
        incoming_edges = set()
        for source_node in self._reverse_index.get(node, set()):
            edge = self.get_edge(source_node, node)
            if edge is not None:
                incoming_edges.add(edge)
        return incoming_edges

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            bool: True if the edge exists, False otherwise
        """
        return self.get_edge(from_node, to_node) is not None

    def get_degree(self, node: str, reverse: bool = False) -> int:
        """
        Get the degree (number of edges) of a node.

        Args:
            node (str): The node to get degree for
            reverse (bool): If True, get in-degree instead of out-degree

        Returns:
            int: The degree of the node
        """
        if reverse:
            return len(self._reverse_index.get(node, set()))
        return len(self._adjacency.get(node, {}))

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes in the graph.

        Returns:
            Set[str]: Set of all node IDs
        """
        return self._nodes.copy()

    def get_edges(self) -> Iterator[Edge]:
        """
        Get all edges in the graph.

        Returns:
            Iterator[Edge]: Iterator over all edges
        """
        for edges in self._adjacency.values():
            yield from edges.values()

    def get_edge_count(self) -> int:
        """
        Get the total number of edges in the graph.

        Returns:
            int: Total number of edges
        """
        return self._edge_count

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node (str): The node to check

        Returns:
            bool: True if the node exists, False otherwise
        """
        return node in self._nodes

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._adjacency.clear()
        self._reverse_index.clear()
        self._nodes.clear()
        self._edge_count = 0

    def build_adjacency_list(self, edges: List[Edge]) -> None:
        """
        Build adjacency list representation from edges.

        Args:
            edges (List[Edge]): List of edges to build the graph from
        """
        self.clear()
        self.add_edges_batch(edges)

    @classmethod
    def from_edges(cls, edges: List[Edge]) -> "BaseGraph":
        """
        Create a new graph from a list of edges.

        Args:
            edges (List[Edge]): List of edges to initialize the graph with

        Returns:
            BaseGraph: New graph instance containing the edges
        """
        graph = cls()
        graph.build_adjacency_list(edges)
        return graph
