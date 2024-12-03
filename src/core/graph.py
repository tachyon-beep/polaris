"""
Core graph data structure with efficient adjacency list representation.

This module provides the fundamental Graph class that represents the knowledge graph
structure using an adjacency list implementation. The graph is directed and supports
efficient neighbor lookups and edge queries between nodes.

The graph maintains directed relationships between nodes, where each edge
is accessible from its source node. This design enables fast traversal
and relationship analysis operations.
"""

import json
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

from ..infrastructure.cache import LRUCache
from .enums import RelationType
from .exceptions import EdgeNotFoundError, NodeNotFoundError
from .models import Edge


class Graph:
    """
    Core graph data structure with efficient adjacency list representation.

    This class implements a directed graph using an adjacency list representation,
    optimized for quick neighbor lookups and edge queries. Each node in the graph
    can have multiple edges to other nodes.

    Attributes:
        adjacency (Dict[str, Dict[str, Edge]]): Adjacency list representation where
            the outer dictionary maps node IDs to their neighbors, and the inner
            dictionary maps neighbor IDs to their corresponding edges.
        _node_set (Set[str]): Cache of all node IDs for efficient lookups
        _edge_count (int): Cache of total number of edges
        _path_cache (LRUCache): Cache for frequently accessed paths
    """

    def __init__(self, edges: List[Edge], cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize graph from a list of edges.

        Creates a new Graph instance by building an adjacency list representation
        from the provided edges. Each edge is stored in its directed form.

        Args:
            edges (List[Edge]): List of Edge objects defining the
                connections between nodes in the graph.
            cache_size (int): Maximum number of paths to cache (default: 1000)
            cache_ttl (int): Time-to-live for cached paths in seconds (default: 1 hour)

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
        """
        self.adjacency: Dict[str, Dict[str, Edge]] = defaultdict(dict)
        self._node_set: Set[str] = set()
        self._edge_count: int = 0
        self._path_cache = LRUCache[List[List[str]]](
            max_size=cache_size,
            ttl=cache_ttl,
            serializer=lambda x: {"paths": x},
            deserializer=lambda x: x["paths"],
        )
        self.build_adjacency_list(edges)

    def _get_path_cache_key(
        self, from_node: str, to_node: str, max_depth: Optional[int] = None
    ) -> str:
        """Generate a cache key for path queries."""
        return json.dumps({"from": from_node, "to": to_node, "max_depth": max_depth})

    def find_paths(
        self, from_node: str, to_node: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find all paths between two nodes up to a maximum depth.

        This method uses caching to improve performance for frequently accessed paths.
        Results are cached and returned from cache on subsequent calls with the same
        parameters.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            max_depth (Optional[int]): Maximum path length to consider

        Returns:
            List[List[str]]: List of paths, where each path is a list of node IDs

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> paths = graph.find_paths("A", "C")
            >>> print(paths)
            [["A", "B", "C"]]
        """
        # Check cache first
        cache_key = self._get_path_cache_key(from_node, to_node, max_depth)
        cached_paths = self._path_cache.get(cache_key)
        if cached_paths is not None:
            return cached_paths

        # If not in cache, compute paths
        paths = self._find_paths_impl(from_node, to_node, max_depth)

        # Cache the result
        self._path_cache.put(cache_key, paths)

        return paths

    def _find_paths_impl(
        self, from_node: str, to_node: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """Implementation of path finding algorithm."""
        if not self.has_node(from_node) or not self.has_node(to_node):
            return []

        paths: List[List[str]] = []
        visited = set()

        def dfs(current: str, target: str, path: List[str], depth: int) -> None:
            if max_depth is not None and depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(from_node, to_node, [from_node], 0)
        return paths

    def clear_path_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()

    def get_path_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the path cache.

        Returns:
            Dict containing current cache size
        """
        return {"size": self._path_cache.get_size()}

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """
        Context manager for atomic graph operations.

        Provides transaction-like behavior for a group of graph operations.
        If any operation within the transaction block raises an exception,
        all changes made during the transaction are rolled back.

        Example:
            >>> graph = Graph([])
            >>> try:
            ...     with graph.transaction():
            ...         graph.add_edge(Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO))
            ...         graph.add_edge(Edge(from_entity="B", to_entity="C", relation_type=RelationType.DEPENDS_ON))
            ... except Exception:
            ...     # All changes are rolled back if any operation fails
            ...     pass
        """
        # Create backup of current state
        backup = {
            "adjacency": deepcopy(self.adjacency),
            "node_set": self._node_set.copy(),
            "edge_count": self._edge_count,
        }
        try:
            yield
        except Exception as e:
            # Restore state from backup on error
            self.adjacency = backup["adjacency"]
            self._node_set = backup["node_set"]
            self._edge_count = backup["edge_count"]
            raise e

    def add_edge(self, edge: Edge) -> None:
        """
        Add a single edge to the graph.

        Args:
            edge (Edge): The edge to add to the graph.

        Example:
            >>> graph = Graph([])
            >>> edge = Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            >>> graph.add_edge(edge)
            >>> graph.has_edge("A", "B")
            True
        """
        # Add both nodes to the node set
        self._node_set.add(edge.from_entity)
        self._node_set.add(edge.to_entity)

        # Ensure both nodes have entries in the adjacency list
        if edge.from_entity not in self.adjacency:
            self.adjacency[edge.from_entity] = {}
        if edge.to_entity not in self.adjacency:
            self.adjacency[edge.to_entity] = {}

        # Add the edge to the adjacency list
        if edge.to_entity not in self.adjacency[edge.from_entity]:
            self._edge_count += 1
        self.adjacency[edge.from_entity][edge.to_entity] = edge

        # Clear path cache as graph structure has changed
        self.clear_path_cache()

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """
        Add multiple edges to the graph efficiently.

        This method adds multiple edges in a batch operation, which is more efficient
        than adding edges individually when dealing with large numbers of edges.
        The operation is atomic - if any edge fails to be added, none of the edges
        in the batch will be added.

        Args:
            edges (List[Edge]): List of Edge objects to add to the graph.

        Example:
            >>> edges = [
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ]
            >>> graph.add_edges_batch(edges)
        """
        with self.transaction():
            for edge in edges:
                self.add_edge(edge)

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            from_node (str): ID of the source node.
            to_node (str): ID of the target node.

        Raises:
            EdgeNotFoundError: If no edge exists between the specified nodes.

        Example:
            >>> graph = Graph([Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)])
            >>> graph.remove_edge("A", "B")
            >>> graph.has_edge("A", "B")
            False
        """
        if not self.has_edge(from_node, to_node):
            raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")

        del self.adjacency[from_node][to_node]
        self._edge_count -= 1

        # Clean up empty adjacency entries
        if not self.adjacency[from_node]:
            del self.adjacency[from_node]

        # Clear path cache as graph structure has changed
        self.clear_path_cache()

    def remove_edges_batch(self, edges: List[Tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph efficiently.

        This method removes multiple edges in a batch operation, which is more efficient
        than removing edges individually when dealing with large numbers of edges.
        The operation is atomic - if any edge fails to be removed, none of the edges
        in the batch will be removed.

        Args:
            edges (List[Tuple[str, str]]): List of (from_node, to_node) tuples
                identifying the edges to remove.

        Example:
            >>> edges_to_remove = [("A", "B"), ("B", "C")]
            >>> graph.remove_edges_batch(edges_to_remove)
        """
        with self.transaction():
            for from_node, to_node in edges:
                self.remove_edge(from_node, to_node)

    def build_adjacency_list(self, edges: List[Edge]) -> None:
        """
        Build adjacency list representation from edges.

        Processes each edge and adds it to the adjacency list in its directed form.
        Also ensures that nodes that only appear as targets (to_entity) are included
        in the adjacency list with empty neighbor dictionaries.

        Args:
            edges (List[Edge]): List of Edge objects to add to the graph.

        Example:
            >>> graph = Graph([])
            >>> graph.build_adjacency_list([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> "B" in graph.adjacency  # B is included even though it has no outgoing edges
            True
        """
        self._node_set.clear()
        self._edge_count = 0
        self.clear_path_cache()
        self.add_edges_batch(edges)

    def get_neighbors(self, node: str) -> Set[str]:
        """
        Get all neighbors of a node.

        Returns a set of node IDs that the specified node has edges to.
        If the node has no outgoing edges or doesn't exist in the graph,
        returns an empty set.

        Args:
            node (str): ID of the node to get neighbors for.

        Returns:
            Set[str]: Set of node IDs that the specified node has edges to.
                     Returns an empty set if the node has no neighbors or
                     doesn't exist in the graph.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="A", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> graph.get_neighbors("A")
            {"B", "C"}
            >>> graph.get_neighbors("D")  # Non-existent node
            set()
        """
        return set(self.adjacency.get(node, {}).keys())

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """
        Get the edge between two nodes if it exists.

        Retrieves the Edge object from the source node to the target node
        if such a directed edge exists.

        Args:
            from_node (str): ID of the source node.
            to_node (str): ID of the target node.

        Returns:
            Optional[Edge]: The Edge object from source to target if it exists,
                          None otherwise.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> edge = graph.get_edge("A", "B")
            >>> print(edge.relation_type)
            RelationType.CONNECTS_TO
        """
        return self.adjacency.get(from_node, {}).get(to_node)

    def get_edge_safe(self, from_node: str, to_node: str) -> Edge:
        """
        Get the edge between two nodes, raising an error if it doesn't exist.

        Similar to get_edge(), but raises exceptions if either node doesn't exist
        or if there's no edge between them.

        Args:
            from_node (str): ID of the source node.
            to_node (str): ID of the target node.

        Returns:
            Edge: The Edge object from source to target.

        Raises:
            NodeNotFoundError: If either node doesn't exist in the graph.
            EdgeNotFoundError: If no edge exists between the nodes.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> try:
            ...     edge = graph.get_edge_safe("A", "C")
            ... except EdgeNotFoundError:
            ...     print("No edge found")
        """
        if from_node not in self._node_set:
            raise NodeNotFoundError(f"Source node '{from_node}' not found in the graph")
        if to_node not in self._node_set:
            raise NodeNotFoundError(f"Target node '{to_node}' not found in the graph")

        edge = self.get_edge(from_node, to_node)
        if edge is None:
            raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")
        return edge

    def get_degree(self, node: str) -> int:
        """
        Get the out-degree (number of outgoing edges) of a node.

        Calculates the number of edges originating from the specified node.
        Returns 0 if the node doesn't exist in the graph.

        Args:
            node (str): ID of the node to get the degree for.

        Returns:
            int: The number of outgoing edges from the node.
                Returns 0 if the node doesn't exist.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="A", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> graph.get_degree("A")
            2
            >>> graph.get_degree("D")  # Non-existent node
            0
        """
        return len(self.adjacency.get(node, {}))

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes in the graph.

        Returns a set of all node IDs in the graph, including nodes that only
        appear as targets of edges.

        Returns:
            Set[str]: Set of all node IDs in the graph.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> graph.get_nodes()
            {"A", "B"}
        """
        return self._node_set.copy()

    def get_edges(self) -> Iterator[Edge]:
        """
        Get all edges in the graph.

        Returns an iterator over all Edge objects in the graph.

        Returns:
            Iterator[Edge]: Iterator yielding all edges in the graph.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> for edge in graph.get_edges():
            ...     print(f"{edge.from_entity} -> {edge.to_entity}")
        """
        for from_node, edges in self.adjacency.items():
            yield from edges.values()

    def get_edge_count(self) -> int:
        """
        Get the total number of edges in the graph.

        Returns:
            int: Total number of edges in the graph.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> graph.get_edge_count()
            2
        """
        return self._edge_count

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node (str): ID of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> graph.has_node("A")
            True
            >>> graph.has_node("C")
            False
        """
        return node in self._node_set

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            from_node (str): ID of the source node.
            to_node (str): ID of the target node.

        Returns:
            bool: True if an edge exists from source to target, False otherwise.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> graph.has_edge("A", "B")
            True
            >>> graph.has_edge("B", "A")
            False
        """
        return self.get_edge(from_node, to_node) is not None

    @classmethod
    def from_edges(cls, edges: List[Edge]) -> "Graph":
        """
        Create a Graph instance from a list of edges.

        Factory method that provides an alternative way to create a Graph instance
        from a list of edges.

        Args:
            edges (List[Edge]): List of Edge objects to initialize the graph with.

        Returns:
            Graph: A new Graph instance initialized with the provided edges.

        Example:
            >>> edges = [
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
            ... ]
            >>> graph = Graph.from_edges(edges)
        """
        return cls(edges)
