"""
Graph module for the Polaris system.

This module provides a complete graph implementation with support for:
- Efficient graph operations using adjacency list representation
- Path finding with multiple algorithm implementations
- Thread-safe state management and transactions
- Event system for graph modifications
- Caching system for frequently accessed paths

Example:
    >>> from polaris.core.graph import Graph
    >>> from polaris.core.models import Edge
    >>> 
    >>> # Create a graph with some edges
    >>> edges = [
    ...     Edge(from_entity="A", to_entity="B", ...),
    ...     Edge(from_entity="B", to_entity="C", ...)
    ... ]
    >>> graph = Graph(edges)
    >>> 
    >>> # Find paths between nodes
    >>> paths = graph.find_paths("A", "C")
"""

from typing import Dict, Iterator, List, Optional, Set, Type, Union

from .base import BaseGraph
from .traversal import PathFinder, DFSPathFinder, BFSPathFinder, BiDirectionalPathFinder
from .state import GraphStateManager, GraphStateView
from .events import (
    GraphEvent,
    GraphEventListener,
    GraphEventManager,
    GraphEventDispatcher,
    GraphEventDetails,
)
from .cache import PathCache
from ..models.edge import Edge
from ..exceptions import EdgeNotFoundError, NodeNotFoundError


class Graph:
    """
    High-level graph interface combining all components.

    This class provides a unified interface to the graph system, integrating
    the base graph structure with state management, event handling, path finding,
    and caching capabilities.

    Attributes:
        state_manager (GraphStateManager): Manages graph state and transactions
        event_manager (GraphEventManager): Handles graph events
        path_cache (PathCache): Caches path finding results
    """

    def __init__(
        self,
        edges: List[Edge],
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        path_finder: Optional[Type[PathFinder]] = None,
    ):
        """
        Initialize the graph with its components.

        Args:
            edges (List[Edge]): Initial edges for the graph
            cache_size (int): Maximum number of paths to cache
            cache_ttl (int): Time-to-live for cached paths in seconds
            path_finder (Optional[Type[PathFinder]]): Path finding implementation
                to use. Defaults to DFSPathFinder.
        """
        self._base_graph = BaseGraph.from_edges(edges)
        self.state_manager = GraphStateManager(self._base_graph)
        self.event_manager = GraphEventManager()
        self.event_dispatcher = GraphEventDispatcher(self.event_manager)
        self.path_cache = PathCache()
        self._path_finder = (path_finder or DFSPathFinder)()

    def add_edge(self, edge: Edge) -> None:
        """
        Add a single edge to the graph.

        Args:
            edge (Edge): The edge to add
        """
        with self.state_manager.transaction() as graph:
            graph.add_edge(edge)
            details = GraphEventDetails()
            details.add_edge(edge)
            details.add_node(edge.from_entity)
            details.add_node(edge.to_entity)
            self.event_dispatcher.dispatch(GraphEvent.EDGE_ADDED, details)
            self.path_cache.clear()

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """
        Add multiple edges to the graph efficiently.

        Args:
            edges (List[Edge]): List of edges to add
        """
        with self.state_manager.transaction() as graph:
            graph.add_edges_batch(edges)
            self.path_cache.clear()
            details = GraphEventDetails()
            for edge in edges:
                details.add_edge(edge)
                details.add_node(edge.from_entity)
                details.add_node(edge.to_entity)
            self.event_dispatcher.dispatch(GraphEvent.EDGE_ADDED, details)

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Raises:
            EdgeNotFoundError: If the edge doesn't exist
        """
        with self.state_manager.transaction() as graph:
            edge = graph.get_edge(from_node, to_node)
            if edge:
                graph.remove_edge(from_node, to_node)
                details = GraphEventDetails()
                details.add_edge(edge)
                self.event_dispatcher.dispatch(GraphEvent.EDGE_REMOVED, details)
                self.path_cache.clear()

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            bool: True if the edge exists, False otherwise
        """
        return self.state_manager.graph.has_edge(from_node, to_node)

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """
        Get the edge between two nodes if it exists.

        Args:
            from_node (str): Source node
            to_node (str): Target node

        Returns:
            Optional[Edge]: The edge if it exists, None otherwise
        """
        return self.state_manager.graph.get_edge(from_node, to_node)

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
        return self.state_manager.graph.get_edge_safe(from_node, to_node)

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node (str): The node to check

        Returns:
            bool: True if the node exists, False otherwise
        """
        return self.state_manager.graph.has_node(node)

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes in the graph.

        Returns:
            Set[str]: Set of all node IDs
        """
        return self.state_manager.graph.get_nodes()

    def get_edges(self) -> Iterator[Edge]:
        """
        Get all edges in the graph.

        Returns:
            Iterator[Edge]: Iterator over all edges
        """
        return self.state_manager.graph.get_edges()

    def get_neighbors(self, node: str, reverse: bool = False) -> Set[str]:
        """
        Get all neighbors of a node.

        Args:
            node (str): The node to get neighbors for
            reverse (bool): If True, get incoming neighbors instead of outgoing

        Returns:
            Set[str]: Set of neighbor node IDs
        """
        return self.state_manager.graph.get_neighbors(node, reverse)

    def get_degree(self, node: str, reverse: bool = False) -> int:
        """
        Get the degree (number of edges) of a node.

        Args:
            node (str): The node to get degree for
            reverse (bool): If True, get in-degree instead of out-degree

        Returns:
            int: The degree of the node
        """
        return self.state_manager.graph.get_degree(node, reverse)

    def find_paths(
        self, from_node: str, to_node: str, max_depth: Optional[int] = None, use_cache: bool = True
    ) -> List[List[str]]:
        """
        Find paths between two nodes.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            max_depth (Optional[int]): Maximum path length
            use_cache (bool): Whether to use path caching

        Returns:
            List[List[str]]: List of paths found
        """
        if use_cache:
            cached = self.path_cache.get_paths(from_node, to_node, max_depth)
            if cached is not None:
                return cached

        paths = self._path_finder.find_paths(
            self.state_manager.graph, from_node, to_node, max_depth
        )

        if use_cache:
            self.path_cache.cache_paths(from_node, to_node, paths, max_depth)

        return paths

    def clear(self) -> None:
        """Clear the graph and all associated caches."""
        with self.state_manager.transaction() as graph:
            graph.clear()
            self.path_cache.clear()
            self.event_dispatcher.dispatch(GraphEvent.GRAPH_CLEARED, GraphEventDetails())

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Union[int, float]]: Cache statistics and metrics
        """
        return self.path_cache.get_stats()


__all__ = [
    "Graph",
    "GraphEvent",
    "GraphEventListener",
    "GraphStateView",
    "PathFinder",
    "DFSPathFinder",
    "BFSPathFinder",
    "BiDirectionalPathFinder",
    "EdgeNotFoundError",
    "NodeNotFoundError",
]
