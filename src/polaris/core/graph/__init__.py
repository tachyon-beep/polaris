"""
Graph module for the Polaris system.

This module provides a complete graph implementation with support for:
- Efficient graph operations using adjacency list representation
- Path finding with multiple algorithm implementations
- Thread-safe state management and transactions
- Event system for graph modifications
- Caching system for frequently accessed paths
"""

from contextlib import contextmanager
from typing import Dict, Generator, Iterator, List, Optional, Set, Type, Union

from .base import BaseGraph
from .state import GraphStateManager, GraphStateView
from .events import (
    GraphEvent,
    GraphEventListener,
    GraphEventManager,
    GraphEventDispatcher,
    GraphEventDetails,
)
from ..models.edge import Edge
from ..exceptions import EdgeNotFoundError, NodeNotFoundError
from ..graph_paths import PathFinding, PathType, PathResult
from ..graph_paths.cache import PathCache


class Graph:
    """
    High-level graph interface combining all components.

    This class provides a unified interface to the graph system, integrating
    the base graph structure with state management, event handling, and path finding.
    """

    def __init__(
        self,
        edges: List[Edge],
        cache_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the graph with its components.

        Args:
            edges (List[Edge]): Initial edges for the graph
            cache_size (int): Maximum number of paths to cache
            cache_ttl (int): Time-to-live for cached paths in seconds
        """
        self._base_graph = BaseGraph.from_edges(edges)
        self.state_manager = GraphStateManager(self._base_graph)
        self.event_manager = GraphEventManager()
        self.event_dispatcher = GraphEventDispatcher(self.event_manager)

    @property
    def adjacency(self) -> Dict[str, Dict[str, Edge]]:
        """Access to adjacency list for backward compatibility."""
        return self.state_manager.graph._adjacency

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for atomic graph operations."""
        with self.state_manager.transaction():
            yield

    def add_edge(self, edge: Edge) -> None:
        """Add a single edge to the graph."""
        with self.state_manager.transaction() as graph:
            graph.add_edge(edge)
            details = GraphEventDetails()
            details.add_edge(edge)
            details.add_node(edge.from_entity)
            details.add_node(edge.to_entity)
            self.event_dispatcher.dispatch(GraphEvent.EDGE_ADDED, details)
            # Clear path cache since graph structure changed
            PathCache.clear()

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """Add multiple edges to the graph efficiently."""
        with self.state_manager.transaction() as graph:
            graph.add_edges_batch(edges)
            details = GraphEventDetails()
            for edge in edges:
                details.add_edge(edge)
                details.add_node(edge.from_entity)
                details.add_node(edge.to_entity)
            self.event_dispatcher.dispatch(GraphEvent.EDGE_ADDED, details)
            # Clear path cache since graph structure changed
            PathCache.clear()

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge from the graph."""
        with self.state_manager.transaction() as graph:
            edge = graph.get_edge(from_node, to_node)
            if edge:
                graph.remove_edge(from_node, to_node)
                details = GraphEventDetails()
                details.add_edge(edge)
                self.event_dispatcher.dispatch(GraphEvent.EDGE_REMOVED, details)
                # Clear path cache since graph structure changed
                PathCache.clear()
            else:
                raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")

    def remove_edges_batch(self, edges: List[tuple[str, str]]) -> None:
        """Remove multiple edges from the graph efficiently."""
        with self.state_manager.transaction() as graph:
            details = GraphEventDetails()
            for from_node, to_node in edges:
                edge = graph.get_edge(from_node, to_node)
                if edge:
                    graph.remove_edge(from_node, to_node)
                    details.add_edge(edge)
                else:
                    raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")
            self.event_dispatcher.dispatch(GraphEvent.EDGE_REMOVED, details)
            # Clear path cache since graph structure changed
            PathCache.clear()

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if an edge exists between two nodes."""
        return self.state_manager.graph.has_edge(from_node, to_node)

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get the edge between two nodes if it exists."""
        return self.state_manager.graph.get_edge(from_node, to_node)

    def get_edge_safe(self, from_node: str, to_node: str) -> Edge:
        """Get the edge between two nodes, raising an error if it doesn't exist."""
        return self.state_manager.graph.get_edge_safe(from_node, to_node)

    def has_node(self, node: str) -> bool:
        """Check if a node exists in the graph."""
        return self.state_manager.graph.has_node(node)

    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        return self.state_manager.graph.get_nodes()

    def get_edges(self) -> Iterator[Edge]:
        """Get all edges in the graph."""
        return self.state_manager.graph.get_edges()

    def get_neighbors(self, node: str, reverse: bool = False) -> Set[str]:
        """Get all neighbors of a node."""
        return self.state_manager.graph.get_neighbors(node, reverse)

    def get_degree(self, node: str, reverse: bool = False) -> int:
        """Get the degree (number of edges) of a node."""
        return self.state_manager.graph.get_degree(node, reverse)

    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return self.state_manager.graph.get_edge_count()

    def find_paths(
        self,
        from_node: str,
        to_node: str,
        max_depth: Optional[int] = None,
        path_type: PathType = PathType.SHORTEST,
        **kwargs,
    ) -> PathResult | Iterator[PathResult]:
        """
        Find paths between nodes using the graph_paths module.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            max_depth (Optional[int]): Maximum path length
            path_type (PathType): Type of path finding to use
            **kwargs: Additional arguments passed to path finder

        Returns:
            Union[PathResult, Iterator[PathResult]]: Found path(s)
        """
        return PathFinding.find_paths(
            self, from_node, to_node, path_type=path_type, max_length=max_depth, **kwargs
        )

    def clear(self) -> None:
        """Clear the graph."""
        with self.state_manager.transaction() as graph:
            graph.clear()
            self.event_dispatcher.dispatch(GraphEvent.GRAPH_CLEARED, GraphEventDetails())
            # Clear path cache since graph structure changed
            PathCache.clear()

    @classmethod
    def from_edges(cls, edges: List[Edge]) -> "Graph":
        """Create a Graph instance from a list of edges."""
        return cls(edges)


__all__ = [
    "Graph",
    "GraphEvent",
    "GraphEventListener",
    "GraphStateView",
    "EdgeNotFoundError",
    "NodeNotFoundError",
]
