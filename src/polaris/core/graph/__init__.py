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
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Union,
)

from .base import BaseGraph
from .state import GraphStateManager, GraphStateView
from .events import (
    GraphEvent,
    GraphEventListener,
    GraphEventManager,
    GraphEventDispatcher,
    GraphEventDetails,
)
from .traversal import (
    PathFinder,
    PerformanceMetrics,
    AllPathsFinder,
    BidirectionalFinder,
    ShortestPathFinder,
    PathResult,
    PathType,
)
from ..models.edge import Edge
from ..exceptions import EdgeNotFoundError, NodeNotFoundError
from .traversal.cache import PathCache


class Graph:
    """
    High-level graph interface combining all components.

    This class provides a unified interface to the graph system, integrating
    the base graph structure with state management, event handling, and path finding.
    """

    def __init__(
        self,
        edges: List[Edge],
        cache_size: int = 100000,  # Increased default cache size
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
        PathCache.reconfigure(max_size=cache_size, ttl=cache_ttl)

    @property
    def adjacency(self) -> Dict[str, Dict[str, Edge]]:
        """Access to adjacency list for backward compatibility."""
        return self.state_manager.graph._adjacency

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for atomic graph operations."""
        with self.state_manager.transaction():
            yield

    def _invalidate_affected_paths(self, edge: Edge) -> None:
        """Invalidate cached paths that could be affected by edge changes."""
        # Get all paths from cache
        metrics = PathCache.get_metrics()
        cache_size = int(metrics["size"])

        # If cache is nearly empty, just clear it
        if cache_size < 10:
            PathCache.clear()
            return

        # Invalidate paths that could be affected by this edge
        from_node = edge.from_entity
        to_node = edge.to_entity

        # Get all neighbors to invalidate paths that might use this edge
        neighbors_from = self.get_neighbors(from_node)
        neighbors_to = self.get_neighbors(to_node)

        # Invalidate paths between affected nodes
        affected_nodes = {from_node, to_node} | neighbors_from | neighbors_to
        for node1 in affected_nodes:
            for node2 in affected_nodes:
                if node1 != node2:
                    # Generate all possible cache keys for this node pair
                    for path_type in ["shortest", "all"]:
                        key = PathCache.get_cache_key(node1, node2, path_type)
                        PathCache.invalidate(key)

    def add_edge(self, edge: Edge) -> None:
        """Add a single edge to the graph."""
        with self.state_manager.transaction() as graph:
            graph.add_edge(edge)
            details = GraphEventDetails()
            details.add_edge(edge)
            details.add_node(edge.from_entity)
            details.add_node(edge.to_entity)
            self.event_dispatcher.dispatch(GraphEvent.EDGE_ADDED, details)
            # Invalidate affected paths instead of clearing entire cache
            self._invalidate_affected_paths(edge)

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
            # For batch operations, clear the entire cache
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
                # Invalidate affected paths instead of clearing entire cache
                self._invalidate_affected_paths(edge)
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
            # For batch operations, clear the entire cache
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

    def _find_shortest_path(
        self,
        from_node: str,
        to_node: str,
        max_depth: Optional[int] = None,
        **kwargs: Any,
    ) -> PathResult:
        """Helper method to find shortest path."""
        paths = ShortestPathFinder(self).find_paths(
            start_node=from_node, end_node=to_node, max_length=max_depth, **kwargs
        )
        try:
            return next(paths)
        except StopIteration as exc:
            raise NodeNotFoundError(f"No path exists between {from_node} and {to_node}") from exc

    def find_paths(
        self,
        from_node: str,
        to_node: str,
        max_depth: Optional[int] = None,
        path_type: PathType = PathType.SHORTEST,
        **kwargs: Any,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """
        Find paths between nodes using the specified algorithm.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            max_depth (Optional[int]): Maximum path length
            path_type (PathType): Type of path finding to use
            **kwargs: Additional arguments passed to path finder

        Returns:
            Union[PathResult, Iterator[PathResult]]: Found path(s)
            For SHORTEST path type, returns a single PathResult
            For ALL path type, returns an Iterator[PathResult]
        """
        if path_type in (PathType.ALL, PathType.ALL_PATHS, PathType.FILTERED):
            return AllPathsFinder(self).find_paths(
                start_node=from_node, end_node=to_node, max_length=max_depth, **kwargs
            )
        return self._find_shortest_path(from_node, to_node, max_depth, **kwargs)

    def clear(self) -> None:
        """Clear the graph."""
        with self.state_manager.transaction() as graph:
            graph.clear()
            self.event_dispatcher.dispatch(GraphEvent.GRAPH_CLEARED, GraphEventDetails())
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
    "PathFinder",
    "PathResult",
    "PerformanceMetrics",
    "AllPathsFinder",
    "BidirectionalFinder",
    "ShortestPathFinder",
]
