"""
Core graph data structure with efficient adjacency list representation.

This module provides the fundamental Graph class that represents the knowledge graph
structure using an adjacency list representation. The graph is directed and supports
efficient neighbor lookups and edge queries between nodes.

The graph maintains directed relationships between nodes, where each edge
is accessible from its source node. This design enables fast traversal
and relationship analysis operations.
"""

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Dict, Generator, Iterator, List, Optional, Protocol, Set, Tuple, Union

from ..infrastructure.cache import LRUCache
from .exceptions import EdgeNotFoundError, NodeNotFoundError
from .models import Edge


class GraphEvent(Enum):
    """Events that can occur in the graph."""

    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    EDGE_ADDED = auto()
    EDGE_REMOVED = auto()
    GRAPH_CLEARED = auto()


class GraphStateListener(Protocol):
    """Protocol for objects that listen to graph state changes."""

    def on_state_change(self, change_type: GraphEvent, details: dict) -> None:
        """Called when the graph state changes."""


@dataclass
class GraphState:
    """Encapsulates the state of a graph."""

    adjacency: Dict[str, Dict[str, Edge]] = field(default_factory=lambda: defaultdict(dict))
    reverse_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    node_set: Set[str] = field(default_factory=set)
    edge_count: int = 0


class Graph:
    """
    Core graph data structure with efficient adjacency list representation.

    This class implements a directed graph using an adjacency list representation,
    optimized for quick neighbor lookups and edge queries. Each node in the graph
    can have multiple edges to other nodes.

    Attributes:
        _state (GraphState): Internal state of the graph
        _state_lock (RLock): Lock for thread-safe state access
        _listeners (List[GraphStateListener]): State change listeners
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
        """
        self._state = GraphState()
        self._state_lock = RLock()
        self._listeners: List[GraphStateListener] = []
        self._path_cache = LRUCache[List[List[str]]](
            max_size=cache_size,
            base_ttl=cache_ttl,
            adaptive_ttl=True,
            min_ttl=cache_ttl // 2,
            max_ttl=cache_ttl * 2,
        )
        self.build_adjacency_list(edges)

    # Properties for backward compatibility
    @property
    def adjacency(self) -> Dict[str, Dict[str, Edge]]:
        """Access to adjacency list for backward compatibility."""
        with self._state_lock:
            return self._state.adjacency

    @property
    def reverse_index(self) -> Dict[str, Set[str]]:
        """Access to reverse index for backward compatibility."""
        with self._state_lock:
            return self._state.reverse_index

    @property
    def _node_set(self) -> Set[str]:
        """Access to node set for backward compatibility."""
        with self._state_lock:
            return self._state.node_set

    @property
    def _edge_count(self) -> int:
        """Access to edge count for backward compatibility."""
        with self._state_lock:
            return self._state.edge_count

    def add_state_listener(self, listener: GraphStateListener) -> None:
        """Add a listener for state changes."""
        self._listeners.append(listener)

    def remove_state_listener(self, listener: GraphStateListener) -> None:
        """Remove a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_state_change(self, change_type: GraphEvent, details: dict) -> None:
        """Notify listeners of a state change."""
        for listener in self._listeners:
            listener.on_state_change(change_type, details)

    def _get_path_cache_key(
        self,
        from_node: str,
        to_node: str,
        max_depth: Optional[int] = None,
        algo: str = "default",
        weight_func_name: Optional[str] = None,
    ) -> str:
        """Generate a cache key for path queries."""
        return f"{from_node}|{to_node}|{max_depth}|{algo}|{weight_func_name}"

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
        """
        cache_key = self._get_path_cache_key(from_node, to_node, max_depth)
        cached_paths = self._path_cache.get(cache_key)
        if cached_paths is not None:
            return cached_paths

        paths = self._find_paths_impl(from_node, to_node, max_depth)
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

    def get_path_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the path cache."""
        return self._path_cache.get_metrics()

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for atomic graph operations."""
        with self._state_lock:
            state_backup = deepcopy(self._state)
            try:
                yield
            except Exception as e:
                self._state = state_backup
                raise e

    def add_edge(self, edge: Edge) -> None:
        """Add a single edge to the graph."""
        with self._state_lock:
            # Add both nodes to the node set
            self._state.node_set.add(edge.from_entity)
            self._state.node_set.add(edge.to_entity)

            # Add the edge to the adjacency list and reverse index
            if edge.to_entity not in self._state.adjacency[edge.from_entity]:
                self._state.edge_count += 1
            self._state.adjacency[edge.from_entity][edge.to_entity] = edge
            self._state.reverse_index[edge.to_entity].add(edge.from_entity)

            # Clear path cache and notify listeners
            self.clear_path_cache()
            self._notify_state_change(
                GraphEvent.EDGE_ADDED,
                {"from_node": edge.from_entity, "to_node": edge.to_entity, "edge": edge},
            )

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """Add multiple edges to the graph efficiently."""
        with self.transaction():
            for edge in edges:
                self.add_edge(edge)

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge from the graph."""
        with self._state_lock:
            if not self.has_edge(from_node, to_node):
                raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")

            edge = self._state.adjacency[from_node][to_node]
            del self._state.adjacency[from_node][to_node]
            self._state.reverse_index[to_node].remove(from_node)
            self._state.edge_count -= 1

            # Clean up empty adjacency entries
            if not self._state.adjacency[from_node]:
                del self._state.adjacency[from_node]
            if not self._state.reverse_index[to_node]:
                del self._state.reverse_index[to_node]

            # Clear path cache and notify listeners
            self.clear_path_cache()
            self._notify_state_change(
                GraphEvent.EDGE_REMOVED,
                {"from_node": from_node, "to_node": to_node, "edge": edge},
            )

    def remove_edges_batch(self, edges: List[Tuple[str, str]]) -> None:
        """Remove multiple edges from the graph efficiently."""
        with self.transaction():
            for from_node, to_node in edges:
                self.remove_edge(from_node, to_node)

    def build_adjacency_list(self, edges: List[Edge]) -> None:
        """Build adjacency list representation from edges."""
        with self._state_lock:
            self._state = GraphState()  # Reset state
            self.clear_path_cache()
            self.add_edges_batch(edges)
            self._notify_state_change(GraphEvent.GRAPH_CLEARED, {})

    def get_neighbors(self, node: str, reverse: bool = False) -> Set[str]:
        """Get all neighbors of a node."""
        with self._state_lock:
            if reverse:
                return self._state.reverse_index.get(node, set()).copy()
            return set(self._state.adjacency.get(node, {}).keys())

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get the edge between two nodes if it exists."""
        with self._state_lock:
            return self._state.adjacency.get(from_node, {}).get(to_node)

    def get_incoming_edges(self, node: str) -> Set[Edge]:
        """Get all incoming edges for a node."""
        with self._state_lock:
            incoming_edges = set()
            for source_node in self._state.reverse_index.get(node, set()):
                edge = self.get_edge(source_node, node)
                if edge is not None:
                    incoming_edges.add(edge)
            return incoming_edges

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if an edge exists between two nodes."""
        return self.get_edge(from_node, to_node) is not None

    def get_edge_safe(self, from_node: str, to_node: str) -> Edge:
        """Get the edge between two nodes, raising an error if it doesn't exist."""
        with self._state_lock:
            if from_node not in self._state.node_set:
                raise NodeNotFoundError(f"Source node '{from_node}' not found in the graph")
            if to_node not in self._state.node_set:
                raise NodeNotFoundError(f"Target node '{to_node}' not found in the graph")

            edge = self.get_edge(from_node, to_node)
            if edge is None:
                raise EdgeNotFoundError(f"No edge exists from '{from_node}' to '{to_node}'")
            return edge

    def get_degree(self, node: str, reverse: bool = False) -> int:
        """Get the degree (number of edges) of a node."""
        with self._state_lock:
            if reverse:
                return len(self._state.reverse_index.get(node, set()))
            return len(self._state.adjacency.get(node, {}))

    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        with self._state_lock:
            return self._state.node_set.copy()

    def get_edges(self) -> Iterator[Edge]:
        """Get all edges in the graph."""
        with self._state_lock:
            for _, edges in self._state.adjacency.items():
                yield from edges.values()

    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        with self._state_lock:
            return self._state.edge_count

    def has_node(self, node: str) -> bool:
        """Check if a node exists in the graph."""
        with self._state_lock:
            return node in self._state.node_set

    @classmethod
    def from_edges(cls, edges: List[Edge]) -> "Graph":
        """Create a Graph instance from a list of edges."""
        return cls(edges)
