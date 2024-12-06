"""
Graph state management and transactions.

This module provides thread-safe state management and transactional operations
for the graph data structure. It handles state changes, backups, and rollbacks
while maintaining thread safety.
"""

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Generator, List, Optional, Set, Dict, Any
from weakref import WeakValueDictionary

from .base import BaseGraph
from ..models.edge import Edge


class CopyOnWriteDict:
    """A dictionary that creates copies of values only when they are modified."""

    def __init__(self):
        self._data: Dict[Any, Any] = {}
        self._copied = False

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._copied:
            self._data = self._data.copy()
            self._copied = True
        self._data[key] = value

    def get(self, key: Any, default: Any = None) -> Any:
        return self._data.get(key, default)

    def copy(self) -> "CopyOnWriteDict":
        new_dict = CopyOnWriteDict()
        new_dict._data = self._data.copy()
        return new_dict


class GraphStateManager:
    """
    Manages graph state and provides transactional operations.

    This class wraps a BaseGraph instance and provides thread-safe access
    and transactional operations. It ensures that state changes are atomic
    and can be rolled back if needed.

    Attributes:
        _graph (BaseGraph): The managed graph instance
        _lock (RLock): Thread lock for synchronization
        _edge_cache (CopyOnWriteDict): Cache of edge lookups
        _neighbor_cache (CopyOnWriteDict): Cache of neighbor lookups
    """

    def __init__(self, graph: Optional[BaseGraph] = None):
        """
        Initialize the state manager.

        Args:
            graph (Optional[BaseGraph]): Initial graph instance.
                If None, creates a new empty graph.
        """
        self._graph = graph if graph is not None else BaseGraph()
        self._lock = RLock()
        self._edge_cache = CopyOnWriteDict()
        self._neighbor_cache = CopyOnWriteDict()

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get edge with caching."""
        with self._lock:
            cache_key = f"{from_node}:{to_node}"
            cached = self._edge_cache.get(cache_key)
            if cached is None:
                edge = self._graph.get_edge(from_node, to_node)
                if edge:
                    # Store a shallow copy in cache
                    cached = Edge(
                        from_entity=edge.from_entity,
                        to_entity=edge.to_entity,
                        relation_type=edge.relation_type,
                        metadata=edge.metadata,
                        impact_score=edge.impact_score,
                        attributes=edge.attributes,
                        context=edge.context,
                        validation_status=edge.validation_status,
                        custom_metrics=edge.custom_metrics,
                    )
                    self._edge_cache[cache_key] = cached
            return cached

    def get_neighbors(self, node: str, reverse: bool = False) -> List[str]:
        """Get neighbors with caching."""
        with self._lock:
            cache_key = f"{node}:{reverse}"
            cached = self._neighbor_cache.get(cache_key)
            if cached is None:
                neighbors = self._graph.get_neighbors(node, reverse)
                cached = list(neighbors)
                self._neighbor_cache[cache_key] = cached
            return cached.copy()

    @property
    def graph(self) -> BaseGraph:
        """
        Get the current graph state.

        Returns:
            BaseGraph: A copy of the current graph state
        """
        with self._lock:
            # Create a new graph instance
            state_copy = BaseGraph()
            # Copy nodes and edges efficiently
            for node in self._graph.get_nodes():
                for neighbor in self.get_neighbors(node):
                    edge = self.get_edge(node, neighbor)
                    if edge:
                        state_copy.add_edge(edge)
            return state_copy

    @contextmanager
    def transaction(self) -> Generator[BaseGraph, None, None]:
        """
        Context manager for atomic graph operations.

        This method provides transactional access to the graph state. Changes
        made within the transaction are atomic - either all changes are applied
        or none are. If an exception occurs during the transaction, the state
        is rolled back to its previous state.

        Yields:
            BaseGraph: The current graph state for modification

        Raises:
            Exception: Any exception that occurs during the transaction
                will trigger a rollback
        """
        with self._lock:
            # Create backup of current state
            state_backup = self.graph
            try:
                # Yield current graph for modification
                yield self._graph
                # Clear caches on successful transaction
                self._edge_cache = CopyOnWriteDict()
                self._neighbor_cache = CopyOnWriteDict()
            except Exception as e:
                # Restore backup on error
                self._graph = state_backup
                raise e

    def update_graph(self, new_graph: BaseGraph) -> None:
        """
        Update the entire graph state atomically.

        This method replaces the current graph state with a new one
        in a thread-safe manner.

        Args:
            new_graph (BaseGraph): New graph state to apply
        """
        with self._lock:
            self._graph = new_graph
            # Clear caches on update
            self._edge_cache = CopyOnWriteDict()
            self._neighbor_cache = CopyOnWriteDict()

    def clear(self) -> None:
        """Clear the graph state."""
        with self._lock:
            self._graph.clear()
            # Clear caches
            self._edge_cache = CopyOnWriteDict()
            self._neighbor_cache = CopyOnWriteDict()

    def get_state_snapshot(self) -> BaseGraph:
        """
        Get a snapshot of the current graph state.

        Returns:
            BaseGraph: Copy of current graph state
        """
        return self.graph


class GraphStateError(Exception):
    """Base exception for graph state errors."""

    pass


class TransactionError(GraphStateError):
    """Exception raised when a transaction fails."""

    pass


class StateAccessError(GraphStateError):
    """Exception raised when state access fails."""

    pass


class GraphStateView:
    """
    Provides a read-only view of the graph state.

    This class offers a way to safely access graph state without
    the possibility of modification, useful for concurrent read
    operations.

    Attributes:
        _state_manager (GraphStateManager): The underlying state manager
    """

    def __init__(self, state_manager: GraphStateManager):
        """
        Initialize the state view.

        Args:
            state_manager (GraphStateManager): State manager to view
        """
        self._state_manager = state_manager

    def get_nodes(self) -> List[str]:
        """Get all nodes in the graph."""
        return list(self._state_manager._graph.get_nodes())

    def get_neighbors(self, node: str, reverse: bool = False) -> List[str]:
        """Get all neighbors of a node."""
        return self._state_manager.get_neighbors(node, reverse)

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get the edge between two nodes if it exists."""
        return self._state_manager.get_edge(from_node, to_node)

    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return self._state_manager._graph.get_edge_count()

    def has_node(self, node: str) -> bool:
        """Check if a node exists in the graph."""
        return self._state_manager._graph.has_node(node)

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if an edge exists between two nodes."""
        return self._state_manager._graph.has_edge(from_node, to_node)

    def get_degree(self, node: str, reverse: bool = False) -> int:
        """Get the degree (number of edges) of a node."""
        return self._state_manager._graph.get_degree(node, reverse)

    def get_incoming_edges(self, node: str) -> Set[Edge]:
        """Get all incoming edges for a node."""
        return self._state_manager._graph.get_incoming_edges(node)
