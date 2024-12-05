"""
Graph state management and transactions.

This module provides thread-safe state management and transactional operations
for the graph data structure. It handles state changes, backups, and rollbacks
while maintaining thread safety.
"""

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Generator, List, Optional, Set

from .base import BaseGraph
from ..models.edge import Edge


class GraphStateManager:
    """
    Manages graph state and provides transactional operations.

    This class wraps a BaseGraph instance and provides thread-safe access
    and transactional operations. It ensures that state changes are atomic
    and can be rolled back if needed.

    Attributes:
        _graph (BaseGraph): The managed graph instance
        _lock (RLock): Thread lock for synchronization
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

    @property
    def graph(self) -> BaseGraph:
        """
        Get the current graph state.

        Returns:
            BaseGraph: A deep copy of the current graph state
        """
        with self._lock:
            return deepcopy(self._graph)

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
            state_backup = deepcopy(self._graph)
            try:
                # Yield current graph for modification
                yield self._graph
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
            self._graph = deepcopy(new_graph)

    def clear(self) -> None:
        """Clear the graph state."""
        with self._lock:
            self._graph.clear()

    def get_state_snapshot(self) -> BaseGraph:
        """
        Get a snapshot of the current graph state.

        Returns:
            BaseGraph: Deep copy of current graph state
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
        return list(self._state_manager._graph.get_neighbors(node, reverse))

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get the edge between two nodes if it exists."""
        return self._state_manager._graph.get_edge(from_node, to_node)

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
