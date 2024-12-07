"""
Graph event system.

This module provides an event system for graph operations, allowing components
to subscribe to and be notified of changes in the graph state. It supports
multiple listeners and thread-safe event dispatch.
"""

from enum import Enum, auto
from threading import RLock
from typing import Any, Dict, List, Protocol, Set as SetType
from dataclasses import dataclass, field

from ..models.edge import Edge


class GraphEvent(Enum):
    """Events that can occur in the graph."""

    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    EDGE_ADDED = auto()
    EDGE_REMOVED = auto()
    GRAPH_CLEARED = auto()


class GraphEventListener(Protocol):
    """Protocol for objects that listen to graph state changes."""

    def on_state_change(self, event: GraphEvent, details: Dict) -> None:
        """
        Called when the graph state changes.

        Args:
            event (GraphEvent): Type of event that occurred
            details (Dict): Additional information about the event
        """
        ...


@dataclass
class GraphEventManager:
    """
    Manages graph event subscriptions and notifications.

    This class handles the registration of event listeners and the dispatch
    of events to those listeners in a thread-safe manner.

    Attributes:
        _listeners (List[GraphEventListener]): Registered event listeners
        _lock (RLock): Thread lock for synchronization
    """

    _listeners: List[GraphEventListener] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock)

    def add_listener(self, listener: GraphEventListener) -> None:
        """
        Add a listener for graph events.

        Args:
            listener (GraphEventListener): The listener to add
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: GraphEventListener) -> None:
        """
        Remove a graph event listener.

        Args:
            listener (GraphEventListener): The listener to remove
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def notify(self, event: GraphEvent, details: Dict) -> None:
        """
        Notify all listeners of a graph event.

        This method safely notifies all registered listeners of the event
        that occurred, passing along any relevant details.

        Args:
            event (GraphEvent): The type of event that occurred
            details (Dict): Additional information about the event
        """
        with self._lock:
            listeners = self._listeners.copy()

        for listener in listeners:
            try:
                listener.on_state_change(event, details)
            except Exception as e:
                # Log error but continue notifying other listeners
                print(f"Error notifying listener {listener}: {e}")

    def clear_listeners(self) -> None:
        """Remove all event listeners."""
        with self._lock:
            self._listeners.clear()


class GraphEventError(Exception):
    """Base exception for graph event errors."""

    pass


class EventDispatchError(GraphEventError):
    """Exception raised when event dispatch fails."""

    pass


@dataclass
class GraphEventDetails:
    """
    Container for graph event details.

    This class provides a structured way to pass event information
    to listeners, ensuring consistency in event details.

    Attributes:
        nodes (Set[str]): Affected node IDs
        edges (Set[Edge]): Affected edges
        metadata (Dict): Additional event metadata
    """

    nodes: SetType[str] = field(default_factory=set)
    edges: SetType[Edge] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: str) -> None:
        """Add an affected node."""
        self.nodes.add(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an affected edge."""
        self.edges.add(edge)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add additional metadata."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert event details to dictionary format."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "from_node": edge.from_entity,
                    "to_node": edge.to_entity,
                    "type": edge.relation_type.name,
                }
                for edge in self.edges
            ],
            "metadata": self.metadata,
        }


class GraphEventDispatcher:
    """
    Handles the dispatching of graph events with retry capability.

    This class provides reliable event dispatch with support for
    retrying failed notifications and handling dispatch errors.

    Attributes:
        _event_manager (GraphEventManager): The event manager to use
        _max_retries (int): Maximum number of retry attempts
    """

    def __init__(self, event_manager: GraphEventManager, max_retries: int = 3):
        """
        Initialize the event dispatcher.

        Args:
            event_manager (GraphEventManager): Event manager to use
            max_retries (int): Maximum retry attempts for failed dispatches
        """
        self._event_manager = event_manager
        self._max_retries = max_retries

    def dispatch(self, event: GraphEvent, details: GraphEventDetails) -> None:
        """
        Dispatch an event with retry capability.

        This method attempts to dispatch the event, retrying on failure
        up to the configured maximum number of attempts.

        Args:
            event (GraphEvent): The event to dispatch
            details (GraphEventDetails): Event details

        Raises:
            EventDispatchError: If dispatch fails after all retries
        """
        retries = 0
        last_error = None

        while retries <= self._max_retries:
            try:
                self._event_manager.notify(event, details.to_dict())
                return
            except Exception as e:
                last_error = e
                retries += 1

        raise EventDispatchError(
            f"Failed to dispatch event after {self._max_retries} attempts"
        ) from last_error
