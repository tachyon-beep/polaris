"""
Models for event store.

This module defines the core data models used by the event store:
- Event: Represents a single event in the system
- EventType: Enumeration of supported event types
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """
    Types of events that can be stored.

    Each event type represents a distinct kind of system event that can be
    stored and processed by the event store.
    """

    NODE_CREATED = "node_created"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    EDGE_CREATED = "edge_created"
    EDGE_UPDATED = "edge_updated"
    EDGE_DELETED = "edge_deleted"
    GRAPH_VALIDATED = "graph_validated"
    SCHEMA_UPDATED = "schema_updated"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


@dataclass
class Event:
    """
    Represents a single event in the system.

    An event captures a specific occurrence in the system, including:
    - The type of event that occurred
    - The data associated with the event
    - When the event occurred
    - Optional metadata about the event

    Attributes:
        type (EventType): Type of the event
        data (Dict[str, Any]): Event payload data
        timestamp (datetime): When the event occurred
        metadata (Optional[Dict[str, Any]]): Additional event metadata
    """

    type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
