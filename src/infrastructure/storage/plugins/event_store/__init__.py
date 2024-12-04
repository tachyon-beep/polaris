"""
Event store plugin implementation.

This package provides a SQLite-based event store implementation for event persistence
and replay. The implementation supports:
- Event storage and retrieval
- Event batch processing
- Dead letter queue for failed events
- Event schema validation
- Event replay functionality

The event store is particularly useful for:
- Maintaining an audit trail of system changes
- Supporting event sourcing patterns
- Enabling event replay for system recovery
- Validating event data through schemas
"""

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_DB_PATH, MAX_RETRY_ATTEMPTS
from .models import Event, EventType
from .store import EventStore

__all__ = [
    "EventStore",
    "Event",
    "EventType",
    "DEFAULT_BATCH_SIZE",
    "MAX_RETRY_ATTEMPTS",
    "DEFAULT_DB_PATH",
]
