# Event Store Plugin

A SQLite-based event store implementation for the Polaris graph database system that provides event persistence, processing, and replay capabilities.

## Overview

The Event Store Plugin provides a robust event storage and processing system. It supports event persistence, batch processing, dead letter queue handling, and event schema validation, making it ideal for event-driven architectures and audit trail requirements.

## Directory Structure

```
event_store/
├── __init__.py      # Plugin initialization and exports
├── constants.py     # SQL schemas and queries
├── models.py        # Event data models
└── store.py         # Core event store implementation
```

## Components

### Event Store (`store.py`)

Core event store implementation:

- **Purpose**: Manages event persistence and processing
- **Key Features**:
  - Event persistence and retrieval
  - Batch processing support
  - Dead letter queue for failed events
  - Event schema validation
  - Event replay functionality
- **Key Methods**:
  - `store_event()`: Store a new event
  - `get_events()`: Retrieve events with filtering
  - `move_to_dead_letter_queue()`: Handle failed events
  - `retry_dead_letter_queue()`: Retry failed events
  - `store_event_schema()`: Store event validation schema

Example usage:
```python
from polaris.infrastructure.storage.plugins.event_store import EventStore, Event, EventType

# Initialize store
store = EventStore("/path/to/db")

# Store an event
event = Event(
    type=EventType.NODE_CREATED,
    data={"node_name": "example"},
    timestamp=datetime.now()
)
event_id = await store.store_event(event)
```

### Event Models (`models.py`)

Event data models and types:

- **Purpose**: Define event structure and types
- **Key Features**:
  - Event type enumeration
  - Event data structure
  - Metadata support
- **Key Classes**:
  - `Event`: Core event data model
  - `EventType`: Event type enumeration

Example usage:
```python
from polaris.infrastructure.storage.plugins.event_store.models import Event, EventType

# Create an event
event = Event(
    type=EventType.NODE_UPDATED,
    data={"node_name": "example", "changes": {...}},
    timestamp=datetime.now(),
    metadata={"user": "admin"}
)
```

## Common Features

1. **Event Processing**
   - Persistent event storage
   - Event filtering and retrieval
   - Batch processing support
   - Event replay capabilities

2. **Error Handling**
   - Dead letter queue for failed events
   - Retry mechanism
   - Error tracking and reporting
   - Maximum retry attempts

3. **Schema Validation**
   - Event schema storage
   - Schema validation support
   - Schema versioning
   - Dynamic schema updates

## Usage Examples

### Basic Event Operations
```python
# Initialize store
store = EventStore("/path/to/db")

# Store an event
event = Event(
    type=EventType.NODE_CREATED,
    data={"node_name": "example"},
    timestamp=datetime.now()
)
event_id = await store.store_event(event)

# Retrieve events
events = await store.get_events(
    event_type=EventType.NODE_CREATED,
    start_time=datetime.now() - timedelta(days=1)
)
```

### Batch Processing
```python
# Create a batch
batch_id = "batch_123"
await store.create_batch(batch_id)

# Store events in batch
for item in items:
    event = Event(
        type=EventType.NODE_CREATED,
        data=item,
        timestamp=datetime.now()
    )
    await store.store_event(event, batch_id)

# Complete batch
await store.complete_batch(batch_id)
```

## Architecture

1. **SQLite Backend**: Uses SQLite for reliable event storage
2. **Event Processing**: Sequential event processing with batch support
3. **Error Recovery**: Dead letter queue with retry mechanism
4. **Schema Management**: Dynamic schema validation support

## Dependencies

- Internal:
  - `core.exceptions`: Error types
- External:
  - `sqlite3`: Python's built-in SQLite support
  - Python 3.7+ for dataclasses

## Error Handling

Comprehensive error handling approach:

- Failed events go to dead letter queue
- Configurable retry attempts
- Error tracking and reporting
- Transaction rollback on failures

Example:
```python
try:
    await store.store_event(event)
except Exception as e:
    # Move to dead letter queue
    await store.move_to_dead_letter_queue(event_id, str(e))
```

## Performance Considerations

1. **Database Optimization**
   - Indexed event queries
   - Batch processing for bulk operations
   - Regular cleanup of processed events

2. **Event Processing**
   - Configurable batch sizes
   - Async event processing
   - Optimized event retrieval

## Best Practices

1. **Event Management**
   - Use appropriate event types
   - Include relevant metadata
   - Implement proper error handling
   - Regular dead letter queue processing

2. **Schema Management**
   - Define clear event schemas
   - Version schemas appropriately
   - Validate events before processing
   - Handle schema evolution

## Implementation Notes

- Events are stored in SQLite tables
- Dead letter queue tracks failed events
- Batch processing for bulk operations
- Schema validation is optional but recommended

## Testing

Testing requirements:

- Unit tests for event operations
- Batch processing tests
- Dead letter queue tests
- Schema validation tests
- Performance benchmarks

## Contributing

Guidelines for contributing:

1. Follow Python PEP 8 style guide
2. Include docstrings for all public methods
3. Add unit tests for new functionality
4. Update documentation for API changes
5. Use type hints for all function signatures
