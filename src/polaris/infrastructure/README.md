# Infrastructure Layer

The infrastructure layer provides core services and implementations for data persistence, caching, event handling, and server operations in the Polaris graph database system.

## Overview

This layer implements the technical capabilities required to support the domain model and business logic, providing robust and scalable infrastructure services. It handles data persistence, caching, event processing, and server operations while maintaining thread safety and performance optimization.

## Directory Structure

```
infrastructure/
├── __init__.py          # Package initialization
├── server.py            # MCP server implementation
├── cache.py             # LRU cache implementation
├── event_bus.py         # Event handling system
└── storage/             # Storage system components
    ├── __init__.py      # Storage package initialization
    ├── base.py          # Base storage implementation
    ├── node_storage.py  # Node storage implementation
    ├── edge_storage.py  # Edge storage implementation
    ├── service.py       # Storage service coordination
    └── plugins/         # Storage backend plugins
        ├── __init__.py  # Plugins initialization
        ├── base/        # Base plugin interfaces
        │   ├── __init__.py
        │   └── interfaces.py
        ├── json_fs/     # JSON filesystem plugin
        │   ├── __init__.py
        │   ├── constants.py
        │   ├── storage/
        │   │   ├── node_storage.py
        │   │   └── edge_storage.py
        │   └── utils/
        │       ├── filtering.py
        │       ├── keys.py
        │       └── persistence.py
        ├── sqlite/      # SQLite database plugin
        │   ├── __init__.py
        │   ├── constants.py
        │   ├── storage/
        │   │   ├── node_storage.py
        │   │   └── edge_storage.py
        │   └── utils/
        │       ├── conversion.py
        │       ├── persistence.py
        │       └── queries.py
        ├── event_store/ # Event storage plugin
        │   ├── __init__.py
        │   ├── constants.py
        │   ├── models.py
        │   └── store.py
        └── utils/       # Shared plugin utilities
            ├── __init__.py
            ├── serialization.py
            └── validation.py
```

## Components

### Server (`server.py`)

- **Purpose**: Provides standardized interface for knowledge graph operations
- **Key Features**:
  - Tool-based operation handling
  - Resource management
  - Node and edge operations
  - Error handling and logging
- **Key Methods**:
  - `handle_operation()`: Process incoming operations
  - `manage_resources()`: Handle resource lifecycle
  - `execute_tool()`: Run specific tools

### Cache (`cache.py`)

- **Purpose**: Provides thread-safe LRU caching with time-based expiration
- **Key Features**:
  - Generic object caching
  - Thread-safe operations
  - Time-based expiration
  - Custom serialization
  - Batch operations
- **Key Methods**:
  - `get()`: Retrieve cached item
  - `set()`: Store item in cache
  - `invalidate()`: Remove item from cache
  - `clear_expired()`: Clean up expired items

### Event Bus (`event_bus.py`)

- **Purpose**: Manages asynchronous event processing
- **Key Features**:
  - Async event processing
  - Event persistence
  - Dead letter queue
  - Event batching
  - Schema validation
- **Key Methods**:
  - `publish()`: Send new event
  - `subscribe()`: Register event handler
  - `replay()`: Replay past events
  - `handle_failure()`: Process failed events

### Storage System (`storage/`)

- **Purpose**: Provides persistent storage for graph data
- **Key Features**:
  - Pluggable storage backends
  - Node and edge storage
  - Query capabilities
  - Transaction support
- **Storage Plugins**:
  - **JSON Filesystem**: Simple file-based storage
  - **SQLite**: SQL database storage
  - **Event Store**: Event persistence and processing

## Common Features

1. **Thread Safety**
   - Concurrent operation support
   - Lock management
   - Race condition prevention

2. **Error Handling**
   - Custom exceptions
   - Error recovery
   - Logging and monitoring

3. **Performance Optimization**
   - Caching strategies
   - Batch processing
   - Connection pooling

## Usage Examples

### Storage Service
```python
# Initialize storage service
storage_service = StorageService(
    storage_dir="data",
    storage_type="json"
)
await storage_service.initialize()

# Store node
node = Node(name="example")
await storage_service.store_node(node)
```

### Event Bus
```python
# Initialize event bus
event_bus = EventBus(event_store=EventStore())

# Subscribe to events
async def handle_event(event):
    print(f"Processing event: {event}")
await event_bus.subscribe("node.created", handle_event)

# Publish event
await event_bus.publish("node.created", {"node_id": "123"})
```

## Architecture

1. **Modularity**
   - Self-contained components
   - Clear interfaces
   - Pluggable architecture

2. **Scalability**
   - Horizontal scaling support
   - Load distribution
   - Resource management

3. **Reliability**
   - Data persistence
   - Error recovery
   - State management

## Dependencies

- Internal:
  - Core package (`src/core`)
- External:
  - `asyncio` for async operations
  - `jsonschema` for validation
  - `aiofiles` for async I/O
  - `mcp` for server implementation
  - `sqlite3` for database operations

## Error Handling

```python
try:
    await storage_service.store_node(node)
except StorageError as e:
    logger.error(f"Storage operation failed: {e}")
    await event_bus.publish("storage.error", {"error": str(e)})
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
```

## Performance Considerations

1. **Caching Strategy**
   - LRU cache implementation
   - Cache invalidation
   - Memory management

2. **Batch Processing**
   - Event batching
   - Bulk operations
   - Queue management

## Best Practices

1. **Resource Management**
   - Proper initialization
   - Clean shutdown
   - Resource cleanup

2. **Error Handling**
   - Comprehensive error catching
   - Proper error propagation
   - Recovery procedures

3. **Configuration**
   - Environment-based config
   - Sensible defaults
   - Documentation

## Implementation Notes

- All components are thread-safe
- Async operations use asyncio
- Storage is pluggable
- Events are persistent
- Cache is size-limited

## Testing

- Unit tests for components
- Integration tests
- Performance benchmarks
- Concurrency tests
- Error handling tests

## Contributing

1. Follow async/await patterns
2. Maintain thread safety
3. Add comprehensive tests
4. Document APIs
5. Consider backward compatibility
