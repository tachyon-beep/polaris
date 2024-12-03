# Storage System

This directory contains the storage system implementation for the knowledge graph, providing persistent storage capabilities for nodes and edges.

## Components

### Base Storage (`base.py`)
Provides foundational storage functionality used by all storage implementations:
- Persistent storage and retrieval of data
- Thread-safe operations with asyncio locks
- JSON serialization/deserialization with datetime support
- Backup and restore functionality
- Error handling and logging
- Generic type support

### Node Storage (`node_storage.py`)
Specialized storage implementation for knowledge graph nodes:
- CRUD operations for nodes
- Node filtering and pagination
- Dependency tracking and resolution
- Type-based node queries
- Thread-safe operations
- Automatic timestamp management
- Node validation

### Edge Storage (`edge_storage.py`)
Specialized storage implementation for knowledge graph edges:
- CRUD operations for edges
- Edge filtering and pagination
- Type-based edge queries
- Node-centric edge queries
- Thread-safe operations
- Automatic timestamp management
- Edge validation

### Storage Service (`service.py`)
High-level service coordinating node and edge storage operations:
- Pluggable storage backends
- Unified interface for storage operations
- Backup and restore functionality
- Resource cleanup and initialization
- Error handling and logging
- Storage type configuration

### Plugins Directory (`/plugins`)
Contains pluggable storage backend implementations:

#### Base Plugin (`plugins/base.py`)
- Interface definitions for storage plugins
- Common plugin functionality
- Plugin registration system

#### JSON File System (`plugins/json_fs.py`)
- File-based storage using JSON
- Directory structure management
- File locking mechanisms
- Atomic write operations

#### SQLite (`plugins/sqlite.py`)
- SQLite database storage backend
- Connection pooling
- Transaction management
- Query optimization

#### Event Store (`plugins/event_store.py`)
- Event storage for event bus
- Event schema management
- Dead letter queue
- Event batch tracking

#### Utilities (`plugins/utils.py`)
- Common utility functions
- Data conversion helpers
- Storage-specific validators

## Usage

### Basic Usage

```python
# Initialize storage service
storage_service = StorageService(
    storage_dir="data",
    storage_type="json"  # or "sqlite"
)

# Initialize storage
await storage_service.initialize()

# Create node
node = Node(
    name="example",
    node_type="concept",
    metadata=NodeMetadata()
)
created_node = await storage_service.create_node(node)

# Create edge
edge = Edge(
    from_entity="source",
    to_entity="target",
    relation_type=RelationType.CONTAINS,
    metadata=EdgeMetadata()
)
created_edge = await storage_service.create_edge(edge)
```

### Using Different Storage Backends

```python
# JSON storage
json_storage = StorageService(storage_type="json")

# SQLite storage
sqlite_storage = StorageService(storage_type="sqlite")
```

### Backup and Restore

```python
# Create backup
await storage_service.backup("backup_directory")

# Restore from backup
await storage_service.restore_backup("backup_directory")
```

## Error Handling

The storage system uses custom exceptions:
- `StorageError`: Base exception for storage operations
- `NodeNotFoundError`: When requested node doesn't exist
- `EdgeNotFoundError`: When requested edge doesn't exist
- `ValidationError`: When data validation fails

Example:
```python
try:
    node = await storage_service.get_node("nonexistent")
except NodeNotFoundError:
    logger.error("Node not found")
except StorageError as e:
    logger.error(f"Storage error: {str(e)}")
```

## Thread Safety

All storage operations are thread-safe:
- Base storage uses asyncio locks
- File operations use proper locking mechanisms
- Database operations use connection pooling
- Atomic operations where possible

## Performance Optimization

The storage system includes several optimizations:
1. Connection pooling for database backends
2. Batch operations for multiple items
3. Efficient indexing strategies
4. Lazy loading of related data
5. Query optimization

## Extending Storage

### Creating New Storage Backend

1. Implement the base plugin interfaces:
```python
class CustomStoragePlugin(NodeStoragePlugin, EdgeStoragePlugin):
    async def initialize(self) -> None:
        # Implementation
        pass

    async def cleanup(self) -> None:
        # Implementation
        pass

    # Implement other required methods
```

2. Register the plugin:
```python
storage_service = StorageService(
    storage_type="custom",
    custom_plugin=CustomStoragePlugin()
)
```

### Adding New Features

1. Add methods to base storage
2. Implement in specific storage classes
3. Expose through storage service
4. Update plugin interfaces if needed

## Testing

Run storage tests:
```bash
pytest tests/storage/
```

Key test areas:
- CRUD operations
- Concurrent access
- Error conditions
- Data consistency
- Backup/restore
- Plugin functionality

## Maintenance

Regular maintenance tasks:
1. Backup verification
2. Index optimization
3. Storage cleanup
4. Performance monitoring
5. Error log review

## Dependencies

- Core layer (`src/core`)
- External libraries:
  - `aiofiles` for async file operations
  - `sqlite3` for SQLite backend
  - `jsonschema` for data validation
