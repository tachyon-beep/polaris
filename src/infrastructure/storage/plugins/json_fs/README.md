# JSON Filesystem Storage Plugin

A storage plugin implementation that uses JSON files for persistent storage of nodes and edges in the Polaris graph database system.

## Overview

The JSON Filesystem Storage Plugin provides a simple, file-based storage solution using JSON files. It's ideal for development, testing, and small to medium-sized graphs where simplicity and human-readability of storage are priorities.

## Directory Structure

```
json_fs/
├── __init__.py          # Plugin initialization and exports
├── constants.py         # Storage constants and file names
├── storage/            # Core storage implementations
│   ├── __init__.py     # Storage module initialization
│   ├── node_storage.py # Node storage implementation
│   └── edge_storage.py # Edge storage implementation
└── utils/              # Utility functions
    ├── __init__.py     # Utils module initialization
    ├── filtering.py    # Query filtering utilities
    ├── keys.py         # Key generation utilities
    └── persistence.py  # File persistence utilities
```

## Components

### Node Storage (`storage/node_storage.py`)

JSON file-based implementation of node storage:

- **Purpose**: Manages persistent storage of nodes in JSON format
- **Key Features**:
  - In-memory caching with JSON file persistence
  - CRUD operations for nodes
  - Query filtering support
  - Atomic file operations
- **Key Methods**:
  - `create_node()`: Create a new node
  - `get_node()`: Retrieve a node by name
  - `update_node()`: Update an existing node
  - `delete_node()`: Delete a node
  - `list_nodes()`: Query nodes with filtering

Example usage:
```python
from polaris.infrastructure.storage.plugins.json_fs.storage import JsonNodeStorage

# Initialize storage
storage = JsonNodeStorage("/path/to/storage")
await storage.initialize()

# Create a node
node = Node(name="example", entity_type=EntityType.CONCEPT)
created_node = await storage.create_node(node)
```

### Edge Storage (`storage/edge_storage.py`)

JSON file-based implementation of edge storage:

- **Purpose**: Manages persistent storage of edges in JSON format
- **Key Features**:
  - In-memory caching with JSON file persistence
  - CRUD operations for edges
  - Query filtering support
  - Atomic file operations
- **Key Methods**:
  - `create_edge()`: Create a new edge
  - `get_edge()`: Retrieve an edge by components
  - `update_edge()`: Update an existing edge
  - `delete_edge()`: Delete an edge
  - `list_edges()`: Query edges with filtering

Example usage:
```python
from polaris.infrastructure.storage.plugins.json_fs.storage import JsonEdgeStorage

# Initialize storage
storage = JsonEdgeStorage("/path/to/storage")
await storage.initialize()

# Create an edge
edge = Edge(from_entity="node1", to_entity="node2", relation_type=RelationType.DEPENDS_ON)
created_edge = await storage.create_edge(edge)
```

## Common Features

1. **JSON File Persistence**
   - Data stored in human-readable JSON format
   - Atomic file operations for data integrity
   - Automatic backup and restore capabilities

2. **In-Memory Caching**
   - Fast read operations with in-memory cache
   - Cache synchronization with file system
   - Thread-safe operations

3. **Query Filtering**
   - Flexible attribute-based filtering
   - Support for nested attribute queries
   - Pagination support

## Usage Examples

### Basic Node Operations
```python
# Initialize storage
storage = JsonNodeStorage("/path/to/storage")
await storage.initialize()

# Create a node
node = Node(
    name="concept1",
    entity_type=EntityType.CONCEPT,
    attributes={"key": "value"}
)
created = await storage.create_node(node)

# Query nodes
nodes = await storage.list_nodes(
    filters={"entity_type": EntityType.CONCEPT},
    limit=10,
    offset=0
)
```

### Basic Edge Operations
```python
# Initialize storage
storage = JsonEdgeStorage("/path/to/storage")
await storage.initialize()

# Create an edge
edge = Edge(
    from_entity="node1",
    to_entity="node2",
    relation_type=RelationType.DEPENDS_ON
)
created = await storage.create_edge(edge)

# Get edges for a node
edges = await storage.get_edges_for_node("node1")
```

## Architecture

1. **File-Based Storage**: Uses JSON files for persistence, providing human-readable storage
2. **In-Memory Cache**: Maintains an in-memory cache for performance
3. **Thread Safety**: All operations are thread-safe using asyncio locks
4. **Atomic Operations**: File operations are atomic to prevent data corruption

## Dependencies

- Internal:
  - `core.models`: Data models
  - `core.exceptions`: Error types
  - `core.enums`: Type enumerations
- External:
  - `aiofiles`: Async file operations
  - Python 3.7+ for dataclasses

## Error Handling

Custom exceptions for common scenarios:

- `NodeNotFoundError`: When accessing non-existent nodes
- `EdgeNotFoundError`: When accessing non-existent edges
- `StorageError`: For general storage operations failures

Example:
```python
try:
    node = await storage.get_node("missing")
except NodeNotFoundError:
    # Handle missing node case
except StorageError as e:
    # Handle general storage errors
```

## Performance Considerations

1. **Memory Usage**
   - All data is cached in memory
   - Consider available RAM for large datasets
   - Regular cleanup recommended

2. **File I/O**
   - Batch operations for better performance
   - Use appropriate pagination for large queries
   - Consider SSD storage for better performance

## Best Practices

1. **Initialization**
   - Always call initialize() after creating storage instance
   - Handle cleanup properly when done
   - Use backup() for important data

2. **Query Optimization**
   - Use appropriate filters to reduce result sets
   - Implement pagination for large queries
   - Cache frequently accessed data at application level

## Implementation Notes

- JSON files are stored in the specified storage directory
- Files are created automatically if they don't exist
- Backup files maintain the same structure as primary files
- Thread safety is handled at the storage instance level

## Testing

Testing requirements:

- Unit tests for all CRUD operations
- Integration tests for file operations
- Concurrency tests for thread safety
- Backup/restore functionality tests

## Contributing

Guidelines for contributing:

1. Follow Python PEP 8 style guide
2. Include docstrings for all public methods
3. Add unit tests for new functionality
4. Update documentation for API changes
5. Use type hints for all function signatures
