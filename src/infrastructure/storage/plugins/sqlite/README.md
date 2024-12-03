# SQLite Storage Plugin

A storage plugin implementation that uses SQLite for persistent storage of nodes and edges in the Polaris graph database system.

## Overview

The SQLite Storage Plugin provides a robust, SQL-based storage solution using SQLite as the backend database. It's ideal for production use cases where data integrity, complex querying, and ACID compliance are priorities.

## Directory Structure

```
sqlite/
├── __init__.py          # Plugin initialization and exports
├── constants.py         # SQL schemas and queries
├── storage/            # Core storage implementations
│   ├── __init__.py     # Storage module initialization
│   ├── node_storage.py # Node storage implementation
│   └── edge_storage.py # Edge storage implementation
└── utils/              # Utility functions
    ├── __init__.py     # Utils module initialization
    ├── conversion.py   # Data conversion utilities
    ├── persistence.py  # Database operations utilities
    └── queries.py      # SQL query building utilities
```

## Components

### Node Storage (`storage/node_storage.py`)

SQLite-based implementation of node storage:

- **Purpose**: Manages persistent storage of nodes in SQLite database
- **Key Features**:
  - ACID compliant storage operations
  - Efficient SQL-based querying
  - Transaction support
  - Built-in indexing
- **Key Methods**:
  - `create_node()`: Create a new node
  - `get_node()`: Retrieve a node by name
  - `update_node()`: Update an existing node
  - `delete_node()`: Delete a node
  - `list_nodes()`: Query nodes with filtering

Example usage:
```python
from polaris.infrastructure.storage.plugins.sqlite.storage import SqliteNodeStorage

# Initialize storage
storage = SqliteNodeStorage("/path/to/storage")
await storage.initialize()

# Create a node
node = Node(name="example", entity_type=EntityType.CONCEPT)
created_node = await storage.create_node(node)
```

### Edge Storage (`storage/edge_storage.py`)

SQLite-based implementation of edge storage:

- **Purpose**: Manages persistent storage of edges in SQLite database
- **Key Features**:
  - ACID compliant storage operations
  - Efficient SQL-based querying
  - Transaction support
  - Built-in indexing
- **Key Methods**:
  - `create_edge()`: Create a new edge
  - `get_edge()`: Retrieve an edge by components
  - `update_edge()`: Update an existing edge
  - `delete_edge()`: Delete an edge
  - `list_edges()`: Query edges with filtering

Example usage:
```python
from polaris.infrastructure.storage.plugins.sqlite.storage import SqliteEdgeStorage

# Initialize storage
storage = SqliteEdgeStorage("/path/to/storage")
await storage.initialize()

# Create an edge
edge = Edge(from_entity="node1", to_entity="node2", relation_type=RelationType.DEPENDS_ON)
created_edge = await storage.create_edge(edge)
```

## Common Features

1. **SQL Database Operations**
   - ACID compliant transactions
   - Efficient indexing and querying
   - Data integrity constraints
   - Atomic operations

2. **Query Building**
   - Dynamic SQL query generation
   - Parameterized queries for safety
   - Complex filtering support
   - Pagination optimization

3. **Data Conversion**
   - Automatic conversion between models and rows
   - JSON serialization for complex attributes
   - Type-safe conversions
   - Datetime handling

## Usage Examples

### Basic Node Operations
```python
# Initialize storage
storage = SqliteNodeStorage("/path/to/storage")
await storage.initialize()

# Create a node
node = Node(
    name="concept1",
    entity_type=EntityType.CONCEPT,
    attributes={"key": "value"}
)
created = await storage.create_node(node)

# Query nodes with filtering
nodes = await storage.list_nodes(
    filters={"entity_type": EntityType.CONCEPT},
    limit=10,
    offset=0
)
```

### Basic Edge Operations
```python
# Initialize storage
storage = SqliteEdgeStorage("/path/to/storage")
await storage.initialize()

# Create an edge
edge = Edge(
    from_entity="node1",
    to_entity="node2",
    relation_type=RelationType.DEPENDS_ON
)
created = await storage.create_edge(edge)

# Query edges by type
edges = await storage.get_edges_by_type(RelationType.DEPENDS_ON)
```

## Architecture

1. **SQLite Backend**: Uses SQLite for reliable, file-based database storage
2. **Async Operations**: All database operations are async using aiosqlite
3. **Transaction Safety**: ACID compliance for data integrity
4. **Query Optimization**: Efficient query building and execution

## Dependencies

- Internal:
  - `core.models`: Data models
  - `core.exceptions`: Error types
  - `core.enums`: Type enumerations
- External:
  - `aiosqlite`: Async SQLite operations
  - `sqlite3`: Python's built-in SQLite support
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

1. **Database Optimization**
   - Proper indexing for frequent queries
   - Connection pooling for concurrent access
   - Regular VACUUM for space optimization

2. **Query Performance**
   - Use appropriate WHERE clauses
   - Implement pagination for large results
   - Optimize complex joins

## Best Practices

1. **Database Management**
   - Regular backups
   - Proper cleanup on shutdown
   - Transaction management

2. **Query Patterns**
   - Use parameterized queries
   - Implement proper error handling
   - Manage connection lifecycle

## Implementation Notes

- SQLite database file is created automatically
- Indexes are created for frequently queried fields
- All operations are wrapped in transactions
- Automatic schema creation on initialization

## Testing

Testing requirements:

- Unit tests for CRUD operations
- Transaction tests
- Concurrency tests
- Performance benchmarks
- Error handling tests

## Contributing

Guidelines for contributing:

1. Follow Python PEP 8 style guide
2. Include docstrings for all public methods
3. Add unit tests for new functionality
4. Update documentation for API changes
5. Use type hints for all function signatures
