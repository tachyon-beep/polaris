# Storage Plugin Base Interfaces

Core interfaces that define the storage plugin architecture for the Polaris graph database system.

## Overview

The base interfaces define the contract that all storage plugins must implement. They provide a consistent API for storing and retrieving nodes and edges, ensuring that different storage implementations (JSON, SQLite, etc.) can be used interchangeably.

## Directory Structure

```
base/
├── __init__.py      # Package initialization
└── interfaces.py    # Core interface definitions
```

## Components

### Storage Plugin Interface (`interfaces.py`)

Base storage plugin interfaces:

- **Purpose**: Define the contract for storage plugins
- **Key Features**:
  - Generic storage operations
  - Async interface
  - Type safety with generics
  - Consistent error handling
- **Key Interfaces**:
  - `StoragePlugin`: Generic base interface
  - `NodeStoragePlugin`: Node storage interface
  - `EdgeStoragePlugin`: Edge storage interface

Example usage:
```python
from polaris.infrastructure.storage.plugins.base.interfaces import NodeStoragePlugin

class CustomNodeStorage(NodeStoragePlugin):
    async def initialize(self) -> None:
        # Implementation
        pass

    async def create_node(self, node: Node) -> Node:
        # Implementation
        pass

    # Implement other required methods
```

## Common Features

1. **Core Storage Operations**
   - Initialization and cleanup
   - Backup and restore
   - CRUD operations
   - Query operations

2. **Type Safety**
   - Generic type parameters
   - Type hints for all methods
   - Runtime type checking support

3. **Error Handling**
   - Standard exception types
   - Error propagation patterns
   - Recovery mechanisms

## Usage Examples

### Implementing a Node Storage Plugin
```python
from polaris.infrastructure.storage.plugins.base.interfaces import NodeStoragePlugin
from polaris.core.models import Node
from polaris.core.exceptions import NodeNotFoundError

class CustomNodeStorage(NodeStoragePlugin):
    def __init__(self, storage_dir: str):
        super().__init__(storage_dir)
        
    async def initialize(self) -> None:
        # Setup storage
        pass

    async def create_node(self, node: Node) -> Node:
        # Store node
        pass

    async def get_node(self, name: str) -> Node:
        # Retrieve node
        pass

    # Implement other required methods
```

### Implementing an Edge Storage Plugin
```python
from polaris.infrastructure.storage.plugins.base.interfaces import EdgeStoragePlugin
from polaris.core.models import Edge
from polaris.core.enums import RelationType

class CustomEdgeStorage(EdgeStoragePlugin):
    def __init__(self, storage_dir: str):
        super().__init__(storage_dir)
        
    async def initialize(self) -> None:
        # Setup storage
        pass

    async def create_edge(self, edge: Edge) -> Edge:
        # Store edge
        pass

    async def get_edge(
        self, 
        from_entity: str, 
        to_entity: str, 
        relation_type: RelationType
    ) -> Edge:
        # Retrieve edge
        pass

    # Implement other required methods
```

## Architecture

1. **Plugin Architecture**: Defines base interfaces for extensible storage implementations
2. **Async First**: All operations are async for scalability
3. **Type Safety**: Uses generics and type hints for safety
4. **Error Handling**: Standardized error handling patterns

## Dependencies

- Internal:
  - `core.models`: Data models
  - `core.exceptions`: Error types
  - `core.enums`: Type enumerations
- External:
  - Python 3.7+ for dataclasses and async support

## Error Handling

Standard exceptions defined for common scenarios:

- `StorageError`: Base class for storage errors
- `NodeNotFoundError`: When node doesn't exist
- `EdgeNotFoundError`: When edge doesn't exist

Example:
```python
async def get_node(self, name: str) -> Node:
    try:
        # Attempt to get node
        if not exists:
            raise NodeNotFoundError(f"Node not found: {name}")
        return node
    except Exception as e:
        raise StorageError(f"Failed to get node: {str(e)}")
```

## Performance Considerations

1. **Interface Design**
   - Async operations for scalability
   - Batch operation support
   - Query optimization patterns

2. **Implementation Guidelines**
   - Connection pooling recommendations
   - Caching strategies
   - Resource management

## Best Practices

1. **Plugin Implementation**
   - Implement all abstract methods
   - Follow error handling patterns
   - Maintain type safety
   - Document implementation details

2. **Resource Management**
   - Proper initialization/cleanup
   - Connection handling
   - Error recovery

## Implementation Notes

- All methods are async
- Generic type parameters for flexibility
- Standard backup/restore interface
- Common filtering patterns

## Testing

Testing requirements for implementations:

- Interface compliance tests
- Error handling tests
- Performance benchmarks
- Resource cleanup tests
- Concurrency tests

## Contributing

Guidelines for contributing:

1. Follow Python PEP 8 style guide
2. Include docstrings for all methods
3. Add unit tests for new features
4. Update documentation
5. Use type hints consistently
