# Repository Layer

The repository layer implements the repository pattern for the Polaris graph database system, providing a clean separation between domain models and data access logic.

## Overview

This layer acts as an abstraction between domain models and data mapping layers, making the domain model completely independent of data access technologies. It provides an object-oriented view of the persistence layer, encapsulating data access logic and exposing a collection-like interface for accessing domain objects.

## Directory Structure

```
repositories/
├── __init__.py      # Package initialization
├── base.py         # Base repository implementation
├── entity.py       # Entity repository implementation
└── relation.py     # Relation repository implementation
```

## Components

### Base Repository (`base.py`)

- **Purpose**: Provides foundation for repository implementations
- **Key Features**:
  - Generic CRUD operations
  - LRU caching support
  - Abstract validation methods
  - Cache management
- **Key Methods**:
  - `create()`: Create new resource
  - `get()`: Retrieve resource
  - `update()`: Update existing resource
  - `delete()`: Remove resource
  - `list()`: List resources

### Entity Repository (`entity.py`)

- **Purpose**: Manages Entity objects in the knowledge graph
- **Key Features**:
  - Entity CRUD operations
  - Custom validation
  - Dependency management
  - Type-based queries
  - Caching optimization
- **Key Methods**:
  - `get_by_type()`: Retrieve entities by type
  - `get_with_dependencies()`: Get entity with dependencies
  - `register_validator()`: Add custom validation
  - `bulk_create()`: Create multiple entities
  - `bulk_update()`: Update multiple entities

### Relation Repository (`relation.py`)

- **Purpose**: Manages relationships between entities
- **Key Features**:
  - Relation CRUD operations
  - Graph operations support
  - Bidirectional querying
  - Custom validation
  - Caching optimization
- **Key Methods**:
  - `find_paths()`: Find entity paths
  - `get_connected_components()`: Get connected entities
  - `get_neighborhood()`: Get entity neighborhood
  - `get_entity_relations()`: Get entity relations

## Common Features

1. **Caching**
   - LRU cache implementation
   - Cache invalidation
   - Cache statistics

2. **Validation**
   - Built-in validation
   - Custom validators
   - Validation pipelines

3. **Async Operations**
   - Async/await pattern
   - Concurrent operations
   - Transaction support

## Usage Examples

### Entity Management
```python
# Create entity repository
entity_repo = EntityRepository(storage_service, cache)

# Create entity
entity = Entity(name="example", type="concept")
created = await entity_repo.create(entity)

# Get with dependencies
entity_graph = await entity_repo.get_with_dependencies("example")
```

### Relation Management
```python
# Create relation repository
relation_repo = RelationRepository(storage_service, cache)

# Create relation
relation = Relation(
    from_entity="A",
    to_entity="B",
    type=RelationType.CONTAINS
)
created = await relation_repo.create(relation)

# Find paths
paths = await relation_repo.find_paths("A", "C", max_depth=3)
```

### Custom Validation
```python
# Add custom validator
async def validate_entity_name(entity: Entity) -> bool:
    return len(entity.name) <= 100

entity_repo.register_validator(validate_entity_name)
```

## Architecture

1. **Repository Pattern**
   - Data access abstraction
   - Domain model isolation
   - Collection-like interface

2. **Caching Strategy**
   - Multi-level caching
   - Cache coherence
   - Invalidation policies

3. **Validation Framework**
   - Extensible validation
   - Rule composition
   - Error aggregation

## Dependencies

- Internal:
  - Core models (`Entity`, `Relation`)
  - Storage service
  - Cache service
- External:
  - `asyncio` for async operations
  - `typing` for type hints

## Error Handling

```python
try:
    entity = await entity_repo.get("example")
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
except ResourceNotFoundError as e:
    logger.error(f"Entity not found: {e}")
except StorageError as e:
    logger.error(f"Storage error: {e}")
```

## Performance Considerations

1. **Caching Strategy**
   - Cache frequently accessed data
   - Implement cache warming
   - Monitor cache hit rates

2. **Batch Operations**
   - Use bulk operations
   - Implement connection pooling
   - Optimize query patterns

## Best Practices

1. **Repository Usage**
   - Use repositories over direct storage
   - Implement custom validators
   - Handle errors appropriately

2. **Cache Management**
   - Monitor cache size
   - Implement proper invalidation
   - Use appropriate TTLs

3. **Validation**
   - Validate early
   - Use custom validators
   - Implement comprehensive rules

## Implementation Notes

- All operations are async
- Repositories are thread-safe
- Caching is optional but recommended
- Validation is extensible
- Transactions are supported

## Testing

- Unit tests for each repository
- Integration tests
- Cache behavior tests
- Validation tests
- Concurrency tests

## Contributing

1. Follow async patterns
2. Add appropriate validation
3. Update cache logic
4. Include comprehensive tests
5. Document new features
