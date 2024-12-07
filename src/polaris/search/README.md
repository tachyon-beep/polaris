# Search Module

The search module provides comprehensive search capabilities for the Polaris graph database, including basic, semantic, and graph-based search functionality.

## Overview

This module implements advanced search capabilities that enable users to find and analyze data within the knowledge graph. It supports various search types including text-based, semantic, and graph-based searches, with features for filtering, sorting, and result aggregation.

## Directory Structure

```
search/
├── __init__.py          # Package initialization
├── semantic.py          # Semantic search implementation
└── query/              # Query components
    ├── __init__.py     # Query package initialization
    ├── base.py         # Base query model
    ├── filters.py      # Search filters
    ├── graph.py        # Graph query model
    ├── results.py      # Results container
    ├── semantic.py     # Semantic query model
    └── sorting.py      # Result sorting
```

## Components

### Semantic Search Engine (`semantic.py`)

- **Purpose**: Provides semantic search using vector embeddings
- **Key Features**:
  - Vector embedding generation
  - Cosine similarity matching
  - Semantic operator support
  - Embedding caching
  - Concurrent processing
- **Key Methods**:
  - `search()`: Execute semantic search
  - `generate_embeddings()`: Create vector embeddings
  - `calculate_similarity()`: Compute similarities

### Query Models (`query/`)

#### Base Query (`base.py`)
- **Purpose**: Foundation for all search queries
- **Key Features**:
  - Text-based search
  - Filtering
  - Sorting
  - Pagination
  - Metadata inclusion
- **Key Methods**:
  - `execute()`: Run the query
  - `validate()`: Validate query parameters
  - `paginate()`: Handle result pagination

#### Filters (`filters.py`)
- **Purpose**: Define search filtering operations
- **Key Features**:
  - Comparison operations
  - Range filtering
  - Pattern matching
  - Type checking
  - Regular expressions
- **Key Methods**:
  - `apply()`: Apply filter to results
  - `validate()`: Validate filter parameters
  - `combine()`: Combine multiple filters

[Additional components follow same structure...]

## Common Features

1. **Query Processing**
   - Query validation
   - Parameter normalization
   - Result formatting

2. **Performance Optimization**
   - Result caching
   - Query optimization
   - Concurrent processing

3. **Error Handling**
   - Query validation
   - Error recovery
   - Result verification

## Usage Examples

### Basic Search
```python
from polaris.search.query import SearchQuery, SearchFilter, SearchSort

# Create search query
query = SearchQuery(
    query_text="example",
    filters=[
        SearchFilter(field="type", operator="eq", value="document")
    ],
    sort=SearchSort(field="created_at", direction="desc"),
    page=1,
    page_size=50
)

# Execute search
results = await search_engine.search(query)
```

### Semantic Search
```python
from polaris.search.query import SemanticQuery
from polaris.search import SemanticSearchEngine

# Initialize engine
engine = SemanticSearchEngine(storage, embedding_service)

# Create query
query = SemanticQuery(
    query_text="Find documents about machine learning",
    similarity_threshold=0.7,
    semantic_operators=["expand_synonyms", "include_related"]
)

# Execute search
results = await engine.search(query)
```

[Additional examples follow same pattern...]

## Architecture

1. **Modularity**
   - Independent search types
   - Pluggable components
   - Extensible design

2. **Scalability**
   - Concurrent processing
   - Result streaming
   - Resource management

3. **Flexibility**
   - Custom query types
   - Extensible filters
   - Custom sorting

## Dependencies

- Internal:
  - Core models
  - Storage service
  - Cache service
- External:
  - Vector embedding libraries
  - Text processing utilities
  - Concurrent processing tools

## Error Handling

```python
try:
    results = await engine.search(query)
except QueryError as e:
    logger.error(f"Invalid query: {e}")
except StorageError as e:
    logger.error(f"Storage error: {e}")
```

## Performance Considerations

1. **Query Optimization**
   - Filter ordering
   - Index utilization
   - Result caching

2. **Resource Management**
   - Connection pooling
   - Memory usage
   - Concurrent requests

## Best Practices

1. **Query Construction**
   - Validate inputs
   - Use appropriate filters
   - Set reasonable limits

2. **Result Handling**
   - Implement pagination
   - Handle large results
   - Process asynchronously

3. **Error Management**
   - Validate early
   - Handle timeouts
   - Provide feedback

## Implementation Notes

- Queries are immutable once created
- Results are paginated by default
- Semantic search requires embeddings
- Filters are composable
- Sorting affects performance

## Testing

- Unit tests for components
- Integration tests
- Performance benchmarks
- Query validation tests
- Result verification

## Contributing

1. Follow query patterns
2. Add comprehensive tests
3. Document new features
4. Consider performance
5. Maintain compatibility
