# Core Package

The core package provides the fundamental functionality for working with knowledge graphs, including graph operations, traversal algorithms, metrics calculation, and domain models.

## Overview

This package serves as the foundation of the Polaris graph database system, implementing essential data structures and algorithms for graph operations. It provides a robust framework for representing and manipulating knowledge graphs while ensuring efficiency, type safety, and data validation.

## Directory Structure

```
core/
├── __init__.py          # Package initialization
├── models.py           # Core data structures and models
├── enums.py           # Type definitions and enumerations
├── graph.py           # Core graph implementation
├── graph_traversal.py # Graph traversal algorithms
├── graph_paths.py     # Path finding algorithms
├── graph_components.py # Component analysis
├── graph_metrics.py   # Metrics calculation
├── graph_subgraphs.py # Subgraph operations
└── exceptions.py      # Custom exceptions
```

## Components

### Domain Models (`models.py`)

- **Purpose**: Defines core data structures for the knowledge graph
- **Key Features**:
  - Entity model for graph nodes with comprehensive validation
  - Relation model for graph edges with validation status tracking
  - Extensive metadata support with automatic validation
  - Quality metrics integration with range validation
  - Custom metric support with configurable ranges
  - Dependency tracking and observation logging
  - Temporal metadata tracking
- **Key Classes**:
  - `Node`: Base model for nodes
    - Supports observations and dependencies
    - Includes validation rules and examples
    - Tracks code references and documentation
  - `NodeMetrics`: Quality metrics with custom metric support
  - `NodeMetadata`: Administrative metadata with temporal tracking
  - `Edge`: Base model for edges
    - Includes validation status tracking
    - Supports custom metrics with range validation
    - Tracks impact scores and context
  - `EdgeMetadata`: Relation metadata with confidence scoring

### Graph Core (`graph.py`)

- **Purpose**: Implements the fundamental graph data structure
- **Key Features**:
  - Efficient adjacency list representation
  - Neighbor lookup optimization
  - Relation management
  - Basic graph operations
  - Edge validation and verification
- **Key Methods**:
  - `add_relation()`: Add new relations with validation
  - `get_neighbors()`: Retrieve connected nodes efficiently
  - `remove_relation()`: Remove existing relations safely
  - `get_edge_safe()`: Get edges with validation

### Graph Operations

#### Traversal (`graph_traversal.py`)
- **Purpose**: Implements graph traversal algorithms
- **Key Features**:
  - BFS and DFS implementations
  - Depth limiting
  - Custom filters
  - Traversal statistics
  - Performance optimizations

#### Path Finding (`graph_paths.py`)
- **Purpose**: Implements path analysis algorithms
- **Key Features**:
  - Dijkstra's algorithm
  - Custom weight functions
  - All-paths discovery
  - Path filtering
  - Path validation
  - Result encapsulation

#### Component Analysis (`graph_components.py`)
- **Purpose**: Analyzes graph components and structure
- **Key Features**:
  - Connected component detection
  - Strongly connected components (Tarjan's algorithm)
  - Clustering coefficient calculation
  - Component-based metrics

#### Metrics Calculation (`graph_metrics.py`)
- **Purpose**: Calculates and analyzes graph metrics
- **Key Features**:
  - Basic metrics (node count, edge count, density)
  - Advanced metrics (clustering, path length, centrality)
  - Automatic selection of calculation strategies:
    - Exact calculations for small graphs
    - Approximate calculations with confidence levels for large graphs
  - Performance-optimized implementations
  - Comprehensive metric result encapsulation
- **Key Metrics**:
  - Node and edge counts
  - Average degree and density
  - Clustering coefficients
  - Path metrics (diameter, average length)
  - Centrality scores
  - Component analysis
  - Custom metric support

## Common Features

1. **Type Safety and Validation**
   - Strong typing throughout codebase
   - Runtime type checking
   - Custom type definitions
   - Comprehensive data validation
   - Range validation for metrics

2. **Performance Optimization**
   - Efficient data structures
   - Optimized algorithms
   - Automatic strategy selection based on graph size
   - Caching strategies
   - Sampling techniques for large graphs

3. **Error Handling**
   - Custom exception hierarchy
   - Detailed error messages
   - Recovery mechanisms
   - Validation status tracking

## Usage Examples

### Creating a Graph with Validation
```python
from core import Graph, Node, Edge, NodeMetadata, EdgeMetadata
from core.enums import EntityType, RelationType
from datetime import datetime

# Create node with metadata
node_metadata = NodeMetadata(
    created_at=datetime.now(),
    last_modified=datetime.now(),
    version=1,
    author="john.doe",
    source="static_analysis"
)

node = Node(
    name="UserService",
    entity_type=EntityType.CODE_MODULE,
    observations=["Handles user authentication"],
    metadata=node_metadata,
    documentation="Service for user management"
)

# Create edge with metadata and validation
edge_metadata = EdgeMetadata(
    created_at=datetime.now(),
    last_modified=datetime.now(),
    confidence=0.95,
    source="static_analysis"
)

edge = Edge(
    from_entity="UserService",
    to_entity="DatabaseService",
    relation_type=RelationType.DEPENDS_ON,
    metadata=edge_metadata,
    impact_score=0.8,
    validation_status="verified"
)

# Initialize graph
graph = Graph([edge])
```

### Calculating Metrics
```python
from core.graph_metrics import MetricsCalculator

# Calculate comprehensive metrics
metrics = MetricsCalculator.calculate_metrics(graph)

# Access various metrics
print(f"Nodes: {metrics.node_count}")
print(f"Edges: {metrics.edge_count}")
print(f"Clustering: {metrics.clustering_coefficient:.2f}")
print(f"Average Path Length: {metrics.average_path_length:.2f}")
```

## Architecture

1. **Modularity**
   - Clear separation of concerns
   - Independent components
   - Well-defined interfaces
   - Comprehensive validation

2. **Extensibility**
   - Plugin architecture
   - Custom algorithm support
   - Flexible data models
   - Custom metric support

3. **Reliability**
   - Comprehensive error handling
   - Data validation
   - State management
   - Status tracking

## Dependencies

- Internal:
  - None (core package)
- External:
  - Type hints from `typing`
  - Dataclasses from `dataclasses`
  - JSON handling from `json`
  - Math functions from `math`

## Error Handling

Custom exceptions defined in `exceptions.py`:
```python
try:
    # Attempt graph operation
    path = graph.get_edge_safe("A", "B")
except NodeNotFoundError as e:
    logger.error(f"Node not found: {e}")
except EdgeNotFoundError as e:
    logger.error(f"Edge not found: {e}")
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
```

## Performance Considerations

1. **Graph Operations**
   - Use adjacency lists for efficient neighbor lookup
   - Implement lazy loading for large graphs
   - Cache frequently accessed paths
   - Automatic strategy selection based on graph size

2. **Memory Management**
   - Optimize data structures for memory usage
   - Implement cleanup for unused resources
   - Use generators for large result sets
   - Efficient sampling for large graphs

## Best Practices

1. **Entity Management**
   - Use meaningful entity names
   - Provide comprehensive metadata
   - Maintain consistent entity types
   - Track dependencies and observations

2. **Relation Definition**
   - Use appropriate relation types
   - Include relevant context
   - Consider bidirectional relationships
   - Track validation status

3. **Graph Operations**
   - Choose appropriate traversal methods
   - Consider performance implications
   - Implement proper error handling
   - Validate operations and data

## Implementation Notes

- Graph implementation is undirected by default
- Operations consider entire graph unless filtered
- Memory usage scales with graph size
- Automatic selection of calculation strategies
- Comprehensive validation throughout
- Status tracking for edges and operations

## Testing

- Unit tests for all components
- Integration tests for graph operations
- Performance benchmarks
- Edge case coverage
- Concurrent operation testing
- Validation testing

## Contributing

1. Follow type hinting conventions
2. Add comprehensive docstrings
3. Include unit tests
4. Update documentation
5. Consider performance implications
6. Implement proper validation
