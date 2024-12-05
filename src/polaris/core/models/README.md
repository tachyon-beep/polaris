# Core Models Package

This package provides the fundamental data structures and models that represent nodes, edges, and their metadata in the knowledge graph system.

## Structure

```
models/
├── __init__.py  - Public API and exports
├── base.py      - Common utilities and shared code
├── node.py      - Node-related models
├── edge.py      - Edge-related models
├── metadata.py  - Shared metadata utilities
└── README.md    - This documentation
```

## Components

### Base Module (`base.py`)
Contains common utilities and validation functions used across the models:
- `validate_date_order()` - Ensures temporal consistency
- `validate_metric_range()` - Validates metric values within ranges
- `validate_custom_metrics()` - Validates custom metric definitions

### Node Models (`node.py`)
Defines the core node-related models:
- `Node` - Base node model representing vertices in the graph
- `NodeMetadata` - Administrative and tracking information for nodes
- `NodeMetrics` - Quality and performance metrics for nodes

### Edge Models (`edge.py`)
Defines the core edge-related models:
- `Edge` - Base edge model representing connections between nodes
- `EdgeMetadata` - Administrative and qualitative information for edges

### Metadata Utilities (`metadata.py`)
Provides shared metadata functionality:
- `MetadataProvider` - Protocol defining metadata-enabled entities
- `BaseMetadata` - Common metadata functionality for all entities

## Usage

```python
from polaris.core.models import Node, Edge, NodeMetadata, EdgeMetadata

# Create a node with metadata
node_metadata = NodeMetadata(
    created_at=datetime.now(),
    last_modified=datetime.now(),
    version=1,
    author="user",
    source="system"
)

node = Node(
    name="example_node",
    entity_type=EntityType.CONCEPT,
    observations=["Initial observation"],
    metadata=node_metadata
)

# Create an edge with metadata
edge_metadata = EdgeMetadata(
    created_at=datetime.now(),
    last_modified=datetime.now(),
    confidence=0.95,
    source="system"
)

edge = Edge(
    from_entity="source_node",
    to_entity="target_node",
    relation_type=RelationType.DEPENDS_ON,
    metadata=edge_metadata,
    impact_score=0.8
)
```

## Features

- **Type Safety**: All models use Python's type hints and dataclasses for type safety
- **Validation**: Automatic validation of data through `__post_init__` hooks
- **Extensibility**: Support for custom attributes and metrics
- **Metadata Tracking**: Built-in temporal and administrative metadata
- **Quality Metrics**: Integrated quality and performance metrics

## Best Practices

1. Always use the provided validation utilities when extending models
2. Maintain temporal consistency in metadata (created_at ≤ last_modified)
3. Use custom attributes and metrics judiciously
4. Follow the MetadataProvider protocol when creating new entity types
5. Validate metric ranges using the provided utilities

## Dependencies

- Python 3.7+
- dataclasses (built-in from Python 3.7+)
- typing (built-in)
- datetime (built-in)

## Contributing

When adding new models or extending existing ones:
1. Follow the established pattern of separating concerns
2. Add appropriate validation in `__post_init__`
3. Update the public API in `__init__.py`
4. Document new functionality in this README
5. Add type hints and docstrings
