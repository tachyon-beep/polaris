# Graph Path Finding Package

This package provides efficient implementations of various path finding algorithms for traversing and analyzing paths between nodes in knowledge graphs.

## Features

- Multiple path finding strategies:
  - Dijkstra's algorithm for shortest paths
  - Bidirectional search for efficient path finding
  - Depth-first search for finding all possible paths
- Support for custom weight functions and path filtering
- Built-in path validation and optimization
- Performance monitoring and caching
- Extensible design for adding new algorithms

## Usage

```python
from src.core.graph_paths import PathFinding, PathType

# Find shortest path
result = PathFinding.shortest_path(graph, "A", "B")

# Use bidirectional search with custom weights
result = PathFinding.bidirectional_search(
    graph, "A", "B",
    weight_func=lambda e: e.metadata.weight
)

# Find all paths with constraints
paths = PathFinding.all_paths(
    graph, "A", "B",
    max_length=5,
    filter_func=lambda edges: sum(e.weight for e in edges) < 10
)

# Use generic interface
result = PathFinding.find_paths(
    graph, "A", "B",
    path_type=PathType.BIDIRECTIONAL,
    max_length=5,
    weight_func=lambda e: e.metadata.weight
)
```

## Package Structure

- `__init__.py`: Public API and factory class
- `models.py`: Data models for paths and metrics
- `utils.py`: Common utility functions
- `cache.py`: Caching functionality
- `base.py`: Abstract base class for algorithms
- `algorithms/`: Path finding implementations
  - `shortest_path.py`: Dijkstra's algorithm
  - `bidirectional.py`: Bidirectional search
  - `all_paths.py`: All paths finder

## Extending

To add a new path finding algorithm:

1. Create a new file in `algorithms/`
2. Implement the `PathFinder` interface
3. Add the algorithm to `PathType` enum
4. Update the `PathFinding.find_paths()` factory method

Example:

```python
from src.core.graph_paths.base import PathFinder
from src.core.graph_paths.models import PathResult

class CustomPathFinder(PathFinder):
    def find_path(self, start_node, end_node, **kwargs) -> PathResult:
        # Implement custom path finding logic
        pass
```

## Performance

The package includes built-in performance monitoring and caching:

- Metrics tracking for each operation
- LRU cache with TTL support
- Configurable cache size and expiration
- Performance statistics via `get_cache_metrics()`
