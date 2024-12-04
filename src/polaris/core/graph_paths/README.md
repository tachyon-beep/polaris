# Graph Paths Package

This package provides comprehensive path finding capabilities for the Polaris knowledge graph system. It includes both basic and advanced algorithms optimized for different use cases.

## Overview

The package is organized into several components:

```
graph_paths/
├── __init__.py
├── base.py         # Base classes and interfaces
├── models.py       # Data models and results
├── utils.py        # Shared utilities
└── algorithms/     # Algorithm implementations
    ├── __init__.py
    ├── shortest_path.py
    ├── all_paths.py
    ├── bidirectional.py
    └── advanced/   # Advanced algorithms
        ├── __init__.py
        ├── alt.py
        ├── contraction_hierarchies.py
        ├── hub_labeling.py
        └── transit_node_routing.py
```

## Features

- Memory-efficient path finding
- Multiple algorithm implementations
- Comprehensive path validation
- Performance monitoring
- Progress tracking
- Memory management

## Basic Algorithms

### Shortest Path
- Dijkstra's algorithm with decrease-key operation
- Priority queue optimization
- Early termination
- Memory-efficient implementation

### All Paths
- Depth-first enumeration
- Cycle detection
- Path filtering
- Memory usage monitoring

### Bidirectional Search
- Meets-in-the-middle approach
- Proper termination conditions
- Efficient path reconstruction
- Progress tracking

## Advanced Algorithms

### Contraction Hierarchies (CH)
- Preprocessing-based speedup technique
- Node contraction with shortcuts
- Importance-based ordering
- Memory-efficient implementation

Features:
- Fast distance queries
- Path unpacking
- Progress tracking
- Memory monitoring

Usage:
```python
from polaris.core.graph_paths.algorithms.advanced import ContractionHierarchies

ch = ContractionHierarchies(graph)
ch.preprocess()  # Build shortcuts
path = ch.find_path("A", "B")
```

### Hub Labeling (HL)
- Label-based distance computation
- Space-efficient label storage
- Fast distance queries
- Path reconstruction support

Features:
- Constant-time queries
- Memory-efficient labels
- Progress tracking
- Validation support

Usage:
```python
from polaris.core.graph_paths.algorithms.advanced import HubLabels

hl = HubLabels(graph)
hl.preprocess()  # Compute labels
path = hl.find_path("A", "B")
```

### Transit Node Routing (TNR)
- Access node computation
- Transit node selection
- Distance table computation
- Locality filter

Features:
- Very fast long-distance queries
- Access node optimization
- Memory usage monitoring
- Progress tracking

Usage:
```python
from polaris.core.graph_paths.algorithms.advanced import TransitNodeRouting

tnr = TransitNodeRouting(graph)
tnr.preprocess()  # Select transit nodes
path = tnr.find_path("A", "B")
```

### A* with Landmarks (ALT)
- Triangle inequality based heuristic
- Landmark selection strategies
- Bidirectional search support
- Memory-efficient implementation

Features:
- Better than basic A*
- Dynamic graph support
- Progress tracking
- Memory monitoring

Usage:
```python
from polaris.core.graph_paths.algorithms.advanced import ALTPathFinder

alt = ALTPathFinder(graph)
alt.preprocess()  # Select landmarks
path = alt.find_path("A", "B")
```

## Utilities

### Memory Management
```python
from polaris.core.graph_paths.utils import MemoryManager

# Monitor memory usage
with MemoryManager(max_mb=1000) as mm:
    # Your memory-intensive code here
    mm.check_memory()  # Raises if limit exceeded
```

### Path Validation
```python
from polaris.core.graph_paths.utils import validate_path

# Validate path properties
validate_path(
    path=path,
    graph=graph,
    weight_func=weight_func,
    max_length=max_length,
    allow_cycles=False
)
```

### Performance Monitoring
```python
from polaris.core.graph_paths.utils import timer

# Time operations
with timer("Path finding"):
    path = finder.find_path("A", "B")
```

## Algorithm Selection Guide

Choose the appropriate algorithm based on your use case:

1. **Contraction Hierarchies (CH)**
   - Best for: Static graphs with frequent queries
   - Pros: Very fast queries
   - Cons: High preprocessing cost, static graphs only

2. **Hub Labeling (HL)**
   - Best for: Dense graphs, many queries
   - Pros: Constant-time queries
   - Cons: High space requirement

3. **Transit Node Routing (TNR)**
   - Best for: Road networks, long-distance queries
   - Pros: Extremely fast for long distances
   - Cons: Not optimal for local queries

4. **A* with Landmarks (ALT)**
   - Best for: Dynamic graphs, sparse graphs
   - Pros: No major preprocessing, good heuristic
   - Cons: Not as fast as CH/HL for static graphs

## Best Practices

1. **Memory Management**
   - Monitor memory usage with MemoryManager
   - Use appropriate batch sizes
   - Clean up temporary data structures

2. **Performance Optimization**
   - Choose appropriate algorithm
   - Use progress tracking for long operations
   - Implement proper error handling

3. **Path Validation**
   - Validate paths after computation
   - Check for cycles if needed
   - Verify edge existence

## Error Handling

All algorithms provide detailed error messages for common issues:

- Invalid nodes
- Memory limits exceeded
- Path validation failures
- Preprocessing requirements

Example:
```python
try:
    path = finder.find_path("A", "B")
except MemoryError:
    print("Memory limit exceeded")
except GraphOperationError as e:
    print(f"Path finding failed: {e}")
```

## Contributing

When adding new algorithms:

1. Implement the PathFinder interface
2. Add comprehensive tests
3. Include memory management
4. Add progress tracking
5. Update documentation
