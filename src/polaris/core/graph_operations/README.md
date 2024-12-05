# Graph Operations

This directory contains modular components for graph operations that extend the core Graph class functionality:

- components.py: Component analysis and connected component operations
- metrics.py: Graph metric calculations and analysis
- partitioning.py: Graph partitioning algorithms and utilities  
- serialization.py: Graph serialization/deserialization utilities
- subgraphs.py: Subgraph extraction and manipulation

Each module is designed to be imported and used by the main Graph class while maintaining separation of concerns.

## Usage

The modules in this directory are typically used through the Graph class interface rather than directly:

```python
from polaris.core.graph import Graph

# Graph instance automatically integrates these operations
graph = Graph()

# Component operations
components = graph.get_components()

# Metric calculations  
metrics = graph.calculate_metrics()

# Partitioning
partitions = graph.create_partitions(k=3)

# Serialization
graph.save("graph.json")
graph.load("graph.json")

# Subgraph extraction
subgraph = graph.extract_subgraph(nodes=["A", "B", "C"])
```

This modular organization keeps the codebase maintainable while providing a clean public interface through the Graph class.
