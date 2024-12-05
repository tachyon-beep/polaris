# Advanced Graph Algorithms

This package provides sophisticated path finding algorithms that offer significant performance improvements through preprocessing and other advanced techniques.

## Algorithm Details

### Contraction Hierarchies (CH)

Technical details:
- Node ordering using edge difference and contracted neighbors
- Witness path search for shortcut necessity
- Memory-efficient shortcut storage
- Bidirectional path finding with shortcuts

Implementation features:
```python
class ContractionHierarchies:
    def _calculate_node_importance(self, node: str) -> float:
        """
        Importance = 5 * edge_difference + 
                    3 * contracted_neighbors +
                    2 * level
        """

    def _contract_node(self, node: str) -> List[Tuple[str, str]]:
        """
        1. Find necessary shortcuts
        2. Create shortcut edges
        3. Update contracted neighbors
        """

    def _is_shortcut_necessary(
        self,
        u: str,
        v: str,
        via: str,
        shortcut_weight: float
    ) -> bool:
        """
        Witness path search with early termination
        """
```

### Hub Labeling (HL)

Technical details:
- Pruned label computation
- Distance-based pruning criterion
- Space-efficient label storage
- Constant-time distance queries

Implementation features:
```python
class HubLabels:
    def _compute_forward_labels(self, node: str) -> None:
        """
        1. Run pruned Dijkstra
        2. Add distance labels
        3. Store first hop for path reconstruction
        """

    def _should_prune_forward(
        self,
        source: str,
        current: str,
        distance: float
    ) -> bool:
        """
        Prune if better path exists through existing labels
        """
```

### Transit Node Routing (TNR)

Technical details:
- Access node computation with locality filter
- Transit node selection using importance criteria
- Distance table computation
- Local/global path finding

Implementation features:
```python
class TransitNodeRouting:
    def _select_transit_nodes(self) -> Set[str]:
        """
        Select nodes based on:
        1. Degree centrality
        2. Betweenness estimate
        3. Geographic coverage
        """

    def _compute_access_nodes(self, node: str) -> None:
        """
        1. Find closest transit nodes
        2. Store access paths
        3. Apply locality radius
        """

    def _is_local_query(
        self,
        source: str,
        target: str
    ) -> bool:
        """
        Check if nodes share access nodes
        """
```

### A* with Landmarks (ALT)

Technical details:
- Maximal separation landmark selection
- Triangle inequality based lower bounds
- Bidirectional ALT implementation
- Dynamic landmark updates

Implementation features:
```python
class ALTPathFinder:
    def _select_landmarks(self) -> List[str]:
        """
        Select landmarks using:
        1. Maximal separation
        2. Coverage optimization
        3. Importance criteria
        """

    def _compute_heuristic(
        self,
        node: str,
        target: str,
        weight_func: Optional[WeightFunc]
    ) -> float:
        """
        Use triangle inequality with landmarks
        """
```

## Memory Management

All advanced algorithms use the MemoryManager for monitoring:

```python
class MemoryManager:
    def check_memory(self) -> None:
        """Monitor memory usage during preprocessing"""

    @contextmanager
    def monitor_allocation(self, label: str) -> Generator:
        """Track memory allocation for operations"""
```

## Progress Tracking

Preprocessing progress is tracked:

```python
def preprocess(self) -> None:
    """
    total = len(nodes)
    for i, node in enumerate(nodes):
        progress = ((i + 1) / total) * 100
        print(f"Progress: {progress:.1f}%")
    """
```

## Performance Comparison

Algorithm characteristics:

| Algorithm | Preprocessing | Query Time | Space | Dynamic |
|-----------|--------------|------------|-------|---------|
| CH        | O(n log n)   | O(log n)   | O(n)  | No      |
| HL        | O(n²)        | O(1)       | O(n√n)| No      |
| TNR       | O(n log n)   | O(1)       | O(n)  | Partial |
| ALT       | O(kn)        | O(n)       | O(kn) | Yes     |

Where:
- n = number of nodes
- k = number of landmarks

## Implementation Notes

### Memory Efficiency

1. Use of `__slots__` for classes:
```python
@dataclass(frozen=True, slots=True)
class PathState:
    """Memory-efficient path state"""
```

2. Lazy computation:
```python
def _compute_distance_table(self) -> None:
    """Compute distances only when needed"""
```

3. Efficient data structures:
```python
from heapq import heappush, heappop
pq = []  # Priority queue for Dijkstra
```

### Error Handling

Comprehensive error checking:
```python
if not self._preprocessed:
    raise GraphOperationError(
        "Preprocessing required"
    )

if node not in self.forward_labels:
    raise GraphOperationError(
        f"No labels for node {node}"
    )
```

### Path Validation

All algorithms validate paths:
```python
def find_path(self, start: str, end: str) -> PathResult:
    """
    path = self._compute_path()
    validate_path(path, self.graph)
    return create_path_result(path)
    """
```

## Usage Examples

### Contraction Hierarchies
```python
ch = ContractionHierarchies(graph, max_memory_mb=1000)
ch.preprocess()
path = ch.find_path(
    start_node="A",
    end_node="B",
    weight_func=lambda e: e.metadata.weight
)
```

### Hub Labeling
```python
hl = HubLabels(graph, max_memory_mb=1000)
hl.preprocess()
path = hl.find_path(
    start_node="A",
    end_node="B",
    validate=True
)
```

### Transit Node Routing
```python
tnr = TransitNodeRouting(
    graph,
    num_transit_nodes=1000,
    max_memory_mb=1000
)
tnr.preprocess()
path = tnr.find_path("A", "B")
```

### ALT
```python
alt = ALTPathFinder(
    graph,
    num_landmarks=16,
    max_memory_mb=1000
)
alt.preprocess()
path = alt.find_path(
    "A", "B",
    bidirectional=True
)
```

## Testing

Each algorithm includes comprehensive tests:

1. Correctness tests
2. Performance tests
3. Memory usage tests
4. Error handling tests
5. Edge case tests

Example test structure:
```python
def test_preprocessing():
    """Test preprocessing correctness"""

def test_path_finding():
    """Test path finding correctness"""

def test_memory_usage():
    """Test memory management"""

def test_error_handling():
    """Test error conditions"""
```

## Contributing

When adding new advanced algorithms:

1. Implement preprocessing if needed
2. Add memory management
3. Include progress tracking
4. Provide comprehensive tests
5. Update documentation
6. Follow existing code style
