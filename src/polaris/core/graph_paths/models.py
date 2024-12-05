"""
Data models for graph path finding.

This module provides the core data structures used throughout the path finding package:
- PathResult: Container for path finding results with validation
- PerformanceMetrics: Container for algorithm performance metrics
- PathValidationError: Exception for path validation failures

These models ensure consistent data handling and validation across all path finding
algorithms while providing useful metrics for performance monitoring.

Example:
    >>> path_result = PathResult(path=edges, total_weight=10.5)
    >>> path_result.validate()  # Ensures path consistency
    >>> path_result.nodes  # Get sequence of node IDs
    ['A', 'B', 'C', 'D']
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from polaris.core.graph import Graph
from polaris.core.models import Edge


class PathValidationError(Exception):
    """
    Raised when a path fails validation checks.

    This exception indicates issues such as:
    - Discontinuities in the path (nodes not properly connected)
    - Length mismatches between path and metadata
    - Weight inconsistencies
    - Self-loops
    - Missing edges in graph
    - Path exceeds maximum length
    - Missing required edge attributes
    """

    pass


@dataclass
class PathResult:
    """
    Container for path finding results.

    This class encapsulates a path through the graph along with its properties,
    providing methods for validation and analysis.

    Attributes:
        path: Sequence of edges forming the path
        total_weight: Total weight of the path
        length: Number of edges in the path (computed dynamically)

    Example:
        >>> result = PathResult(path=edges, total_weight=5.0)
        >>> result.validate(graph)  # Verify path consistency
        >>> print(f"Path goes through nodes: {result.nodes}")
    """

    path: List[Edge]
    total_weight: float

    def __post_init__(self):
        """Validate initialization parameters."""
        if not isinstance(self.path, list):
            raise TypeError("path must be a list")

        if not all(isinstance(edge, Edge) for edge in self.path):
            raise TypeError("path must contain only Edge objects")

        if None in self.path:
            raise TypeError("path cannot contain None values")

        if not isinstance(self.total_weight, (int, float)):
            raise TypeError("total_weight must be a numeric value")

        # Validate required edge attributes
        required_attrs = {"from_entity", "to_entity", "metadata", "relation_type"}
        for i, edge in enumerate(self.path):
            missing_attrs = required_attrs - set(vars(edge).keys())
            if missing_attrs:
                raise PathValidationError(
                    f"Edge at index {i} missing required attributes: {missing_attrs}"
                )

    def __len__(self) -> int:
        """Return the number of edges in the path."""
        return len(self.path)

    def __getitem__(self, index: int) -> Edge:
        """Get an edge from the path by index."""
        return self.path[index]

    def __iter__(self):
        """Return an iterator over the path edges."""
        return iter(self.path)

    @property
    def nodes(self) -> List[str]:
        """
        Get the sequence of node IDs in the path.

        Returns:
            List of node IDs in order of traversal
        """
        if not self.path:
            return []
        result = [self.path[0].from_entity]
        result.extend(edge.to_entity for edge in self.path)
        return result

    def validate(
        self,
        graph: Graph,
        weight_func: Optional[Any] = None,
        max_length: Optional[int] = None,
        weight_epsilon: float = 1e-9,
    ) -> None:
        """
        Validate the path's consistency.

        Performs comprehensive validation checks:
        - Path continuity (each edge connects to the next)
        - Edge existence (all edges exist in the graph)
        - Weight consistency (total weight matches sum of edge weights)
        - No self-loops
        - Maximum length constraints
        - Required edge attributes

        Args:
            graph: The graph instance to validate against
            weight_func: Optional function for custom weight calculation
            max_length: Optional maximum allowed path length
            weight_epsilon: Precision for weight comparisons (default: 1e-9)

        Raises:
            PathValidationError: If any validation check fails
            TypeError: If parameters have invalid types

        Example:
            >>> result.validate(
            ...     graph,
            ...     weight_func=lambda e: e.metadata.weight,
            ...     max_length=10,
            ...     weight_epsilon=1e-6
            ... )
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph instance")

        if max_length is not None:
            if not isinstance(max_length, int):
                raise TypeError("max_length must be an integer")
            if max_length < 0:
                raise ValueError("max_length must be non-negative")
            if len(self.path) > max_length:
                raise PathValidationError(
                    f"Path length {len(self.path)} exceeds maximum allowed length {max_length}"
                )

        if not isinstance(weight_epsilon, (int, float)):
            raise TypeError("weight_epsilon must be a numeric value")
        if weight_epsilon <= 0:
            raise ValueError("weight_epsilon must be positive")

        if not self.path:
            return

        # Check node connectivity
        for i in range(len(self.path) - 1):
            if self.path[i].to_entity != self.path[i + 1].from_entity:
                raise PathValidationError(
                    f"Path discontinuity between edges {i} and {i+1}: "
                    f"{self.path[i].to_entity} != {self.path[i + 1].from_entity}"
                )

        # Check for self-loops
        for edge in self.path:
            if edge.from_entity == edge.to_entity:
                raise PathValidationError(f"Self-loop detected at node {edge.from_entity}")

        # Verify edges exist in graph
        for edge in self.path:
            if not graph.has_edge(edge.from_entity, edge.to_entity):
                raise PathValidationError(
                    f"Edge from {edge.from_entity} to {edge.to_entity} not found in graph"
                )

        # Verify weight consistency if weight_func provided
        if weight_func is not None:
            try:
                calculated_weight = sum(weight_func(edge) for edge in self.path)
            except Exception as e:
                raise PathValidationError(f"Error calculating path weight: {str(e)}")

            if abs(calculated_weight - self.total_weight) > weight_epsilon:
                raise PathValidationError(
                    f"Weight mismatch: calculated {calculated_weight} != stored {self.total_weight}"
                )


@dataclass
class PerformanceMetrics:
    """
    Container for path finding performance metrics.

    This class tracks various performance metrics for path finding operations,
    providing insights into algorithm efficiency and cache effectiveness.

    Attributes:
        operation: Name of the path finding operation
        start_time: Operation start timestamp
        end_time: Operation end timestamp (0.0 if not completed)
        path_length: Length of found path (if applicable)
        cache_hit: Whether result was from cache
        nodes_explored: Number of nodes explored during search
        max_memory_used: Peak memory usage during operation (bytes)

    Example:
        >>> metrics = PerformanceMetrics(operation="shortest_path", start_time=time())
        >>> # ... perform operation ...
        >>> metrics.end_time = time()
        >>> print(f"Operation took {metrics.duration:.2f}ms")
    """

    operation: str
    start_time: float
    end_time: float = 0.0
    path_length: Optional[int] = None
    cache_hit: bool = False
    nodes_explored: Optional[int] = None
    max_memory_used: Optional[int] = None

    def __post_init__(self):
        """Validate metrics after initialization."""
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string")

        if not isinstance(self.start_time, (int, float)):
            raise TypeError("start_time must be a numeric value")

        if not isinstance(self.end_time, (int, float)):
            raise TypeError("end_time must be a numeric value")

        if self.end_time < 0:
            raise ValueError("end_time cannot be negative")

        if self.end_time and self.end_time < self.start_time:
            raise ValueError("end_time cannot be before start_time")

        if self.path_length is not None:
            if not isinstance(self.path_length, int):
                raise TypeError("path_length must be an integer")
            if self.path_length < 0:
                raise ValueError("path_length cannot be negative")

        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a boolean")

        if self.nodes_explored is not None:
            if not isinstance(self.nodes_explored, int):
                raise TypeError("nodes_explored must be an integer")
            if self.nodes_explored < 0:
                raise ValueError("nodes_explored cannot be negative")

        if self.max_memory_used is not None:
            if not isinstance(self.max_memory_used, int):
                raise TypeError("max_memory_used must be an integer")
            if self.max_memory_used < 0:
                raise ValueError("max_memory_used cannot be negative")

    @property
    def duration(self) -> float:
        """
        Calculate operation duration in milliseconds.

        Returns:
            Duration of the operation in milliseconds
        """
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, None]]:
        """
        Convert metrics to dictionary format.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "operation": self.operation,
            "duration_ms": self.duration,
            "path_length": self.path_length,
            "cache_hit": self.cache_hit,
            "nodes_explored": self.nodes_explored,
            "max_memory_used": self.max_memory_used,
        }
