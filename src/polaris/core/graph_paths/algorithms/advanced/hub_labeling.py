"""
Hub Labeling implementation for extremely fast path queries.

This module provides an implementation of Hub Labeling, a preprocessing-based
technique that enables constant-time shortest path distance queries. Key features:
- Efficient preprocessing with pruned label computation
- Extremely fast distance queries
- Memory-efficient label storage
- Path reconstruction support
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult
from polaris.core.graph_paths.utils import (
    MemoryManager,
    WeightFunc,
    create_path_result,
    get_edge_weight,
    validate_path,
)
from polaris.core.models import Edge

if TYPE_CHECKING:
    from polaris.core.graph import Graph


@dataclass(frozen=True)
class HubLabel:
    """
    Hub label containing distance and path information.

    Attributes:
        hub: Hub node ID
        distance: Distance to/from hub
        first_hop: First edge in path to hub (for path reconstruction)
    """

    hub: str
    distance: float
    first_hop: Optional[Edge]


class HubLabels(PathFinder[PathResult]):
    """
    Hub Labeling implementation for fast path queries.

    Features:
    - Constant-time distance queries
    - Space-efficient label storage
    - Path reconstruction support
    - Memory usage monitoring
    """

    def __init__(self, graph: "Graph", max_memory_mb: Optional[float] = None):
        """Initialize with graph and optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.forward_labels: Dict[str, List[HubLabel]] = {}
        self.backward_labels: Dict[str, List[HubLabel]] = {}
        self._preprocessed = False

    def preprocess(self) -> None:
        """
        Preprocess graph to compute hub labels.

        This computes forward and backward labels for each node using
        a pruned labeling approach.
        """
        # Compute node ordering for better label quality
        node_order = self._compute_node_order()
        total_nodes = len(node_order)

        # Process nodes in order
        for i, node in enumerate(node_order):
            self.memory_manager.check_memory()

            # Show progress
            if (i + 1) % 100 == 0 or i + 1 == total_nodes:
                progress = ((i + 1) / total_nodes) * 100
                print(f"Preprocessing: {progress:.1f}% complete")

            # Compute forward labels
            self._compute_forward_labels(node)
            # Compute backward labels
            self._compute_backward_labels(node)

        self._preprocessed = True
        print(
            f"Preprocessing complete, average labels per node: "
            f"{self._get_average_labels_per_node():.1f}"
        )

    def _compute_node_order(self) -> List[str]:
        """
        Compute node ordering for label computation.

        Returns nodes ordered by importance (degree + betweenness estimate).
        """
        # Calculate node importance based on degree
        importance = {}
        for node in self.graph.get_nodes():
            out_degree = len(list(self.graph.get_neighbors(node)))
            in_degree = len(list(self.graph.get_neighbors(node, reverse=True)))
            importance[node] = out_degree + in_degree

        # Sort nodes by importance
        return sorted(self.graph.get_nodes(), key=lambda n: importance[n], reverse=True)

    def _compute_forward_labels(self, node: str) -> None:
        """
        Compute forward labels for a node using pruned Dijkstra.

        Args:
            node: Node to compute labels for
        """
        distances = {node: 0.0}
        first_hops: Dict[str, Edge] = {}
        pq = [(0.0, node)]

        while pq:
            dist, current = heappop(pq)

            # Pruning condition
            if self._should_prune_forward(node, current, dist):
                continue

            # Add label
            if node not in self.forward_labels:
                self.forward_labels[node] = []
            self.forward_labels[node].append(
                HubLabel(hub=current, distance=dist, first_hop=first_hops.get(current))
            )

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current):
                edge = self.graph.get_edge(current, neighbor)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    if current == node:
                        first_hops[neighbor] = edge
                    elif current in first_hops:
                        first_hops[neighbor] = first_hops[current]
                    heappush(pq, (new_dist, neighbor))

    def _compute_backward_labels(self, node: str) -> None:
        """
        Compute backward labels for a node using pruned Dijkstra.

        Args:
            node: Node to compute labels for
        """
        distances = {node: 0.0}
        first_hops: Dict[str, Edge] = {}
        pq = [(0.0, node)]

        while pq:
            dist, current = heappop(pq)

            # Pruning condition
            if self._should_prune_backward(node, current, dist):
                continue

            # Add label
            if node not in self.backward_labels:
                self.backward_labels[node] = []
            self.backward_labels[node].append(
                HubLabel(hub=current, distance=dist, first_hop=first_hops.get(current))
            )

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current, reverse=True):
                edge = self.graph.get_edge(neighbor, current)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    if current == node:
                        first_hops[neighbor] = edge
                    elif current in first_hops:
                        first_hops[neighbor] = first_hops[current]
                    heappush(pq, (new_dist, neighbor))

    def _should_prune_forward(self, source: str, current: str, distance: float) -> bool:
        """
        Check if forward search can be pruned.

        Returns True if distance can be computed using existing labels.
        """
        if current not in self.forward_labels:
            return False

        # Check if distance can be computed using existing labels
        min_dist = float("inf")
        for hub_label in self.forward_labels[source]:
            if hub_label.hub in self.backward_labels:
                for target_label in self.backward_labels[hub_label.hub]:
                    if target_label.hub == current:
                        min_dist = min(min_dist, hub_label.distance + target_label.distance)

        return distance >= min_dist

    def _should_prune_backward(self, target: str, current: str, distance: float) -> bool:
        """
        Check if backward search can be pruned.

        Returns True if distance can be computed using existing labels.
        """
        if current not in self.backward_labels:
            return False

        # Check if distance can be computed using existing labels
        min_dist = float("inf")
        for hub_label in self.backward_labels[target]:
            if hub_label.hub in self.forward_labels:
                for source_label in self.forward_labels[hub_label.hub]:
                    if source_label.hub == current:
                        min_dist = min(min_dist, hub_label.distance + source_label.distance)

        return distance >= min_dist

    def _get_average_labels_per_node(self) -> float:
        """Calculate average number of labels per node."""
        total_labels = sum(len(labels) for labels in self.forward_labels.values()) + sum(
            len(labels) for labels in self.backward_labels.values()
        )
        return total_labels / (2 * len(self.graph.get_nodes()))

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """
        Find shortest path using hub labels.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            max_paths: Not used (always returns single path)
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists or graph not preprocessed
        """
        if not self._preprocessed:
            raise GraphOperationError("Graph must be preprocessed before finding paths")

        validate = kwargs.get("validate", True)

        # Find meeting hub that gives shortest distance
        best_dist = float("inf")
        best_hub = None

        if start_node not in self.forward_labels:
            raise GraphOperationError(f"No labels for start node {start_node}")
        if end_node not in self.backward_labels:
            raise GraphOperationError(f"No labels for end node {end_node}")

        for forward_label in self.forward_labels[start_node]:
            hub = forward_label.hub
            if hub in self.backward_labels[end_node]:
                backward_label = next(
                    label for label in self.backward_labels[end_node] if label.hub == hub
                )
                dist = forward_label.distance + backward_label.distance
                if dist < best_dist:
                    best_dist = dist
                    best_hub = hub

        if best_hub is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path through best hub
        path = self._reconstruct_path(start_node, end_node, best_hub)

        # Apply filter if provided
        if filter_func and not filter_func(path):
            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # Create and validate result
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(path, self.graph, weight_func, max_length)

        return result

    def _reconstruct_path(self, start_node: str, end_node: str, hub: str) -> List[Edge]:
        """
        Reconstruct path through hub using stored first hop edges.

        Args:
            start_node: Starting node
            end_node: Target node
            hub: Meeting hub node

        Returns:
            List of edges forming the complete path
        """
        path = []

        # Forward path to hub
        current = start_node
        while current != hub:
            forward_label = next(
                label for label in self.forward_labels[current] if label.hub == hub
            )
            if not forward_label.first_hop:
                break
            path.append(forward_label.first_hop)
            current = forward_label.first_hop.to_entity

        # Backward path from hub
        current = end_node
        backward_path = []
        while current != hub:
            backward_label = next(
                label for label in self.backward_labels[current] if label.hub == hub
            )
            if not backward_label.first_hop:
                break
            backward_path.append(backward_label.first_hop)
            current = backward_label.first_hop.from_entity

        # Combine paths
        path.extend(reversed(backward_path))
        return path
