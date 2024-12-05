"""Graph partitioning operations.

This module provides functionality for partitioning graphs:
- Balanced k-way partitioning
- Spectral partitioning
- Partition quality metrics
- Cross-partition edge analysis
"""

from typing import Dict, List, NamedTuple, Set, Tuple, Optional
from collections import defaultdict
import random
from ..models import Edge
from ..exceptions import GraphOperationError


class PartitionMetrics(NamedTuple):
    """Container for partition quality metrics."""

    partition_size: int  # Number of nodes in partition
    internal_edges: int  # Edges within partition
    external_edges: int  # Edges crossing partition boundary
    density: float  # Internal edge density


class Partition:
    """Represents a graph partition with quality metrics."""

    def __init__(self, nodes: Set[str], partition_id: int):
        """Initialize partition.

        Args:
            nodes: Set of node IDs in this partition
            partition_id: Unique identifier for this partition
        """
        self.nodes = nodes
        self.partition_id = partition_id
        self.internal_edges: Set[Tuple[str, str]] = set()
        self.external_edges: Set[Tuple[str, str]] = set()

    def add_edge(self, from_node: str, to_node: str, is_internal: bool) -> None:
        """Add an edge to the partition.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            is_internal: True if edge is internal to partition
        """
        edge = (from_node, to_node)
        if is_internal:
            self.internal_edges.add(edge)
        else:
            self.external_edges.add(edge)

    def get_metrics(self) -> PartitionMetrics:
        """Calculate partition metrics.

        Returns:
            PartitionMetrics containing quality metrics
        """
        n = len(self.nodes)
        possible_edges = n * (n - 1) if n > 1 else 0
        density = len(self.internal_edges) / possible_edges if possible_edges > 0 else 0.0

        return PartitionMetrics(
            partition_size=n,
            internal_edges=len(self.internal_edges),
            external_edges=len(self.external_edges),
            density=density,
        )


class GraphPartitioner:
    """Handles graph partitioning operations."""

    def __init__(self, edges: List[Edge]):
        """Initialize partitioner.

        Args:
            edges: List of edges in the graph
        """
        self.edges = edges
        self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> None:
        """Build adjacency lists for efficient partitioning."""
        self.adj_list: Dict[str, Set[str]] = defaultdict(set)
        self.nodes = set()

        for edge in self.edges:
            self.adj_list[edge.from_node].add(edge.to_node)
            self.adj_list[edge.to_node].add(edge.from_node)  # Treat as undirected for partitioning
            self.nodes.add(edge.from_node)
            self.nodes.add(edge.to_node)

    def create_balanced_partitions(self, k: int) -> Dict[int, Partition]:
        """Create k roughly equal-sized partitions using a greedy algorithm.

        Args:
            k: Number of partitions to create

        Returns:
            Dictionary mapping partition IDs to Partition objects

        Raises:
            GraphOperationError: If k <= 0 or k > number of nodes
        """
        if k <= 0:
            raise GraphOperationError("Number of partitions must be positive")
        if k > len(self.nodes):
            raise GraphOperationError("Cannot create more partitions than nodes")

        # Initialize partitions
        partitions: Dict[int, Partition] = {}
        unassigned = list(self.nodes)
        random.shuffle(unassigned)  # Randomize initial assignment

        # Create empty partitions
        target_size = len(self.nodes) // k
        remainder = len(self.nodes) % k

        for i in range(k):
            size = target_size + (1 if i < remainder else 0)
            partition_nodes = set(unassigned[:size])
            unassigned = unassigned[size:]
            partitions[i] = Partition(partition_nodes, i)

        # Assign edges to partitions
        for edge in self.edges:
            from_partition: Optional[Partition] = None
            to_partition: Optional[Partition] = None

            # Find partitions containing edge endpoints
            for p in partitions.values():
                if edge.from_node in p.nodes:
                    from_partition = p
                if edge.to_node in p.nodes:
                    to_partition = p
                if from_partition and to_partition:
                    break

            if from_partition is not None and to_partition is not None:
                if from_partition == to_partition:
                    # Internal edge
                    from_partition.add_edge(edge.from_node, edge.to_node, True)
                else:
                    # External edge
                    from_partition.add_edge(edge.from_node, edge.to_node, False)
                    to_partition.add_edge(edge.from_node, edge.to_node, False)

        return partitions

    def optimize_partitions(
        self, partitions: Dict[int, Partition], iterations: int = 10
    ) -> Dict[int, Partition]:
        """Optimize partition quality using node swapping.

        Args:
            partitions: Current partitions to optimize
            iterations: Number of optimization iterations

        Returns:
            Optimized partitions
        """
        for _ in range(iterations):
            improved = False

            # Try swapping nodes between partitions
            for p1_id, p1 in partitions.items():
                for p2_id, p2 in partitions.items():
                    if p1_id >= p2_id:
                        continue

                    # Find nodes that might benefit from swapping
                    for n1 in p1.nodes:
                        external_edges_n1 = sum(1 for n in self.adj_list[n1] if n in p2.nodes)
                        internal_edges_n1 = sum(1 for n in self.adj_list[n1] if n in p1.nodes)

                        if external_edges_n1 > internal_edges_n1:
                            # Node might benefit from moving to p2
                            for n2 in p2.nodes:
                                external_edges_n2 = sum(
                                    1 for n in self.adj_list[n2] if n in p1.nodes
                                )
                                internal_edges_n2 = sum(
                                    1 for n in self.adj_list[n2] if n in p2.nodes
                                )

                                if external_edges_n2 > internal_edges_n2:
                                    # Swap nodes
                                    p1.nodes.remove(n1)
                                    p2.nodes.remove(n2)
                                    p1.nodes.add(n2)
                                    p2.nodes.add(n1)
                                    improved = True
                                    break

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        # Recalculate edge assignments
        for p in partitions.values():
            p.internal_edges.clear()
            p.external_edges.clear()

        for edge in self.edges:
            from_partition: Optional[Partition] = None
            to_partition: Optional[Partition] = None

            for p in partitions.values():
                if edge.from_node in p.nodes:
                    from_partition = p
                if edge.to_node in p.nodes:
                    to_partition = p
                if from_partition and to_partition:
                    break

            if from_partition is not None and to_partition is not None:
                if from_partition == to_partition:
                    from_partition.add_edge(edge.from_node, edge.to_node, True)
                else:
                    from_partition.add_edge(edge.from_node, edge.to_node, False)
                    to_partition.add_edge(edge.from_node, edge.to_node, False)

        return partitions

    def get_cross_partition_edges(self, partitions: Dict[int, Partition]) -> List[Edge]:
        """Get edges that cross partition boundaries.

        Args:
            partitions: Current graph partitions

        Returns:
            List of edges that connect different partitions
        """
        cross_edges = []

        for edge in self.edges:
            from_partition = None
            to_partition = None

            for p in partitions.values():
                if edge.from_node in p.nodes:
                    from_partition = p
                if edge.to_node in p.nodes:
                    to_partition = p
                if from_partition and to_partition:
                    break

            if from_partition != to_partition:
                cross_edges.append(edge)

        return cross_edges

    def get_partition_metrics(
        self, partitions: Dict[int, Partition]
    ) -> Dict[int, PartitionMetrics]:
        """Calculate metrics for each partition.

        Args:
            partitions: Current graph partitions

        Returns:
            Dictionary mapping partition IDs to their metrics
        """
        return {pid: p.get_metrics() for pid, p in partitions.items()}
