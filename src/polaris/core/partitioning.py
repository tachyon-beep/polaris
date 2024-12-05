from typing import Dict, Set, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import random
import math
from collections import defaultdict

from .graph import Graph
from .models import Edge


@dataclass
class PartitionMetrics:
    """Metrics about a graph partition."""

    partition_size: int
    internal_edges: int
    external_edges: int
    density: float
    modularity: float


class GraphPartitioner:
    """Handles partitioning of large graphs into smaller subgraphs."""

    def __init__(self, graph: Graph, num_partitions: int):
        self.graph = graph
        self.num_partitions = num_partitions

    def partition_by_modularity(self) -> Dict[int, Set[str]]:
        """Partition graph using the Louvain method for community detection."""
        nodes = list(self.graph.get_nodes())
        if not nodes:
            return {}

        # Initialize each node in its own community
        communities: Dict[str, int] = {node: i for i, node in enumerate(nodes)}
        node_degrees: Dict[str, int] = {
            node: len(list(self.graph.get_neighbors(node))) for node in nodes
        }
        total_edges = sum(node_degrees.values()) // 2

        while True:
            improvement = False
            # Phase 1: Modularity optimization
            for node in nodes:
                current_community = communities[node]
                best_gain = 0.0
                best_community = current_community

                # Calculate gains for moving to each neighbor's community
                neighbor_communities = set(
                    communities[neighbor] for neighbor in self.graph.get_neighbors(node)
                )

                for target_community in neighbor_communities:
                    if target_community != current_community:
                        gain = self._calculate_modularity_gain(
                            node,
                            current_community,
                            target_community,
                            communities,
                            node_degrees,
                            total_edges,
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_community = target_community

                if best_community != current_community:
                    communities[node] = best_community
                    improvement = True

            if not improvement:
                break

        # Convert to partition format
        partitions: Dict[int, Set[str]] = defaultdict(set)
        for node, community in communities.items():
            partitions[community].add(node)

        # Merge small partitions if needed
        return self._merge_small_partitions(dict(partitions))

    def partition_by_degree(self) -> Dict[int, Set[str]]:
        """Partition graph by node degrees."""
        nodes = sorted(
            self.graph.get_nodes(),
            key=lambda n: len(list(self.graph.get_neighbors(n))),
            reverse=True,
        )

        partitions: Dict[int, Set[str]] = {i: set() for i in range(self.num_partitions)}

        # Distribute nodes across partitions
        for i, node in enumerate(nodes):
            partition_id = i % self.num_partitions
            partitions[partition_id].add(node)

        return partitions

    def partition_by_bfs(self) -> Dict[int, Set[str]]:
        """Partition graph using breadth-first search."""
        nodes = set(self.graph.get_nodes())
        if not nodes:
            return {}

        partitions: Dict[int, Set[str]] = {}
        partition_id = 0

        while nodes and partition_id < self.num_partitions:
            # Start new partition from highest degree unassigned node
            start_node = max(nodes, key=lambda n: len(list(self.graph.get_neighbors(n))))
            partition = self._grow_partition(
                start_node, nodes, len(nodes) // (self.num_partitions - partition_id)
            )

            partitions[partition_id] = partition
            nodes -= partition
            partition_id += 1

        # Assign any remaining nodes to the last partition
        if nodes and partitions:
            partitions[partition_id - 1].update(nodes)

        return partitions

    def _grow_partition(
        self, start_node: str, available_nodes: Set[str], target_size: int
    ) -> Set[str]:
        """Grow a partition from a start node using BFS."""
        partition = {start_node}
        queue = [start_node]

        while queue and len(partition) < target_size:
            node = queue.pop(0)
            for neighbor in self.graph.get_neighbors(node):
                if (
                    neighbor in available_nodes
                    and neighbor not in partition
                    and len(partition) < target_size
                ):
                    partition.add(neighbor)
                    queue.append(neighbor)

        return partition

    def _calculate_modularity_gain(
        self,
        node: str,
        current_community: int,
        target_community: int,
        communities: Dict[str, int],
        node_degrees: Dict[str, int],
        total_edges: int,
    ) -> float:
        """Calculate modularity gain for moving a node to a different community."""
        if total_edges == 0:
            return 0.0

        node_degree = node_degrees[node]

        # Calculate connections to current and target communities
        current_connections = sum(
            1
            for neighbor in self.graph.get_neighbors(node)
            if communities[neighbor] == current_community
        )
        target_connections = sum(
            1
            for neighbor in self.graph.get_neighbors(node)
            if communities[neighbor] == target_community
        )

        # Calculate community degrees
        current_degree = sum(
            node_degrees[n] for n, c in communities.items() if c == current_community
        )
        target_degree = sum(
            node_degrees[n] for n, c in communities.items() if c == target_community
        )

        gain = (
            target_connections
            - current_connections
            + node_degree * (current_degree - target_degree) / (2 * total_edges)
        )

        return gain / total_edges

    def _merge_small_partitions(self, partitions: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
        """Merge partitions that are too small."""
        min_size = len(list(self.graph.get_nodes())) // (self.num_partitions * 2)

        # Sort partitions by size
        sorted_partitions = sorted(partitions.items(), key=lambda x: len(x[1]))

        # Merge small partitions into larger ones
        result: Dict[int, Set[str]] = {}
        current_partition: Set[str] = set()
        current_id = 0

        for pid, nodes in sorted_partitions:
            if len(current_partition) + len(nodes) <= min_size:
                current_partition.update(nodes)
            else:
                if current_partition:
                    result[current_id] = current_partition
                    current_id += 1
                current_partition = nodes

        if current_partition:
            result[current_id] = current_partition

        return result


class PartitionedGraph:
    """Represents a graph that has been partitioned into subgraphs."""

    def __init__(self, base_graph: Graph, partitions: Dict[int, Set[str]]):
        self.base_graph = base_graph
        self.partitions = partitions
        self.partition_graphs: Dict[int, Graph] = {}
        self._build_partition_graphs()

    def _build_partition_graphs(self) -> None:
        """Build subgraphs for each partition."""
        for partition_id, nodes in self.partitions.items():
            # Create subgraph for partition
            partition_edges = [
                edge
                for edge in self.base_graph.get_edges()
                if edge.from_entity in nodes and edge.to_entity in nodes
            ]
            self.partition_graphs[partition_id] = Graph(partition_edges)

    def get_partition_graph(self, partition_id: int) -> Optional[Graph]:
        """Get the subgraph for a specific partition."""
        return self.partition_graphs.get(partition_id)

    def get_cross_partition_edges(self) -> List[Tuple[int, int, Edge]]:
        """Get edges that cross partition boundaries."""
        cross_edges = []
        for edge in self.base_graph.get_edges():
            from_partition = self._get_partition_id(edge.from_entity)
            to_partition = self._get_partition_id(edge.to_entity)
            if from_partition != to_partition:
                cross_edges.append((from_partition, to_partition, edge))
        return cross_edges

    def _get_partition_id(self, node: str) -> int:
        """Get the partition ID for a node."""
        for partition_id, nodes in self.partitions.items():
            if node in nodes:
                return partition_id
        raise ValueError(f"Node {node} not found in any partition")

    def get_partition_metrics(self) -> Dict[int, PartitionMetrics]:
        """Calculate metrics for each partition."""
        metrics = {}
        cross_edges = self.get_cross_partition_edges()

        for partition_id, nodes in self.partitions.items():
            subgraph = self.partition_graphs[partition_id]
            internal_edges = len(list(subgraph.get_edges()))
            external_edges = sum(
                1
                for from_p, to_p, _ in cross_edges
                if from_p == partition_id or to_p == partition_id
            )

            # Calculate density and modularity
            possible_edges = len(nodes) * (len(nodes) - 1) / 2
            density = internal_edges / possible_edges if possible_edges > 0 else 0

            modularity = (
                internal_edges / (internal_edges + external_edges)
                if internal_edges + external_edges > 0
                else 0
            )

            metrics[partition_id] = PartitionMetrics(
                partition_size=len(nodes),
                internal_edges=internal_edges,
                external_edges=external_edges,
                density=density,
                modularity=modularity,
            )

        return metrics
