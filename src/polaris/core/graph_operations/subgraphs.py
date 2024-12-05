"""Graph subgraph extraction functionality."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, cast

from ..models import Edge


class SubgraphExtractor:
    """
    Extracts subgraphs from a larger graph.

    This class provides functionality to extract various types of subgraphs:
    - Node neighborhoods
    - Path contexts
    - Components
    - Custom subgraphs based on node/edge criteria
    """

    def __init__(self, edges: List[Edge]):
        """Initialize extractor with graph edges."""
        self.edges = edges
        self.out_adj: Dict[str, Set[str]] = defaultdict(set)
        self.in_adj: Dict[str, Set[str]] = defaultdict(set)
        self.nodes = set()
        self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> None:
        """Build adjacency lists for efficient subgraph operations."""
        self.out_adj = defaultdict(set)
        self.in_adj = defaultdict(set)
        self.nodes = set()

        for edge in self.edges:
            self.out_adj[edge.from_entity].add(edge.to_entity)
            self.in_adj[edge.to_entity].add(edge.from_entity)
            self.nodes.add(edge.from_entity)
            self.nodes.add(edge.to_entity)

    def extract_neighborhood(
        self, center_node: str, radius: int = 1, include_reverse: bool = True
    ) -> List[Edge]:
        """Extract neighborhood subgraph around a center node."""
        if center_node not in self.nodes:
            return []

        # Find nodes within radius using BFS
        neighborhood = {center_node}
        distances = {center_node: 0}
        queue = [(center_node, 0)]  # (node, distance)

        while queue:
            node, dist = queue.pop(0)
            if dist >= radius:
                continue

            # Forward edges
            for neighbor in self.out_adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    neighborhood.add(neighbor)
                    queue.append((neighbor, dist + 1))

            # Reverse edges if requested
            if include_reverse:
                for neighbor in self.in_adj[node]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        neighborhood.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Extract edges between neighborhood nodes
        return [
            edge
            for edge in self.edges
            if (
                edge.from_entity == center_node
                or edge.to_entity == center_node  # Direct connections
                or (
                    edge.from_entity in neighborhood
                    and edge.to_entity in neighborhood  # Within radius
                    and distances[edge.from_entity] < radius
                )
            )
        ]

    def extract_between(self, source_nodes: Set[str], target_nodes: Set[str]) -> List[Edge]:
        """Extract subgraph of paths between source and target node sets."""
        if not (source_nodes & self.nodes) or not (target_nodes & self.nodes):
            return []

        # Find all nodes on paths between sources and targets
        path_nodes = set()
        visited = set()
        queue = [(node, True) for node in source_nodes]  # (node, is_forward)

        while queue:
            node, is_forward = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            path_nodes.add(node)

            if is_forward:
                # Forward search from sources
                for neighbor in self.out_adj[node]:
                    if neighbor in target_nodes:
                        path_nodes.add(neighbor)
                    elif neighbor not in visited:
                        queue.append((neighbor, True))
            else:
                # Backward search from targets
                for neighbor in self.in_adj[node]:
                    if neighbor in source_nodes:
                        path_nodes.add(neighbor)
                    elif neighbor not in visited:
                        queue.append((neighbor, False))

        # Extract edges between path nodes
        return [
            edge
            for edge in self.edges
            if edge.from_entity in path_nodes and edge.to_entity in path_nodes
        ]

    def extract_path_context(
        self, path: Union[List[Edge], List[str]], context_size: int = 1
    ) -> List[Edge]:
        """Extract subgraph around a path with given context radius.

        Args:
            path: Either a list of Edge objects or a list of node IDs representing the path
            context_size: Number of hops to include around the path (default: 1)
        """
        if not path:
            return []

        # Get nodes in path
        path_nodes: Set[str] = set()
        if path and isinstance(path[0], Edge):
            edges = cast(List[Edge], path)
            path_nodes = {edge.from_entity for edge in edges} | {edge.to_entity for edge in edges}
        else:
            path_nodes = set(cast(List[str], path))

        # Find nodes within radius of path using BFS
        context_nodes = set(path_nodes)
        distances = {node: 0 for node in path_nodes}
        queue = [(node, 0) for node in path_nodes]  # (node, distance)

        while queue:
            node, dist = queue.pop(0)
            if dist >= context_size:
                continue

            # Forward edges
            for neighbor in self.out_adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    context_nodes.add(neighbor)
                    queue.append((neighbor, dist + 1))

            # Backward edges
            for neighbor in self.in_adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    context_nodes.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Extract edges between context nodes
        return [
            edge
            for edge in self.edges
            if edge.from_entity in context_nodes
            and edge.to_entity in context_nodes
            and (
                edge.from_entity in path_nodes
                or edge.to_entity in path_nodes
                or distances[edge.from_entity] < context_size
            )
        ]

    def extract_subgraph(self, nodes: Set[str]) -> List[Edge]:
        """Extract subgraph induced by a set of nodes."""
        return [
            edge for edge in self.edges if edge.from_entity in nodes and edge.to_entity in nodes
        ]
