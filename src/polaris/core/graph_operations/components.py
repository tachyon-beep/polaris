"""Graph component analysis functionality."""

from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..models import Edge


class ComponentAnalysis:
    """
    Analyzes connected components in a graph.

    This class provides functionality to identify and analyze connected
    components within a graph, including finding isolated nodes and
    determining component sizes.
    """

    def __init__(self, edges: List[Edge]):
        """Initialize component analyzer with graph edges."""
        self.edges = edges
        self.components: Dict[str, Set[str]] = {}  # Map nodes to their component
        self.component_sizes: Dict[int, int] = {}  # Map component IDs to sizes
        self.isolated_nodes: Set[str] = set()
        self._analyze_components()

    def _analyze_components(self) -> None:
        """Identify connected components using depth-first search."""
        visited = set()
        component_id = 0

        # Build adjacency list for efficient traversal
        adj_list: Dict[str, Set[str]] = {}
        for edge in self.edges:
            if edge.from_entity not in adj_list:
                adj_list[edge.from_entity] = set()
            if edge.to_entity not in adj_list:
                adj_list[edge.to_entity] = set()
            adj_list[edge.from_entity].add(edge.to_entity)
            adj_list[edge.to_entity].add(edge.from_entity)  # Treat as undirected for components

        # Find components using DFS
        def dfs(node: str, component: Set[str]) -> None:
            visited.add(node)
            component.add(node)
            self.components[node] = component
            for neighbor in adj_list.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        # Process each node
        for edge in self.edges:
            for node in [edge.from_entity, edge.to_entity]:
                if node not in visited:
                    current_component = set()
                    dfs(node, current_component)
                    self.component_sizes[component_id] = len(current_component)
                    component_id += 1

        # Find isolated nodes (degree 0 or only self-loops)
        all_nodes = set()
        for edge in self.edges:
            all_nodes.add(edge.from_entity)
            all_nodes.add(edge.to_entity)

        for node in all_nodes:
            neighbors = adj_list.get(node, set())
            if not neighbors or (len(neighbors) == 1 and node in neighbors):
                self.isolated_nodes.add(node)

    def get_components(self) -> List[Set[str]]:
        """Get list of all components (sets of node IDs)."""
        unique_components = set(frozenset(comp) for comp in self.components.values())
        return [set(comp) for comp in unique_components]

    def get_component_count(self) -> int:
        """Get number of connected components."""
        return len(self.component_sizes)

    def get_isolated_nodes(self) -> Set[str]:
        """Get set of isolated nodes (degree 0 or only self-loops)."""
        return self.isolated_nodes

    def are_connected(self, node1: str, node2: str) -> bool:
        """Check if two nodes are in the same component."""
        if node1 not in self.components or node2 not in self.components:
            return False
        return self.components[node1] is self.components[node2]

    def get_largest_component(self) -> Set[str]:
        """Get the largest connected component."""
        if not self.component_sizes:
            return set()
        largest_id = max(self.component_sizes.items(), key=lambda x: x[1])[0]
        return next(
            comp
            for comp in self.components.values()
            if len(comp) == self.component_sizes[largest_id]
        )
