"""Connected component analysis for the knowledge graph.

This module provides functionality for analyzing the structural components and
characteristics of the knowledge graph. It includes methods for:
- Finding weakly connected components (subgraphs where nodes are reachable ignoring edge direction)
- Finding strongly connected components (subgraphs where nodes are mutually reachable 
                                        following edge direction)
- Calculating clustering coefficients (measure of node clustering/transitivity)

The analysis helps understand the graph's topology and identify distinct clusters
of related nodes, which is valuable for:
- Understanding data organization and relationships
- Identifying isolated subgraphs
- Analyzing node clustering patterns
- Detecting community structures
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Optional

from .exceptions import GraphOperationError
from .graph import Graph


class ComponentAnalysis:
    """Connected component analysis for knowledge graphs.

    This class provides static methods for analyzing structural properties of the
    knowledge graph, particularly focusing on connected components and clustering
    characteristics. These analyses help understand the graph's topology and
    identify patterns in node relationships.

    The analysis methods are implemented as static methods to provide utility-style
    functionality that can be used with any Graph instance without maintaining
    state.
    """

    @staticmethod
    def _collect_all_nodes(graph: Graph) -> Set[str]:
        """Collect all nodes in the graph, including isolated ones.

        Args:
            graph (Graph): The graph instance to analyze.

        Returns:
            Set[str]: Set of all node IDs in the graph.
        """
        nodes = set(graph.adjacency.keys())
        for neighbors in graph.adjacency.values():
            nodes.update(neighbors)
        return nodes

    @staticmethod
    def _build_undirected_adjacency(graph: Graph) -> Dict[str, Set[str]]:
        """Build undirected adjacency map from directed graph.

        Creates a mapping where each edge is represented in both directions,
        allowing for component detection that considers reachability in either direction.

        Args:
            graph (Graph): The directed graph instance.

        Returns:
            Dict[str, Set[str]]: Undirected adjacency map.
        """
        undirected_adjacency = defaultdict(set)
        for node, neighbors in graph.adjacency.items():
            for neighbor in neighbors:
                undirected_adjacency[node].add(neighbor)
                undirected_adjacency[neighbor].add(node)
        return undirected_adjacency

    @staticmethod
    def _find_component_bfs(
        start: str, adjacency: Dict[str, Set[str]], visited: Set[str]
    ) -> Set[str]:
        """Find all nodes in a component using breadth-first search.

        Args:
            start (str): Starting node.
            adjacency (Dict[str, Set[str]]): Adjacency map.
            visited (Set[str]): Set of visited nodes.

        Returns:
            Set[str]: Set of nodes in the component.
        """
        component = {start}
        queue = deque([start])
        visited.add(start)

        while queue:
            current_node = queue.popleft()
            for neighbor in adjacency[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        return component

    @staticmethod
    def _find_components_impl(graph: Graph) -> List[Set[str]]:
        """Implementation of component finding using BFS.

        Args:
            graph (Graph): The graph instance to analyze.

        Returns:
            List[Set[str]]: List of components.
        """
        components = []
        visited = set()
        all_nodes = ComponentAnalysis._collect_all_nodes(graph)
        undirected_adjacency = ComponentAnalysis._build_undirected_adjacency(graph)

        # Find components using undirected adjacency
        for node in all_nodes:
            if node not in visited:
                component = ComponentAnalysis._find_component_bfs(
                    node, undirected_adjacency, visited
                )
                if component:
                    components.append(component)

        return components

    @staticmethod
    def _find_components_iterative(graph: Graph) -> List[Set[str]]:
        """Iterative implementation for finding components.

        This is a fallback implementation that uses an explicit stack instead of recursion,
        making it suitable for very large graphs where recursion might cause stack overflow.

        Args:
            graph (Graph): The graph instance to analyze.

        Returns:
            List[Set[str]]: List of components.
        """
        components = []
        visited = set()
        all_nodes = ComponentAnalysis._collect_all_nodes(graph)
        undirected_adjacency = ComponentAnalysis._build_undirected_adjacency(graph)

        for start_node in all_nodes:
            if start_node in visited:
                continue

            # Initialize component
            component = {start_node}
            stack = [start_node]
            visited.add(start_node)

            # Process stack iteratively
            while stack:
                current_node = stack.pop()
                for neighbor in undirected_adjacency[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)

            if component:
                components.append(component)

        return components

    @staticmethod
    def _tarjan_scc(
        graph: Graph,
        node: str,
        index: List[int],
        indices: Dict[str, int],
        lowlinks: Dict[str, int],
        stack: List[str],
        on_stack: Set[str],
        strongly_connected_components: List[Set[str]],
    ) -> None:
        """Helper method for Tarjan's strongly connected components algorithm.

        Args:
            graph (Graph): The graph instance.
            node (str): Current node being processed.
            index (List[int]): Current DFS index (in list for mutability).
            indices (Dict[str, int]): Discovery time indices for nodes.
            lowlinks (Dict[str, int]): Lowest reachable index for nodes.
            stack (List[str]): Stack of nodes being processed.
            on_stack (Set[str]): Set of nodes currently on stack.
            strongly_connected_components (List[Set[str]]): List to collect
                strongly connected components.
        """
        # Initialize discovery index and lowlink value for current node
        indices[node] = index[0]
        lowlinks[node] = index[0]
        index[0] += 1
        stack.append(node)
        on_stack.add(node)

        # Consider successors of node
        if node in graph.adjacency:
            for successor in graph.adjacency[node]:
                if successor not in indices:
                    # Successor has not yet been visited; recurse on it
                    ComponentAnalysis._tarjan_scc(
                        graph,
                        successor,
                        index,
                        indices,
                        lowlinks,
                        stack,
                        on_stack,
                        strongly_connected_components,
                    )
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    # Successor is in current strongly connected component
                    lowlinks[node] = min(lowlinks[node], indices[successor])

        # If node is root of strongly connected component, collect it
        if lowlinks[node] == indices[node]:
            component = set()
            while True:
                successor = stack.pop()
                on_stack.remove(successor)
                component.add(successor)
                if successor == node:
                    break
            strongly_connected_components.append(component)

    @staticmethod
    def find_strongly_connected_components(graph: Graph) -> List[Set[str]]:
        """Find all strongly connected components in the directed graph.

        A strongly connected component (SCC) is a subgraph where every node is
        reachable from every other node following the direction of edges. This
        method uses Tarjan's algorithm to find all SCCs in the graph.

        Args:
            graph (Graph): The graph instance to analyze.

        Returns:
            List[Set[str]]: A list of sets, where each set contains the node IDs
                           belonging to a distinct strongly connected component.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="A", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="C", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> sccs = ComponentAnalysis.find_strongly_connected_components(graph)
            >>> # Returns [{"A", "B"}, {"C"}] as A and B form an SCC, C is separate

        Note:
            - Components are returned in reverse topological order
            - Each node appears in exactly one component
            - Isolated nodes form their own single-node components
            - Unlike weakly connected components, SCCs respect edge directions
            - SCCs provide stronger guarantees about node reachability
        """
        index = [0]  # Wrapped in list for mutability
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        strongly_connected_components: List[Set[str]] = []

        # Process all nodes
        all_nodes = ComponentAnalysis._collect_all_nodes(graph)
        for node in all_nodes:
            if node not in indices:
                ComponentAnalysis._tarjan_scc(
                    graph,
                    node,
                    index,
                    indices,
                    lowlinks,
                    stack,
                    on_stack,
                    strongly_connected_components,
                )

        return strongly_connected_components

    @staticmethod
    def find_components(graph: Graph) -> List[Set[str]]:
        """Find all weakly connected components in the graph.

        A weakly connected component is a subgraph where every pair of nodes has
        a path between them when ignoring edge directions. This method identifies
        all such components using breadth-first search traversal.

        This is a less strict form of connectivity than strongly connected components,
        as it doesn't consider edge directions. For directed graphs where edge
        direction is important, consider using find_strongly_connected_components().

        Args:
            graph (Graph): The graph instance to analyze.

        Returns:
            List[Set[str]]: A list of sets, where each set contains the node IDs
                           belonging to a distinct weakly connected component.

        Raises:
            GraphOperationError: If the graph is empty or if there's an error during component finding.

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="C", to_entity="D", relation_type=RelationType.DEPENDS_ON)
            ... ])
            >>> components = ComponentAnalysis.find_components(graph)
            >>> # Returns [{"A", "B"}, {"C", "D"}] as two separate components

        Note:
            - Components are returned in no particular order
            - Each node appears in exactly one component
            - Isolated nodes form their own single-node components
            - Edge directions are ignored when finding components
            - For directed graphs where edge direction matters, use
              find_strongly_connected_components() instead
        """
        if not graph.get_nodes():
            return []

        try:
            components = ComponentAnalysis._find_components_impl(graph)
        except RecursionError:
            # Fall back to iterative implementation for large graphs
            components = ComponentAnalysis._find_components_iterative(graph)
        except Exception as e:
            raise GraphOperationError(f"Error finding components: {str(e)}")

        return components

    @staticmethod
    def calculate_clustering_coefficient(graph: Graph, node: str) -> float:
        """Calculate clustering coefficient for a node in a directed graph.

        The clustering coefficient measures how close the node's neighbors
        are to forming a complete graph. For directed graphs, this takes into
        account the directionality of connections between neighbors.

        In a directed graph:
        - Each pair of neighbors can have up to 2 connections (one in each direction)
        - Total possible connections is k(k-1) where k is number of neighbors
        - Actual connections count each directed edge between neighbors separately

        Args:
            graph (Graph): The graph instance containing the node.
            node (str): Node ID to calculate coefficient for.

        Returns:
            float: Clustering coefficient value between 0 and 1, where:
                  - 0 indicates no connections between neighbors
                  - 1 indicates all possible directed connections exist between neighbors
                  - For nodes with fewer than 2 neighbors, returns 0.0
                  - Returns 0.0 if node doesn't exist in graph

        Example:
            >>> graph = Graph([
            ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="B", to_entity="C", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="C", to_entity="A", relation_type=RelationType.CONNECTS_TO),
            ...     Edge(from_entity="A", to_entity="C", relation_type=RelationType.CONNECTS_TO)
            ... ])
            >>> coefficient = ComponentAnalysis.calculate_clustering_coefficient(graph, "B")
            >>> # Returns 0.5 as half of possible directed connections exist

        Note:
            - For directed graphs, each direction counts as a separate possible connection
            - Self-loops are excluded from the calculation
            - The coefficient considers both incoming and outgoing edges
            - For k neighbors, the maximum number of directed connections is k(k-1)
            - The coefficient is 0.0 for nodes with fewer than 2 neighbors
            - The order of neighbor traversal is not guaranteed due to the use of sets
        """
        if not graph.has_node(node):
            return 0.0

        # Get all neighbors (both incoming and outgoing)
        out_neighbors = graph.get_neighbors(node)
        in_neighbors = set(
            neighbor for neighbor, neighbors in graph.adjacency.items() if node in neighbors
        )
        neighbors = out_neighbors.union(in_neighbors)

        # Remove self-loops from the neighbor set
        neighbors.discard(node)

        # Return 0.0 if not enough neighbors for meaningful clustering
        if len(neighbors) < 2:
            return 0.0

        # For directed graphs, total possible connections is k(k-1)
        total_possible = len(neighbors) * (len(neighbors) - 1)
        actual_connections = 0

        # Count actual directed connections between neighbors
        # Exclude self-loops in the neighbor connections
        for neighbor1 in neighbors:
            # Get neighbors of neighbor1, excluding any self-loops
            neighbor1_neighbors = set(graph.get_neighbors(neighbor1))
            neighbor1_neighbors.discard(neighbor1)  # Remove self-loops in neighbor connections

            for neighbor2 in neighbors:
                # Only count connection if:
                # 1. Not the same node (no self-loops)
                # 2. neighbor2 is in neighbor1's neighbors (excluding self-loops)
                if neighbor1 != neighbor2 and neighbor2 in neighbor1_neighbors:
                    actual_connections += 1

        return actual_connections / total_possible if total_possible > 0 else 0.0
