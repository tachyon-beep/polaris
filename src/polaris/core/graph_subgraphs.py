"""
Subgraph extraction operations for the knowledge graph.

This module provides functionality for extracting meaningful subgraphs from the
knowledge graph based on various criteria:
- Neighborhood extraction around specific nodes
- Type-based filtering of nodes and edges
- Custom filtering using callback functions

The subgraph extraction capabilities are valuable for:
- Analyzing local graph structure around nodes of interest
- Reducing graph complexity for specific analyses
- Focusing on relevant portions of the graph
- Creating meaningful visualizations
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

from .graph import Graph
from .graph_traversal import GraphTraversal
from .models import Edge, Node

NodeFilterFunc = Callable[[str], bool]
EdgeFilterFunc = Callable[[Edge], bool]


class SubgraphExtraction:
    """
    Subgraph extraction operations for knowledge graphs.

    This class provides static methods for extracting subgraphs based on various
    criteria such as node neighborhoods, types, and custom filters. The extracted
    subgraphs maintain the structural properties of the original graph while
    focusing on specific areas of interest.
    """

    @staticmethod
    def _collect_neighborhood_nodes(
        graph: Graph,
        center_node: str,
        radius: int,
        filter_func: Optional[NodeFilterFunc],
    ) -> Set[str]:
        """
        Collect nodes within neighborhood radius.

        Uses breadth-first search to identify all nodes within a specified
        distance from the center node, optionally filtered by a custom function.

        Args:
            graph (Graph): The graph instance to analyze
            center_node (str): Central node to build neighborhood around
            radius (int): Maximum distance from center node
            filter_func (Optional[NodeFilterFunc]): Optional function to filter nodes

        Returns:
            Set[str]: Set of node IDs within the neighborhood

        Note:
            - Distance is measured in number of edges (hops)
            - Includes nodes exactly at the radius distance
            - Filter function can exclude nodes even within radius
            - If filter_func excludes the center node, only the center node is returned
        """
        if filter_func and not filter_func(center_node):
            return {center_node}  # Return only center node if it's filtered out

        nodes = {center_node}
        for node, depth in GraphTraversal.bfs(graph, center_node, max_depth=radius):
            if filter_func is None or filter_func(node):
                nodes.add(node)
        return nodes

    @staticmethod
    def _get_neighborhood_edge_key(from_node: str, to_node: str) -> Tuple[str, str]:
        """
        Generate a unique key for an edge pair.

        Creates a consistent unique identifier for an undirected edge
        between two nodes by ordering the node IDs.

        Args:
            from_node (str): First node ID
            to_node (str): Second node ID

        Returns:
            Tuple[str, str]: Ordered pair of node IDs
        """
        return (min(from_node, to_node), max(from_node, to_node))

    @staticmethod
    def _get_edge_key(edge: Edge) -> Tuple[str, str, str]:
        """
        Generate a unique key for an edge.

        Creates a consistent unique identifier for a directed edge.

        Args:
            edge (Edge): Edge to generate key for

        Returns:
            Tuple[str, str, str]: Tuple of (from_node, to_node, relation_type)
        """
        return (edge.from_entity, edge.to_entity, edge.relation_type.name)

    @staticmethod
    def _is_valid_neighborhood_edge(
        edge: Edge,
        neighbor: str,
        nodes: Set[str],
        filter_func: Optional[EdgeFilterFunc],
    ) -> bool:
        """
        Check if an edge should be included in the neighborhood.

        Validates an edge against multiple criteria:
        - Edge must exist
        - Both nodes must be in the collected nodes set
        - Edge must pass the optional filter function

        Args:
            edge (Edge): Edge to validate
            neighbor (str): Neighbor node ID
            nodes (Set[str]): Set of valid node IDs
            filter_func (Optional[EdgeFilterFunc]): Optional edge filter

        Returns:
            bool: True if edge should be included, False otherwise
        """
        # Both nodes must be in the valid nodes set
        if edge.from_entity not in nodes or edge.to_entity not in nodes:
            return False

        # Apply filter function if provided
        if filter_func is not None:
            return filter_func(edge)
        return True

    @staticmethod
    def _collect_neighborhood_edges(
        graph: Graph,
        nodes: Set[str],
        filter_func: Optional[EdgeFilterFunc],
    ) -> List[Edge]:
        """
        Collect edges between neighborhood nodes.

        Gathers all edges between nodes in the neighborhood set,
        avoiding duplicates and applying optional filtering.

        Args:
            graph (Graph): The graph instance
            nodes (Set[str]): Set of nodes to consider
            filter_func (Optional[EdgeFilterFunc]): Optional function to filter edges

        Returns:
            List[Edge]: List of unique edges between the nodes
        """
        edges = []
        seen_edges = set()

        for node in nodes:
            for neighbor in graph.get_neighbors(node):
                edge = graph.get_edge(node, neighbor)
                if edge:
                    edge_key = SubgraphExtraction._get_edge_key(edge)
                    if (
                        edge_key not in seen_edges
                        and SubgraphExtraction._is_valid_neighborhood_edge(
                            edge, neighbor, nodes, filter_func
                        )
                    ):
                        edges.append(edge)
                        seen_edges.add(edge_key)

        return edges

    @staticmethod
    def extract_neighborhood(
        graph: Graph,
        center_node: str,
        radius: int = 1,
        node_filter: Optional[NodeFilterFunc] = None,
        edge_filter: Optional[EdgeFilterFunc] = None,
    ) -> Tuple[Set[str], List[Edge]]:
        """
        Extract neighborhood subgraph around a node.

        Creates a subgraph containing all nodes within a specified radius
        of a center node, along with their interconnecting edges.

        Args:
            graph (Graph): The graph instance
            center_node (str): Central node to build neighborhood around
            radius (int): Maximum distance from center node (default: 1)
            node_filter (Optional[NodeFilterFunc]): Optional function to filter nodes
            edge_filter (Optional[EdgeFilterFunc]): Optional function to filter edges

        Returns:
            Tuple[Set[str], List[Edge]]: Tuple containing:
                - Set of node IDs in the neighborhood
                - List of edges between those nodes

        Example:
            >>> def node_filter(node_id: str) -> bool:
            ...     return not node_id.startswith("temp_")
            >>> def edge_filter(edge: Edge) -> bool:
            ...     return edge.type != "TEMPORARY"
            >>> nodes, edges = SubgraphExtraction.extract_neighborhood(
            ...     graph, "node_123",
            ...     radius=2,
            ...     node_filter=node_filter,
            ...     edge_filter=edge_filter
            ... )

        Note:
            - If node_filter excludes the center node, only the center node is returned
              with no edges
            - Edge filtering is applied after node filtering
            - Both nodes of an edge must be in the filtered node set
            - When radius is 0, returns only the center node with no edges
        """
        # Handle radius 0 case - return only center node with no edges
        if radius == 0:
            return {center_node}, []

        nodes = SubgraphExtraction._collect_neighborhood_nodes(
            graph, center_node, radius, node_filter
        )
        edges = SubgraphExtraction._collect_neighborhood_edges(graph, nodes, edge_filter)
        return nodes, edges

    @staticmethod
    def _filter_nodes(
        nodes: List[Node],
        entity_types: Optional[Set[str]],
    ) -> List[Node]:
        """
        Filter nodes by type.

        Filters a list of nodes to include only those of specified types.

        Args:
            nodes (List[Node]): List of nodes to filter
            entity_types (Optional[Set[str]]): Set of allowed entity type names

        Returns:
            List[Node]: Filtered list of nodes
        """
        if not entity_types:
            return nodes
        return [n for n in nodes if n.entity_type.name in entity_types]

    @staticmethod
    def _is_valid_edge(
        edge: Edge,
        node_types: Dict[str, str],
        filtered_node_ids: Set[str],
        entity_types: Optional[Set[str]],
        relation_types: Optional[Set[str]],
    ) -> bool:
        """
        Check if an edge meets the filtering criteria.

        Validates an edge against type and node membership criteria.

        Args:
            edge (Edge): Edge to validate
            node_types (Dict[str, str]): Mapping of node IDs to their types
            filtered_node_ids (Set[str]): Set of valid node IDs
            entity_types (Optional[Set[str]]): Set of allowed entity type names
            relation_types (Optional[Set[str]]): Set of allowed relation type names

        Returns:
            bool: True if edge meets criteria, False otherwise

        Note:
            - When entity_types is specified, both nodes must be of the specified types
            - When relation_types is specified, the edge's relation type must match
            - Both conditions must be met when both filters are specified
        """
        # Check relation type if specified
        if relation_types and edge.relation_type.name not in relation_types:
            return False

        # If entity types are specified, both nodes must be of allowed type
        if entity_types:
            from_type = node_types.get(edge.from_entity)
            to_type = node_types.get(edge.to_entity)
            if not from_type or not to_type:
                return False
            if from_type not in entity_types or to_type not in entity_types:
                return False

        return True

    @staticmethod
    def extract_by_type(
        graph: Graph,
        nodes: List[Node],
        entity_types: Optional[Set[str]] = None,
        relation_types: Optional[Set[str]] = None,
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Extract subgraph based on entity and relation types.

        Creates a subgraph containing only nodes and edges of specified types.
        If no types are specified, returns the entire graph.

        Args:
            graph (Graph): The graph instance
            nodes (List[Node]): List of all nodes
            entity_types (Optional[Set[str]]): Set of entity types to include
            relation_types (Optional[Set[str]]): Set of relation types to include

        Returns:
            Tuple[List[Node], List[Edge]]: Tuple containing:
                - List of filtered nodes
                - List of filtered edges

        Example:
            >>> entity_types = {"CODE_MODULE", "CODE_FUNCTION"}
            >>> relation_types = {"CALLS", "IMPORTS"}
            >>> nodes, edges = SubgraphExtraction.extract_by_type(
            ...     graph, all_nodes,
            ...     entity_types=entity_types,
            ...     relation_types=relation_types
            ... )

        Note:
            - If neither entity_types nor relation_types is provided,
              returns the complete graph
            - When entity_types is specified, both nodes of an edge must be
              of the specified types
            - When relation_types is specified, the edge's relation type must match
            - Both conditions must be met when both filters are specified
        """
        # If no filters specified, return complete graph
        if not entity_types and not relation_types:
            edges = []
            seen_keys = set()
            for from_node, neighbors in graph.adjacency.items():
                for to_node in neighbors:
                    edge = graph.get_edge(from_node, to_node)
                    if edge:
                        edge_key = SubgraphExtraction._get_edge_key(edge)
                        if edge_key not in seen_keys:
                            edges.append(edge)
                            seen_keys.add(edge_key)
            return nodes, edges

        # Create a mapping of node IDs to their types for efficient lookup
        node_types = {node.name: node.entity_type.name for node in nodes}

        # Filter nodes if entity types specified
        filtered_nodes = SubgraphExtraction._filter_nodes(nodes, entity_types)
        filtered_node_ids = {node.name for node in filtered_nodes}

        # Filter edges
        filtered_edges = []
        seen_keys = set()

        for from_node, neighbors in graph.adjacency.items():
            for to_node in neighbors:
                edge = graph.get_edge(from_node, to_node)
                if edge:
                    edge_key = SubgraphExtraction._get_edge_key(edge)
                    if edge_key not in seen_keys and SubgraphExtraction._is_valid_edge(
                        edge,
                        node_types,
                        filtered_node_ids,
                        entity_types,
                        relation_types,
                    ):
                        filtered_edges.append(edge)
                        seen_keys.add(edge_key)

        return filtered_nodes, filtered_edges
