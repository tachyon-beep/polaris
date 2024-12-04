"""
Edge repository implementation for the knowledge graph.

This module implements the repository pattern for Edge objects, providing
a high-level interface for managing relationships between nodes in the
knowledge graph. It handles edge persistence, validation, and caching
while supporting advanced graph operations.

The edge repository provides:
- CRUD operations for edges
- Custom validation support
- Graph operations (path finding, connected components, neighborhood analysis)
- Edge type management
- Bidirectional relationship querying
"""

from typing import Any, Dict, List, Optional, Set

from ..core.enums import RelationType
from ..core.exceptions import ValidationError
from ..core.graph import Graph
from ..core.graph_paths import PathFinding
from ..core.graph_subgraphs import SubgraphExtraction
from ..core.graph_traversal import GraphTraversal
from ..core.models import Edge
from ..infrastructure.cache import LRUCache
from ..infrastructure.storage import StorageService
from ..utils.validation import DataIntegrityValidator
from .base import BaseRepository


class EdgeValidator:
    """
    Dedicated validator for edges.

    This class encapsulates validation logic specific to edges,
    supporting both built-in integrity checks and custom validation rules.

    Attributes:
        custom_validators (List[callable]): List of custom validation functions
    """

    def __init__(self):
        """Initialize the edge validator with empty custom validators list."""
        self.custom_validators = []

    def register_validator(self, validator_func):
        """
        Register a custom validation function.

        Args:
            validator_func: Function that takes an Edge and returns bool
        """
        self.custom_validators.append(validator_func)

    async def validate(self, edge: Edge) -> bool:
        """
        Validate an edge using built-in and custom validators.

        Args:
            edge: Edge to validate

        Returns:
            True if validation passes, False otherwise

        Raises:
            ValidationError: If validation encounters an error
        """
        try:
            # Built-in validation
            validation_result = DataIntegrityValidator.validate_edge_integrity(edge)
            if not validation_result.is_valid:
                return False

            # Custom validators
            for validator in self.custom_validators:
                if not await validator(edge):
                    return False

            return True
        except ValidationError:
            return False


class EdgeRepository(BaseRepository[Edge]):
    """
    Repository for managing Edge objects in the knowledge graph.

    This repository handles the persistence and retrieval of Edge objects,
    providing a high-level interface for relationship management while supporting
    advanced graph operations. It focuses on data access patterns while delegating
    complex graph operations to specialized services.

    Attributes:
        storage (StorageService): The underlying storage service for persistence
        cache (Optional[LRUCache]): Optional LRU cache for performance optimization
        validator (EdgeValidator): Dedicated validator for edges
    """

    def __init__(self, storage: StorageService, cache: Optional[LRUCache] = None):
        """
        Initialize the edge repository.

        Args:
            storage: Storage service for data persistence
            cache: Optional LRU cache for performance optimization
        """
        super().__init__(storage, cache)
        self.validator = EdgeValidator()

    def register_validator(self, validator_func):
        """
        Register a custom validation function.

        Args:
            validator_func: Function that takes an Edge and returns bool
        """
        self.validator.register_validator(validator_func)

    def _make_edge_id(self, from_node: str, to_node: str, relation_type: RelationType) -> str:
        """
        Create a composite ID for an edge.

        Args:
            from_node: Source node identifier
            to_node: Target node identifier
            relation_type: Type of the edge

        Returns:
            Composite ID string
        """
        return f"{from_node}:{relation_type.value}:{to_node}"

    def _parse_edge_id(self, id: str) -> Dict[str, Any]:
        """
        Parse a composite edge ID into its components.

        Args:
            id: Composite edge ID

        Returns:
            Dictionary containing from_node, to_node, and relation_type
        """
        from_node, type_value, to_node = id.split(":")
        return {
            "from_node": from_node,
            "to_node": to_node,
            "relation_type": RelationType(type_value),
        }

    async def create(self, item: Edge) -> Edge:
        """
        Create a new edge in the knowledge graph.

        Args:
            item: Edge object to create

        Returns:
            Created edge with updated metadata

        Raises:
            ValidationError: If edge validation fails
            StorageError: If there's an error with storage operations
        """
        if not await self.validator.validate(item):
            raise ValidationError(f"Edge validation failed: {item.from_entity}->{item.to_entity}")

        created_edge = await self.storage.create_edge(item)
        await self._cache_edge(created_edge)
        return created_edge

    async def get(self, id: str) -> Edge:
        """
        Retrieve an edge by its composite ID.

        Args:
            id: Composite edge identifier

        Returns:
            Retrieved edge

        Raises:
            ResourceNotFoundError: If edge doesn't exist
            StorageError: If there's an error with storage operations
        """
        cached_edge = await self._get_from_cache(id)
        if cached_edge:
            return cached_edge

        components = self._parse_edge_id(id)
        edge = await self.storage.get_edge(
            components["from_node"],
            components["to_node"],
            components["relation_type"],
        )

        await self._cache_edge(edge)
        return edge

    async def update(self, item: Edge) -> Edge:
        """
        Update an existing edge.

        Args:
            item: Edge object to update

        Returns:
            Updated edge

        Raises:
            ValidationError: If edge validation fails
            ResourceNotFoundError: If edge doesn't exist
            StorageError: If there's an error with storage operations
        """
        if not await self.validator.validate(item):
            raise ValidationError(f"Edge validation failed: {item.from_entity}->{item.to_entity}")

        updated_edge = await self.storage.update_edge(item)
        await self._cache_edge(updated_edge)
        return updated_edge

    async def delete(self, id: str) -> None:
        """
        Delete an edge by its composite ID.

        Args:
            id: Composite edge identifier

        Raises:
            ResourceNotFoundError: If edge doesn't exist
            StorageError: If there's an error with storage operations
        """
        components = self._parse_edge_id(id)
        await self.storage.delete_edge(
            components["from_node"],
            components["to_node"],
            components["relation_type"],
        )
        await self._remove_from_cache(id)

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Edge]:
        """
        List edges with optional filtering.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of edges to return
            offset: Number of edges to skip

        Returns:
            List of edges matching the criteria

        Raises:
            StorageError: If there's an error with storage operations
        """
        return await self.storage.list_edges(filters, limit, offset)

    async def validate(self, item: Edge) -> bool:
        """
        Validate an edge using the dedicated validator.

        Args:
            item: Edge to validate

        Returns:
            True if validation passes, False otherwise
        """
        return await self.validator.validate(item)

    async def get_node_edges(self, node_name: str, direction: str = "both") -> List[Edge]:
        """
        Get all edges for a specific node.

        Args:
            node_name: Name of the node
            direction: Direction of edges to retrieve ("incoming", "outgoing", or "both")

        Returns:
            List of edges involving the node

        Raises:
            StorageError: If there's an error with storage operations
        """
        filters = {}
        if direction == "outgoing":
            filters["from_entity"] = node_name
        elif direction == "incoming":
            filters["to_entity"] = node_name
        else:
            return await self.list(
                filters={"$or": [{"from_entity": node_name}, {"to_entity": node_name}]}
            )

        return await self.list(filters=filters)

    async def get_by_type(self, relation_type: RelationType) -> List[Edge]:
        """
        Get all edges of a specific type.

        Args:
            relation_type: Type of edges to retrieve

        Returns:
            List of edges of the specified type

        Raises:
            StorageError: If there's an error with storage operations
        """
        return await self.list(filters={"relation_type": relation_type})

    async def _cache_edge(self, edge: Edge) -> None:
        """
        Cache an edge using its composite ID.

        Args:
            edge: Edge to cache
        """
        if self.cache:
            edge_id = self._make_edge_id(
                edge.from_entity,
                edge.to_entity,
                edge.relation_type,
            )
            await self._set_in_cache(edge_id, edge)

    # Graph Operations

    async def find_paths(
        self, from_node: str, to_node: str, max_depth: int = 3
    ) -> List[List[Edge]]:
        """
        Find all paths between two nodes using the PathFinding service.

        This method constructs a graph from all edges and uses it to find
        paths between the specified nodes.

        Args:
            from_node: Starting node
            to_node: Target node
            max_depth: Maximum path length to consider

        Returns:
            List of paths, where each path is a list of edges

        Raises:
            StorageError: If there's an error retrieving edges
        """
        edges = await self.list()
        graph = Graph.from_edges(edges)
        path_results = list(PathFinding.all_paths(graph, from_node, to_node, max_length=max_depth))
        return [path_result.path for path_result in path_results]

    async def get_connected_components(self) -> List[Set[str]]:
        """
        Find connected components using graph traversal.

        This method identifies groups of nodes that are connected to each
        other through edges.

        Returns:
            List of sets, where each set contains node IDs in a component

        Raises:
            StorageError: If there's an error retrieving edges
        """
        edges = await self.list()
        graph = Graph.from_edges(edges)
        visited: Set[str] = set()
        components = []

        for node in graph.adjacency:
            if node not in visited:
                component = set()
                for node_id, _ in GraphTraversal.bfs(graph, node):
                    component.add(node_id)
                    visited.add(node_id)
                components.append(component)

        return components

    async def get_neighborhood(self, node: str, radius: int = 1) -> List[Edge]:
        """
        Get neighborhood subgraph around a node.

        This method retrieves all edges within a specified number of hops
        from the given node.

        Args:
            node: Central node to get neighborhood for
            radius: Number of hops to include in neighborhood

        Returns:
            List of edges in the neighborhood

        Raises:
            StorageError: If there's an error retrieving edges
        """
        edges = await self.list()
        graph = Graph.from_edges(edges)
        _, subgraph_edges = SubgraphExtraction.extract_neighborhood(graph, node, radius)
        return subgraph_edges
