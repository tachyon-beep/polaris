"""Graph serialization and deserialization operations.

This module provides functionality for importing/exporting graphs:
- JSON serialization
- Custom format handling
- Metadata preservation
- Validation during import/export
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Union

from ..enums import EntityType, RelationType
from ..exceptions import GraphOperationError
from ..models import Edge, EdgeMetadata, Node, NodeMetadata


class GraphSerializer:
    """Handles graph serialization operations."""

    def __init__(self, edges: List[Edge], nodes: Optional[List[Node]] = None):
        """Initialize serializer.

        Args:
            edges: List of edges in the graph
            nodes: Optional list of node objects (for metadata)
        """
        self.edges = edges
        self.nodes = nodes or []
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate graph data for serialization.

        Raises:
            GraphOperationError: If data validation fails
        """
        # Validate edges
        node_ids = {node.id for node in self.nodes} if self.nodes else set()
        edge_nodes = {edge.from_node for edge in self.edges} | {edge.to_node for edge in self.edges}

        # If nodes were provided, ensure all edge endpoints exist
        if self.nodes and not edge_nodes.issubset(node_ids):
            missing = edge_nodes - node_ids
            raise GraphOperationError(f"Edges reference nonexistent nodes: {missing}")

        # Validate edge types
        for edge in self.edges:
            if not isinstance(edge.metadata.relation_type, RelationType):
                raise GraphOperationError(
                    f"Invalid relation type for edge {edge.from_node}->{edge.to_node}: "
                    f"{edge.metadata.relation_type}"
                )

        # Validate node types if nodes provided
        for node in self.nodes:
            if not isinstance(node.metadata.entity_type, EntityType):
                raise GraphOperationError(
                    f"Invalid entity type for node {node.id}: {node.metadata.entity_type}"
                )

    def to_dict(self) -> Dict:
        """Convert graph to dictionary format.

        Returns:
            Dictionary containing serialized graph data
        """

        def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
            """Convert datetime to ISO format string."""
            return dt.isoformat() if dt else None

        def serialize_metadata(metadata: Union[EdgeMetadata, NodeMetadata]) -> Dict:
            """Convert metadata to serializable format."""
            base_dict = {
                "created": serialize_datetime(metadata.created),
                "last_modified": serialize_datetime(metadata.last_modified),
                "version": metadata.version,
            }
            # Add remaining attributes while filtering out None values
            metadata_dict = {
                k: v for k, v in metadata.__dict__.items() if k not in base_dict and v is not None
            }
            return {**base_dict, **metadata_dict}

        # Serialize edges
        edges_data = [
            {
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "metadata": serialize_metadata(edge.metadata),
            }
            for edge in self.edges
        ]

        # Serialize nodes if present
        nodes_data = (
            [{"id": node.id, "metadata": serialize_metadata(node.metadata)} for node in self.nodes]
            if self.nodes
            else []
        )

        return {
            "schema_version": "1.0",
            "created": datetime.now().isoformat(),
            "nodes": nodes_data,
            "edges": edges_data,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert graph to JSON string.

        Args:
            indent: Number of spaces for pretty printing (default: None)

        Returns:
            JSON string representation of graph
        """
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Dict) -> tuple[List[Edge], List[Node]]:
        """Create graph from dictionary data.

        Args:
            data: Dictionary containing graph data

        Returns:
            Tuple of (edges, nodes) lists

        Raises:
            GraphOperationError: If data format is invalid
        """

        def deserialize_datetime(dt_str: Optional[str]) -> Optional[datetime]:
            """Convert ISO format string to datetime."""
            return datetime.fromisoformat(dt_str) if dt_str else None

        def deserialize_edge_metadata(metadata: Dict) -> EdgeMetadata:
            """Convert dictionary to EdgeMetadata."""
            return EdgeMetadata(
                relation_type=RelationType(metadata.get("relation_type", "generic")),
                created=deserialize_datetime(metadata.get("created")),
                last_modified=deserialize_datetime(metadata.get("last_modified")),
                version=metadata.get("version", 1),
                weight=metadata.get("weight", 1.0),
                properties=metadata.get("properties", {}),
            )

        def deserialize_node_metadata(metadata: Dict) -> NodeMetadata:
            """Convert dictionary to NodeMetadata."""
            return NodeMetadata(
                entity_type=EntityType(metadata.get("entity_type", "generic")),
                created=deserialize_datetime(metadata.get("created")),
                last_modified=deserialize_datetime(metadata.get("last_modified")),
                version=metadata.get("version", 1),
                properties=metadata.get("properties", {}),
            )

        try:
            # Deserialize edges
            edges = [
                Edge(
                    from_node=edge_data["from_node"],
                    to_node=edge_data["to_node"],
                    metadata=deserialize_edge_metadata(edge_data["metadata"]),
                )
                for edge_data in data.get("edges", [])
            ]

            # Deserialize nodes
            nodes = [
                Node(id=node_data["id"], metadata=deserialize_node_metadata(node_data["metadata"]))
                for node_data in data.get("nodes", [])
            ]

            return edges, nodes

        except (KeyError, ValueError) as e:
            raise GraphOperationError(f"Invalid graph data format: {str(e)}")

    @staticmethod
    def from_json(json_str: str) -> tuple[List[Edge], List[Node]]:
        """Create graph from JSON string.

        Args:
            json_str: JSON string containing graph data

        Returns:
            Tuple of (edges, nodes) lists

        Raises:
            GraphOperationError: If JSON is invalid
        """
        try:
            data = json.loads(json_str)
            return GraphSerializer.from_dict(data)
        except json.JSONDecodeError as e:
            raise GraphOperationError(f"Invalid JSON format: {str(e)}")

    def to_adjacency_list(self) -> Dict[str, Set[str]]:
        """Convert graph to adjacency list format.

        Returns:
            Dictionary mapping node IDs to sets of neighbor IDs
        """
        adj_list: Dict[str, Set[str]] = {}

        for edge in self.edges:
            if edge.from_node not in adj_list:
                adj_list[edge.from_node] = set()
            adj_list[edge.from_node].add(edge.to_node)

            # Ensure isolated target nodes appear in adjacency list
            if edge.to_node not in adj_list:
                adj_list[edge.to_node] = set()

        return adj_list

    def to_edge_list(self) -> List[tuple[str, str, Dict]]:
        """Convert graph to edge list format.

        Returns:
            List of (from_node, to_node, attributes) tuples
        """
        return [
            (
                edge.from_node,
                edge.to_node,
                {
                    "relation_type": edge.metadata.relation_type.value,
                    "weight": edge.metadata.weight,
                    **edge.metadata.properties,
                },
            )
            for edge in self.edges
        ]

    @staticmethod
    def from_edge_list(
        edges: List[tuple[str, str, Dict]], default_relation: str = "generic"
    ) -> List[Edge]:
        """Create graph from edge list format.

        Args:
            edges: List of (from_node, to_node, attributes) tuples
            default_relation: Default relation type value

        Returns:
            List of Edge objects
        """
        return [
            Edge(
                from_node=from_node,
                to_node=to_node,
                metadata=EdgeMetadata(
                    relation_type=RelationType(attrs.get("relation_type", default_relation)),
                    weight=float(attrs.get("weight", 1.0)),
                    properties={
                        k: v for k, v in attrs.items() if k not in {"relation_type", "weight"}
                    },
                ),
            )
            for from_node, to_node, attrs in edges
        ]
