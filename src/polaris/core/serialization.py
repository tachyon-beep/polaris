from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import json
import pickle
from enum import Enum

from .graph import Graph
from .models import Edge, EdgeMetadata


# Define RelationType here since import failed
class RelationType(Enum):
    RELATED = "related"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    REFERENCES = "references"


class GraphSerializer(ABC):
    @abstractmethod
    def serialize(self, graph: Graph) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Graph:
        pass


class JSONGraphSerializer(GraphSerializer):
    def serialize(self, graph: Graph) -> bytes:
        serialized_data = {
            "nodes": list(graph.get_nodes()),
            "edges": [
                {
                    "from": edge.from_entity,
                    "to": edge.to_entity,
                    "type": edge.relation_type.value,
                    "metadata": edge.metadata.__dict__,
                }
                for edge in graph.get_edges()
            ],
        }
        return json.dumps(serialized_data).encode("utf-8")

    def deserialize(self, data: bytes) -> Graph:
        graph_data = json.loads(data.decode("utf-8"))
        edges = [
            Edge(
                from_entity=e["from"],
                to_entity=e["to"],
                relation_type=RelationType(e["type"]),
                metadata=EdgeMetadata(**e["metadata"]),
            )
            for e in graph_data["edges"]
        ]
        return Graph(edges)


class PickleGraphSerializer(GraphSerializer):
    def serialize(self, graph: Graph) -> bytes:
        return pickle.dumps(graph)

    def deserialize(self, data: bytes) -> Graph:
        return pickle.loads(data)


class SerializationRegistry:
    _serializers: Dict[str, Type[GraphSerializer]] = {}

    @classmethod
    def register(cls, format_name: str, serializer: Type[GraphSerializer]):
        cls._serializers[format_name] = serializer

    @classmethod
    def get_serializer(cls, format_name: str) -> GraphSerializer:
        if format_name not in cls._serializers:
            raise ValueError(f"No serializer registered for format: {format_name}")
        return cls._serializers[format_name]()


# Register default serializers
SerializationRegistry.register("json", JSONGraphSerializer)
SerializationRegistry.register("pickle", PickleGraphSerializer)
