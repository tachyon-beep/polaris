from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Iterator
from datetime import datetime
import json

from .graph import Graph
from .models import Edge, EdgeMetadata
from .enums import RelationType  # Updated import path


class DeltaType(Enum):
    """Types of graph delta operations."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


@dataclass
class GraphDelta:
    """Represents a single change in the graph."""

    operation: DeltaType
    from_entity: str
    to_entity: str
    edge: Optional[Edge] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert delta to dictionary for serialization."""
        return {
            "operation": self.operation.value,
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "edge": (
                {
                    "from_entity": self.edge.from_entity,
                    "to_entity": self.edge.to_entity,
                    "relation_type": self.edge.relation_type.value,
                    "metadata": self.edge.metadata.__dict__,
                }
                if self.edge
                else None
            ),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GraphDelta":
        """Create delta from dictionary representation."""
        edge_data = data.get("edge")
        edge = None
        if edge_data:
            edge = Edge(
                from_entity=edge_data["from_entity"],
                to_entity=edge_data["to_entity"],
                relation_type=RelationType(edge_data["relation_type"]),
                metadata=EdgeMetadata(**edge_data["metadata"]),
            )

        return cls(
            operation=DeltaType(data["operation"]),
            from_entity=data["from_entity"],
            to_entity=data["to_entity"],
            edge=edge,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class DeltaTrackingGraph(Graph):
    """Graph implementation that tracks all changes."""

    def __init__(self, edges: Optional[List[Edge]] = None):
        super().__init__(edges if edges is not None else [])
        self._deltas: List[GraphDelta] = []
        self._snapshots: Dict[datetime, Set[Edge]] = {}

    def add_edge(self, edge: Edge) -> None:
        """Add an edge and track the change."""
        super().add_edge(edge)
        self._deltas.append(
            GraphDelta(
                operation=DeltaType.ADD,
                from_entity=edge.from_entity,
                to_entity=edge.to_entity,
                edge=edge,
            )
        )

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge and track the change."""
        # Find the edge before removing it
        edge = next(
            (e for e in self.get_edges() if e.from_entity == from_node and e.to_entity == to_node),
            None,
        )

        super().remove_edge(from_node, to_node)

        self._deltas.append(
            GraphDelta(
                operation=DeltaType.REMOVE, from_entity=from_node, to_entity=to_node, edge=edge
            )
        )

    def modify_edge(self, edge: Edge) -> None:
        """Modify an existing edge and track the change."""
        old_edge = next(
            (
                e
                for e in self.get_edges()
                if e.from_entity == edge.from_entity and e.to_entity == edge.to_entity
            ),
            None,
        )

        if not old_edge:
            raise ValueError(f"Edge {edge.from_entity} -> {edge.to_entity} not found")

        # Remove old edge and add new one
        self.remove_edge(edge.from_entity, edge.to_entity)
        self.add_edge(edge)

        self._deltas.append(
            GraphDelta(
                operation=DeltaType.MODIFY,
                from_entity=edge.from_entity,
                to_entity=edge.to_entity,
                edge=edge,
                metadata={"previous_type": old_edge.relation_type.value},
            )
        )

    def create_snapshot(self) -> datetime:
        """Create a snapshot of the current graph state."""
        timestamp = datetime.now()
        self._snapshots[timestamp] = set(self.get_edges())
        return timestamp

    def restore_snapshot(self, timestamp: datetime) -> None:
        """Restore the graph to a previous snapshot."""
        if timestamp not in self._snapshots:
            raise ValueError(f"No snapshot found for timestamp {timestamp}")

        # Clear current graph and restore from snapshot
        edges = list(self.get_edges())
        for edge in edges:
            self.remove_edge(edge.from_entity, edge.to_entity)

        for edge in self._snapshots[timestamp]:
            self.add_edge(edge)

    def get_deltas_since(self, timestamp: datetime) -> List[GraphDelta]:
        """Get all changes since a specific timestamp."""
        return [delta for delta in self._deltas if delta.timestamp > timestamp]

    def get_delta_summary(self, start_time: Optional[datetime] = None) -> Dict[str, int]:
        """Get a summary of changes since start_time."""
        deltas = self.get_deltas_since(start_time) if start_time else self._deltas

        return {
            "total_changes": len(deltas),
            "additions": sum(1 for d in deltas if d.operation == DeltaType.ADD),
            "removals": sum(1 for d in deltas if d.operation == DeltaType.REMOVE),
            "modifications": sum(1 for d in deltas if d.operation == DeltaType.MODIFY),
        }

    def export_deltas(self, filepath: str) -> None:
        """Export deltas to a JSON file."""
        with open(filepath, "w") as f:
            json.dump([delta.to_dict() for delta in self._deltas], f, indent=2)

    def import_deltas(self, filepath: str) -> None:
        """Import deltas from a JSON file."""
        with open(filepath, "r") as f:
            delta_dicts = json.load(f)

        self._deltas = [GraphDelta.from_dict(delta_dict) for delta_dict in delta_dicts]

    def replay_deltas(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Graph:
        """Replay a sequence of deltas to reconstruct a graph state."""
        graph = Graph([])  # Initialize with empty edge list
        deltas = [
            d
            for d in self._deltas
            if (not start_time or d.timestamp >= start_time)
            and (not end_time or d.timestamp <= end_time)
        ]

        for delta in deltas:
            if delta.operation == DeltaType.ADD and delta.edge:
                graph.add_edge(delta.edge)
            elif delta.operation == DeltaType.REMOVE:
                try:
                    graph.remove_edge(delta.from_entity, delta.to_entity)
                except ValueError:
                    pass  # Edge might have been removed already
            elif delta.operation == DeltaType.MODIFY and delta.edge:
                try:
                    graph.remove_edge(delta.from_entity, delta.to_entity)
                except ValueError:
                    pass  # Edge might have been removed
                graph.add_edge(delta.edge)

        return graph

    def get_change_frequency(self, interval_minutes: int = 60) -> Dict[datetime, int]:
        """Get the frequency of changes over time intervals."""
        if not self._deltas:
            return {}

        from collections import defaultdict

        frequency = defaultdict(int)

        # Round timestamps to intervals
        for delta in self._deltas:
            interval = delta.timestamp.replace(
                minute=delta.timestamp.minute // interval_minutes * interval_minutes,
                second=0,
                microsecond=0,
            )
            frequency[interval] += 1

        return dict(sorted(frequency.items()))
