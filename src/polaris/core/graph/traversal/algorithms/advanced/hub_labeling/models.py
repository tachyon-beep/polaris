"""
Data models for Hub Labeling algorithm.

This module provides the data structures used by the Hub Labeling algorithm
to store and manage distance labels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TypeVar, Generic
import math

from polaris.core.models import Edge

T = TypeVar("T")


@dataclass
class HubLabel:
    """
    Distance label for a node.

    Represents the distance from a source node to a hub node,
    along with the first edge on the path to that hub.
    """

    hub: str
    distance: float
    first_hop: Optional[Edge]

    def __post_init__(self) -> None:
        """Validate label attributes."""
        if not isinstance(self.distance, (int, float)):
            raise ValueError("Distance must be numeric")
        if self.distance < 0:
            raise ValueError("Distance cannot be negative")
        if math.isinf(self.distance):
            raise ValueError("Distance cannot be infinite")
        if not isinstance(self.hub, str):
            raise ValueError("Hub must be a string")
        if not self.hub:
            raise ValueError("Hub cannot be empty")


class HubLabelSet:
    """
    Set of hub labels for a node.

    This class manages a collection of labels for a single node,
    providing methods to add and query labels.
    """

    def __init__(self) -> None:
        """Initialize empty label set."""
        self.labels: List[HubLabel] = []
        self._hub_to_labels: Dict[str, List[HubLabel]] = {}
        self._hub_to_index: Dict[str, int] = {}  # For O(1) lookups

    def add_label(self, label: HubLabel) -> None:
        """
        Add label, keeping only shortest distance for each hub.
        Ensures atomic updates and proper deduplication.

        Args:
            label: Label to add
        """
        # Validate label
        if not isinstance(label, HubLabel):
            raise ValueError("Label must be a HubLabel instance")

        # Get existing index if any
        existing_idx = self._hub_to_index.get(label.hub)

        if existing_idx is not None:
            existing_label = self.labels[existing_idx]
            # Only replace if new label has shorter distance
            if label.distance < existing_label.distance:
                # Atomic update
                self.labels[existing_idx] = label
                self._hub_to_labels[label.hub] = [label]
        else:
            # Add new label
            self.labels.append(label)
            self._hub_to_index[label.hub] = len(self.labels) - 1
            self._hub_to_labels[label.hub] = [label]

    def get_label(self, hub: str) -> Optional[HubLabel]:
        """
        Get label for a specific hub.
        O(1) lookup using index mapping.

        Args:
            hub: Hub node to get label for

        Returns:
            Label if it exists, None otherwise
        """
        idx = self._hub_to_index.get(hub)
        if idx is not None:
            return self.labels[idx]
        return None

    def get_distance(self, hub: str) -> Optional[float]:
        """
        Get distance to a specific hub.
        O(1) lookup using index mapping.

        Args:
            hub: Hub node to get distance to

        Returns:
            Distance if hub exists in labels, None otherwise
        """
        label = self.get_label(hub)
        return label.distance if label else None

    def remove_label(self, hub: str) -> None:
        """
        Remove label for a specific hub.
        Ensures atomic updates across all data structures.

        Args:
            hub: Hub to remove label for
        """
        idx = self._hub_to_index.get(hub)
        if idx is not None:
            # Remove from all data structures atomically
            self.labels.pop(idx)
            del self._hub_to_labels[hub]
            del self._hub_to_index[hub]
            # Update indices for remaining labels
            for h, i in self._hub_to_index.items():
                if i > idx:
                    self._hub_to_index[h] = i - 1


class HubLabelState:
    """
    Global state for Hub Labeling algorithm.

    This class manages the complete set of labels for all nodes
    in the graph, along with the hub ordering information.
    """

    MAX_ORDER = 1_000_000  # Large value for nodes without explicit order

    def __init__(self) -> None:
        """Initialize empty state."""
        self._forward_labels: Dict[str, HubLabelSet] = {}
        self._backward_labels: Dict[str, HubLabelSet] = {}
        self._hub_order: Dict[str, int] = {}

    def get_forward_labels(self, node: str) -> HubLabelSet:
        """
        Get forward labels for a node.

        Args:
            node: Node to get labels for

        Returns:
            Forward label set for node
        """
        if node not in self._forward_labels:
            self._forward_labels[node] = HubLabelSet()
        return self._forward_labels[node]

    def get_backward_labels(self, node: str) -> HubLabelSet:
        """
        Get backward labels for a node.

        Args:
            node: Node to get labels for

        Returns:
            Backward label set for node
        """
        if node not in self._backward_labels:
            self._backward_labels[node] = HubLabelSet()
        return self._backward_labels[node]

    def get_hub_order(self, node: str) -> int:
        """
        Get ordering value for a node.

        Args:
            node: Node to get order for

        Returns:
            Order value (lower = more important)
        """
        return self._hub_order.get(node, self.MAX_ORDER)

    def set_hub_order(self, node: str, order: int) -> None:
        """
        Set ordering value for a node.

        Args:
            node: Node to set order for
            order: Order value (lower = more important)
        """
        if not isinstance(order, int):
            raise ValueError("Order must be an integer")
        if order < 0:
            raise ValueError("Order cannot be negative")
        self._hub_order[node] = order

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes that have labels.

        Returns:
            Set of node IDs
        """
        return set(self._forward_labels.keys())

    def validate_state(self) -> None:
        """
        Validate internal state consistency.
        Raises ValueError if state is invalid.
        """
        for node, label_set in self._forward_labels.items():
            for label in label_set.labels:
                if label.hub not in self._hub_order:
                    raise ValueError(f"Hub {label.hub} missing from ordering")

        for node, label_set in self._backward_labels.items():
            for label in label_set.labels:
                if label.hub not in self._hub_order:
                    raise ValueError(f"Hub {label.hub} missing from ordering")
