"""
Data models for Hub Labeling algorithm.

This module provides the data structures used by the Hub Labeling algorithm
to store and manage distance labels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from polaris.core.models import Edge


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


class HubLabelSet:
    """
    Set of hub labels for a node.

    This class manages a collection of labels for a single node,
    providing methods to add and query labels.
    """

    def __init__(self):
        """Initialize empty label set."""
        self.labels: List[HubLabel] = []
        self._hub_to_labels: Dict[str, List[HubLabel]] = {}

    def add_label(self, label: HubLabel) -> None:
        """
        Add label, keeping only shortest distance for each hub.

        Args:
            label: Label to add
        """
        # Check if we already have a label for this hub
        existing = None
        for idx, existing_label in enumerate(self.labels):
            if existing_label.hub == label.hub:
                existing = (idx, existing_label)
                break

        if existing:
            idx, existing_label = existing
            # Only replace if new label has shorter distance
            if label.distance < existing_label.distance:
                self.labels[idx] = label
                # Update hub_to_labels mapping
                self._hub_to_labels[label.hub] = [label]
        else:
            self.labels.append(label)
            self._hub_to_labels[label.hub] = [label]

    def get_label(self, hub: str) -> Optional[HubLabel]:
        """
        Get label for a specific hub.

        Args:
            hub: Hub node to get label for

        Returns:
            Label if it exists, None otherwise
        """
        labels = self._hub_to_labels.get(hub, [])
        return labels[0] if labels else None

    def get_distance(self, hub: str) -> Optional[float]:
        """
        Get distance to a specific hub.

        Args:
            hub: Hub node to get distance to

        Returns:
            Distance if hub exists in labels, None otherwise
        """
        label = self.get_label(hub)
        return label.distance if label else None


class HubLabelState:
    """
    Global state for Hub Labeling algorithm.

    This class manages the complete set of labels for all nodes
    in the graph, along with the hub ordering information.
    """

    MAX_ORDER = 1_000_000  # Large value for nodes without explicit order

    def __init__(self):
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
        self._hub_order[node] = order

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes that have labels.

        Returns:
            Set of node IDs
        """
        return set(self._forward_labels.keys())
