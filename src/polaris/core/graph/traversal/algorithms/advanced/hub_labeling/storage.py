"""
Storage interface for Hub Labeling algorithm.

This module provides storage functionality for persisting hub labels.
"""

from typing import Dict, Optional, Set
from .models import HubLabel, HubLabelSet


class HubLabelStorage:
    """Storage for hub labels."""

    def __init__(self):
        """Initialize storage."""
        self._forward_labels: Dict[str, HubLabelSet] = {}
        self._backward_labels: Dict[str, HubLabelSet] = {}

    def save_labels(
        self,
        node: str,
        forward_labels: HubLabelSet,
        backward_labels: HubLabelSet,
    ) -> None:
        """
        Save labels for a node.

        Args:
            node: Node to save labels for
            forward_labels: Forward labels to save
            backward_labels: Backward labels to save
        """
        self._forward_labels[node] = forward_labels
        self._backward_labels[node] = backward_labels

    def get_forward_labels(self, node: str) -> Optional[HubLabelSet]:
        """
        Get forward labels for a node.

        Args:
            node: Node to get labels for

        Returns:
            Forward labels if they exist, None otherwise
        """
        return self._forward_labels.get(node)

    def get_backward_labels(self, node: str) -> Optional[HubLabelSet]:
        """
        Get backward labels for a node.

        Args:
            node: Node to get labels for

        Returns:
            Backward labels if they exist, None otherwise
        """
        return self._backward_labels.get(node)

    def clear(self) -> None:
        """Clear all stored labels."""
        self._forward_labels.clear()
        self._backward_labels.clear()

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes that have labels stored.

        Returns:
            Set of node IDs
        """
        return set(self._forward_labels.keys())
