"""
Core domain models package for the knowledge graph system.

This package provides the fundamental data structures and models
that represent nodes, edges, and their metadata in the knowledge graph.
"""

from .base import validate_metric_range, validate_date_order, validate_custom_metrics
from .edge import Edge, EdgeMetadata
from .metadata import BaseMetadata, MetadataProvider
from .node import Node, NodeMetadata, NodeMetrics

__all__ = [
    # Base utilities
    "validate_metric_range",
    "validate_date_order",
    "validate_custom_metrics",
    # Node models
    "Node",
    "NodeMetadata",
    "NodeMetrics",
    # Edge models
    "Edge",
    "EdgeMetadata",
    # Metadata models
    "BaseMetadata",
    "MetadataProvider",
]
