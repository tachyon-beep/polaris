"""
Polaris - Graph-based Data Analysis and Processing Framework

This package provides a comprehensive suite of tools and utilities for graph-based
data analysis, processing, and storage. It includes:

- Core graph processing and analysis capabilities
- Infrastructure components for data storage and event handling
- Repository patterns for data access
- Search functionality including semantic search capabilities
- Utility functions and validation tools

For more information, please see the documentation.
"""

__version__ = "0.1.0"
__author__ = "Polaris Team"
__license__ = "See LICENSE file"

# Version compatibility check
import sys

if sys.version_info < (3, 12):
    raise RuntimeError("Polaris requires Python 3.12 or higher")

# Import commonly used components for easier access
from .core.graph import Graph
from .core.models import Edge, Node

__all__ = [
    "Graph",
    "Node",
    "Edge",
]
