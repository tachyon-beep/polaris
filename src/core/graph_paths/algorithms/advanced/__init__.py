"""
Advanced graph algorithms package.

This package provides sophisticated path finding algorithms that offer significant
performance improvements through preprocessing and other advanced techniques:

1. Contraction Hierarchies (CH):
   - Preprocessing-based speedup technique
   - Excellent for static graphs
   - Supports fast distance queries

2. Hub Labeling (HL):
   - Fast distance queries
   - Good for dense graphs
   - High preprocessing cost but very fast queries

3. Transit Node Routing (TNR):
   - Excellent for road networks
   - Uses access nodes concept
   - Very fast for long-distance queries

4. A* with Landmarks (ALT):
   - Uses triangle inequality for better heuristics
   - Good for dynamic graphs
   - Better than basic A* for sparse graphs
"""

from .contraction_hierarchies import ContractionHierarchies
from .hub_labeling import HubLabels
from .transit_node_routing import TransitNodeRouting
from .alt import ALTPathFinder

__all__ = ["ContractionHierarchies", "HubLabels", "TransitNodeRouting", "ALTPathFinder"]
