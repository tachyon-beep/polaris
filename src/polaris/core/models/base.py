"""
Core domain models base module for the knowledge graph system.

This module provides common imports and base functionality used across
the different model types in the knowledge graph system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...utils.validation import validate_dataclass
from ..enums import EntityType, RelationType


# Common validation functions
def validate_date_order(created_at: datetime, last_modified: datetime) -> None:
    """Validate that last_modified is not before created_at."""
    if last_modified < created_at:
        raise ValueError("last modified cannot be before created_at")


def validate_metric_range(
    name: str, value: float, min_range: float = 0.0, max_range: float = 1.0
) -> None:
    """Validate that a metric value falls within the specified range."""
    if not min_range <= value <= max_range:
        raise ValueError(f"{name} value {value} must be between {min_range} and {max_range}")


def validate_custom_metrics(custom_metrics: Dict[str, Tuple[float, float, float]]) -> None:
    """Validate custom metrics and their ranges."""
    for metric_name, (value, min_range, max_range) in custom_metrics.items():
        if min_range >= max_range:
            raise ValueError(f"Invalid range for {metric_name}: min must be less than max")
        if not min_range <= value <= max_range:
            raise ValueError(
                f"Custom metric {metric_name} value {value} must be "
                f"between {min_range} and {max_range}"
            )
