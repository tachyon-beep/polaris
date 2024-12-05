"""
Tests for core model base utilities.
"""

from datetime import datetime, timedelta

import pytest

from polaris.core.models import validate_metric_range, validate_date_order, validate_custom_metrics


def test_validate_metric_range():
    """Test metric range validation utility."""
    # Valid cases
    validate_metric_range("test", 0.5)  # Default range 0-1
    validate_metric_range("test", 5, 0, 10)  # Custom range

    # Invalid cases
    with pytest.raises(ValueError):
        validate_metric_range("test", 1.5)  # Outside default range
    with pytest.raises(ValueError):
        validate_metric_range("test", 15, 0, 10)  # Outside custom range


def test_validate_date_order():
    """Test date order validation utility."""
    now = datetime.now()
    earlier = now - timedelta(days=1)
    later = now + timedelta(days=1)

    # Valid case
    validate_date_order(earlier, now)
    validate_date_order(now, later)

    # Invalid case
    with pytest.raises(ValueError):
        validate_date_order(now, earlier)


def test_validate_custom_metrics():
    """Test custom metrics validation utility."""
    valid_metrics = {"metric1": (0.5, 0, 1), "metric2": (5, 0, 10)}
    validate_custom_metrics(valid_metrics)

    invalid_metrics = {
        "metric1": (1.5, 0, 1),  # Value outside range
        "metric2": (5, 10, 0),  # Invalid range (min > max)
    }
    with pytest.raises(ValueError):
        validate_custom_metrics(invalid_metrics)
