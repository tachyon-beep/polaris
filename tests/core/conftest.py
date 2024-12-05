"""Shared test fixtures."""

from datetime import datetime

import pytest

from polaris.core.enums import RelationType
from polaris.core.models import EdgeMetadata


@pytest.fixture
def sample_edge_metadata() -> EdgeMetadata:
    """Fixture providing sample edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test",
        weight=1.0,
        bidirectional=False,
        temporal=False,
        custom_attributes={},
    )
