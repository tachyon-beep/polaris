"""Shared test fixtures."""

import pytest
from datetime import datetime
from polaris.core.models import EdgeMetadata
from polaris.core.enums import RelationType


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
