"""
Tests for metadata models and protocols.
"""

from datetime import datetime

import pytest

from polaris.core.models import BaseMetadata, MetadataProvider


def test_base_metadata():
    """Test BaseMetadata functionality."""
    now = datetime.now()
    metadata = BaseMetadata(created_at=now, last_modified=now, source="test_source")

    assert metadata.source == "test_source"
    assert metadata.created_at == now
    assert metadata.last_modified == now
    assert isinstance(metadata.custom_attributes, dict)
    assert len(metadata.custom_attributes) == 0


def test_base_metadata_custom_attributes():
    """Test BaseMetadata custom attributes functionality."""
    metadata = BaseMetadata(
        created_at=datetime.now(), last_modified=datetime.now(), source="test_source"
    )

    # Add custom attribute
    metadata.add_custom_attribute("key1", "value1")
    assert "key1" in metadata.custom_attributes
    assert metadata.custom_attributes["key1"] == "value1"

    # Add another custom attribute
    metadata.add_custom_attribute("key2", {"nested": "value"})
    assert "key2" in metadata.custom_attributes
    assert metadata.custom_attributes["key2"]["nested"] == "value"

    # Remove custom attribute
    metadata.remove_custom_attribute("key1")
    assert "key1" not in metadata.custom_attributes
    assert "key2" in metadata.custom_attributes

    # Remove non-existent attribute (should not raise error)
    metadata.remove_custom_attribute("non_existent")

    # Test update_modified
    original_modified = metadata.last_modified
    metadata.update_modified()
    assert metadata.last_modified > original_modified


def test_base_metadata_validation():
    """Test BaseMetadata validation."""
    now = datetime.now()

    # Test empty source
    with pytest.raises(ValueError):
        BaseMetadata(created_at=now, last_modified=now, source="")

    # Test whitespace source
    with pytest.raises(ValueError):
        BaseMetadata(created_at=now, last_modified=now, source="   ")


def test_metadata_provider_protocol():
    """Test MetadataProvider protocol compliance."""

    class TestMetadata:
        @property
        def created_at(self) -> datetime:
            return datetime.now()

        @property
        def last_modified(self) -> datetime:
            return datetime.now()

        @property
        def source(self) -> str:
            return "test"

    # Should not raise TypeError
    test_metadata = TestMetadata()
    assert isinstance(test_metadata, MetadataProvider)

    # Test non-compliant class
    class NonCompliantMetadata:
        def created_at(self) -> datetime:  # Not a property
            return datetime.now()

    non_compliant = NonCompliantMetadata()
    assert not isinstance(non_compliant, MetadataProvider)
