"""Utilities for persisting data to JSON files."""

import json
import logging
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict

import aiofiles

from ......core.exceptions import StorageError
from ...utils import serialize_datetime

logger = logging.getLogger(__name__)


async def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary of loaded data

    Raises:
        StorageError: If loading fails
    """
    try:
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                if content.strip():
                    return json.loads(content)
        return {}
    except Exception as e:
        logger.error(f"Failed to load from {file_path}: {str(e)}")
        raise StorageError(f"Failed to load from {file_path}: {str(e)}")


async def save_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """
    Save data to a JSON file atomically.

    Args:
        file_path: Path to JSON file
        data: Dictionary of data to save

    Raises:
        StorageError: If saving fails
    """
    try:
        # Convert data to JSON-serializable format with datetime handling
        serialized_data = {
            key: asdict(
                value,
                dict_factory=lambda x: {k: serialize_datetime(v) for k, v in x},
            )
            for key, value in data.items()
        }

        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(serialized_data, indent=2))
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {str(e)}")
        raise StorageError(f"Failed to save to {file_path}: {str(e)}")


def backup_file(source_path: str, backup_dir: str, filename: str) -> None:
    """
    Create a backup of a file.

    Args:
        source_path: Path to source file
        backup_dir: Directory to store backup
        filename: Name of backup file

    Raises:
        StorageError: If backup fails
    """
    try:
        if os.path.exists(source_path):
            backup_file = os.path.join(backup_dir, filename)
            shutil.copy2(source_path, backup_file)
    except Exception as e:
        logger.error(f"Failed to backup {source_path}: {str(e)}")
        raise StorageError(f"Failed to backup {source_path}: {str(e)}")
