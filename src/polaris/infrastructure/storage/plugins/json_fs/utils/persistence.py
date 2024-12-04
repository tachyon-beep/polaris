"""Utilities for persisting JSON data to the filesystem.

This module provides functions for loading and saving JSON data to files,
with support for backup and error handling.
"""

import json
import os
import shutil
from typing import Any, Dict, Optional


def backup_file(file_path: str) -> None:
    """Create a backup copy of a file.

    Args:
        file_path (str): Path to the file to backup.

    Raises:
        OSError: If backup creation fails.
    """
    if not os.path.exists(file_path):
        return

    backup_path = f"{file_path}.bak"
    try:
        shutil.copy2(file_path, backup_path)
    except OSError as e:
        raise OSError(f"Failed to create backup file: {e}")


async def load_json_file(
    file_path: str, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load JSON data from a file asynchronously.

    Args:
        file_path (str): Path to the JSON file.
        default (Optional[Dict[str, Any]], optional): Default value if file doesn't exist.
            Defaults to None.

    Returns:
        Dict[str, Any]: Loaded JSON data or default value.

    Raises:
        OSError: If file reading fails.
        json.JSONDecodeError: If JSON parsing fails.
    """
    if not os.path.exists(file_path):
        if default is not None:
            return default
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e.msg}", e.doc, e.pos)
    except OSError as e:
        raise OSError(f"Failed to read file {file_path}: {e}")


async def save_json_file(file_path: str, data: Dict[str, Any], create_backup: bool = True) -> None:
    """Save JSON data to a file asynchronously.

    Args:
        file_path (str): Path where to save the JSON file.
        data (Dict[str, Any]): Data to save.
        create_backup (bool, optional): Whether to create a backup of existing file.
            Defaults to True.

    Raises:
        OSError: If file saving fails.
        TypeError: If data is not JSON serializable.
    """
    if create_backup:
        backup_file(file_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except (OSError, TypeError) as e:
        raise OSError(f"Failed to save file {file_path}: {e}")
