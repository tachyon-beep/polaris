"""Utilities for SQLite database operations."""

import logging
import os
import sqlite3
from typing import Any

import aiosqlite

from polaris.core.exceptions import StorageError

logger = logging.getLogger(__name__)


async def initialize_table(db_path: str, schema: str) -> None:
    """
    Initialize a table in the SQLite database.

    Args:
        db_path: Path to SQLite database
        schema: SQL schema for table creation

    Raises:
        StorageError: If initialization fails
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(schema)
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to initialize table: {str(e)}")
        raise StorageError(f"Failed to initialize table: {str(e)}")


def backup_database(source_path: str, backup_dir: str, filename: str) -> None:
    """
    Create a backup of SQLite database.

    Uses SQLite's built-in backup functionality for atomic backups.

    Args:
        source_path: Path to source database
        backup_dir: Directory to store backup
        filename: Name of backup file

    Raises:
        StorageError: If backup fails
    """
    try:
        backup_path = os.path.join(backup_dir, filename)
        with (
            sqlite3.connect(source_path) as src,
            sqlite3.connect(backup_path) as dst,
        ):
            src.backup(dst)
    except Exception as e:
        logger.error(f"Failed to backup database: {str(e)}")
        raise StorageError(f"Failed to backup database: {str(e)}")


def restore_database(backup_dir: str, target_path: str, filename: str) -> None:
    """
    Restore SQLite database from backup.

    Args:
        backup_dir: Directory containing backup
        target_path: Path to target database
        filename: Name of backup file

    Raises:
        StorageError: If restore fails
    """
    try:
        backup_path = os.path.join(backup_dir, filename)
        if os.path.exists(backup_path):
            with (
                sqlite3.connect(backup_path) as src,
                sqlite3.connect(target_path) as dst,
            ):
                src.backup(dst)
    except Exception as e:
        logger.error(f"Failed to restore database: {str(e)}")
        raise StorageError(f"Failed to restore database: {str(e)}")


def row_to_tuple(row: Any) -> tuple:
    """
    Convert SQLite row to tuple.

    Args:
        row: SQLite row (tuple or Row object)

    Returns:
        Tuple of values
    """
    return tuple(row) if isinstance(row, aiosqlite.Row) else row
