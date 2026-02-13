#!/usr/bin/env python3
"""
Database connection manager for SemanticText2SQL.
Provides a reusable context manager for PostgreSQL connections.
"""

import logging
import psycopg2
from typing import Dict, Any

from config.settings import DB_CONFIG

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Context manager for PostgreSQL database connections.

    Usage:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
    """

    def __init__(self, db_config: Dict[str, Any] = None):
        """
        Initialize with optional custom DB config.

        Args:
            db_config: Override default DB_CONFIG from settings if provided.
        """
        self.db_config = db_config or DB_CONFIG
        self.connection = None

    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            logger.info("Database connection established")
            return self.connection
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.debug("Database connection closed")

    def __enter__(self):
        self.connect()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.connection:
            self.connection.rollback()
        self.close()
        return False
