#!/usr/bin/env python3
"""
Batch embedding generator for SemanticText2SQL.
Scans database tables for _embed columns and generates embeddings for corresponding text fields.
"""

import sys
import time
import logging
from typing import Dict, List, Any

from database.connection import DatabaseConnection
from embeddings.client import EmbeddingClient

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and store embeddings for database text fields."""

    def __init__(self, db_config: Dict[str, Any] = None):
        """
        Initialize the embedding generator.

        Args:
            db_config: Optional custom DB config. Uses default from settings if not provided.
        """
        self.embedding_client = EmbeddingClient()
        self.db_connection = DatabaseConnection(db_config)
        self.connection = self.db_connection.connect()
        self.total_embeddings = 0
        self.total_cost_estimate = 0.0
        self.tables_with_embeddings = self._discover_embedding_fields()

    def _discover_embedding_fields(self) -> Dict[str, List[tuple]]:
        """
        Automatically discover all tables and fields with embeddings.

        Returns:
            Dict mapping table names to list of (text_field, embed_field, id_field) tuples.
        """
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        tables_with_embeddings = {}

        for table in tables:
            # Get all columns for this table
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table,))

            columns = cursor.fetchall()

            # Find embedding fields (fields ending with _embed)
            embed_fields = [col[0] for col in columns if col[0].endswith('_embed')]

            if not embed_fields:
                continue

            # Get primary key for this table
            cursor.execute("""
                SELECT column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position;
            """, (table,))

            pk_result = cursor.fetchone()
            if not pk_result:
                continue

            id_field = pk_result[0]

            # Map each embedding field to its source text field
            field_mappings = []
            for embed_field in embed_fields:
                # Remove _embed suffix to get text field name
                text_field = embed_field[:-6]  # Remove '_embed'

                # Verify text field exists
                if any(col[0] == text_field for col in columns):
                    field_mappings.append((text_field, embed_field, id_field))

            if field_mappings:
                tables_with_embeddings[table] = field_mappings

        cursor.close()
        return tables_with_embeddings

    def get_rows_to_process(self, table_name: str, text_field: str,
                            embed_field: str, id_field: str) -> List[Dict[str, Any]]:
        """Get all rows that need embeddings generated."""
        cursor = self.connection.cursor()

        query = f"""
            SELECT {id_field}, {text_field}
            FROM {table_name}
            WHERE {text_field} IS NOT NULL 
              AND {embed_field} IS NULL
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [{'id': row[0], 'text': row[1]} for row in rows]

    def update_embedding(self, table_name: str, embed_field: str,
                         id_field: str, row_id: int, embedding: List[float]) -> bool:
        """Update a single row with its embedding."""
        cursor = self.connection.cursor()

        try:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            query = f"""
                UPDATE {table_name}
                SET {embed_field} = %s::vector
                WHERE {id_field} = %s
            """

            cursor.execute(query, (embedding_str, row_id))
            self.connection.commit()
            cursor.close()
            return True

        except Exception as e:
            print(f"Error updating embedding: {e}")
            self.connection.rollback()
            cursor.close()
            return False

    def process_table_field(self, table_name: str, text_field: str,
                            embed_field: str, id_field: str):
        """Process all rows for a specific table and field combination."""
        print(f"\nProcessing {table_name}.{text_field} → {embed_field}")
        print("-" * 60)

        rows = self.get_rows_to_process(table_name, text_field, embed_field, id_field)

        if not rows:
            print(f"  No rows to process (all embeddings already generated)")
            return

        print(f"  Found {len(rows)} rows to process")

        for i, row in enumerate(rows, 1):
            try:
                embedding = self.embedding_client.generate(row['text'])
            except Exception:
                print(f"  [{i}/{len(rows)}] Skipped {id_field}={row['id']} (empty text or error)")
                continue

            if embedding:
                self.total_cost_estimate += self.embedding_client.estimate_cost(row['text'])
                success = self.update_embedding(table_name, embed_field, id_field, row['id'], embedding)

                if success:
                    self.total_embeddings += 1
                    print(f"  [{i}/{len(rows)}] Updated {id_field}={row['id']}")
                else:
                    print(f"  [{i}/{len(rows)}] Failed to update {id_field}={row['id']}")

            # Small delay to avoid rate limits
            if i % 10 == 0:
                time.sleep(0.5)

    def process_all_tables(self):
        """Process all tables with embedding fields."""
        from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

        print("=" * 80)
        print("GENERATING EMBEDDINGS FOR ALL TABLES")
        print("=" * 80)
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
        print("=" * 80)

        # Show discovered tables and fields
        print("\nDiscovered embedding fields:")
        for table_name, fields in self.tables_with_embeddings.items():
            print(f"  {table_name}: {len(fields)} field(s)")
            for text_field, embed_field, id_field in fields:
                print(f"    - {text_field} → {embed_field}")
        print("=" * 80)

        # Process each table
        for table_name, fields in self.tables_with_embeddings.items():
            for text_field, embed_field, id_field in fields:
                self.process_table_field(table_name, text_field, embed_field, id_field)

        # Summary
        print("\n" + "=" * 80)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total embeddings generated: {self.total_embeddings}")
        print(f"Estimated cost: ${self.total_cost_estimate:.4f}")
        print("=" * 80)

    def close(self):
        """Close database connection."""
        self.db_connection.close()
        print("\nDatabase connection closed.")
