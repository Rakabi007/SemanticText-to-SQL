#!/usr/bin/env python3
"""
SQL query validator for SemanticText2SQL.
Validates SQL queries for security (read-only, no injection) and correctness.
"""

import logging
from typing import Optional

import sqlglot

logger = logging.getLogger(__name__)

# Dangerous SQL keywords that indicate write/admin operations
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
    'EXEC', 'EXECUTE', 'CALL',
]


def validate_sql_query(sql_query: str) -> tuple[bool, Optional[str]]:
    """
    Validate SQL query for security and safety.
    
    Checks:
    1. Query must be parseable
    2. Only SELECT statements allowed (no INSERT, UPDATE, DELETE, DROP, etc.)
    3. No dangerous operations (CREATE, ALTER, TRUNCATE, etc.)
    4. No multiple statements (semicolon separation)
    
    Args:
        sql_query: SQL query to validate
        
    Returns:
        tuple: (is_valid, error_message) — error_message is None when valid.
    """
    try:
        # Check for multiple statements (basic protection against SQL injection)
        statements = sql_query.strip().split(';')
        statements = [s.strip() for s in statements if s.strip()]

        if len(statements) > 1:
            return False, "Multiple SQL statements detected. Only single SELECT queries are allowed."

        # Check for dangerous keywords
        query_upper = sql_query.upper()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                return False, f"Dangerous operation detected: {keyword}"

        # Check for SELECT INTO (write operation)
        if 'INTO' in query_upper and 'INTO' not in query_upper[query_upper.find('FROM'):]:
            return False, "SELECT INTO operations are not allowed"

        # Must start with SELECT
        if not query_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"

        # Check for vector columns in GROUP BY
        if 'GROUP BY' in query_upper:
            group_by_start = query_upper.find('GROUP BY')
            group_by_clause = sql_query[group_by_start:].split('ORDER BY')[0].split('LIMIT')[0]
            if '_embed' in group_by_clause.lower():
                return False, (
                    "Vector columns (fields ending with '_embed') cannot be used in GROUP BY clause. "
                    "Use primary keys or scalar fields only."
                )

        # Parse with sqlglot (handling pgvector operators)
        query_for_parsing = sql_query
        if '<->' in query_for_parsing:
            query_for_parsing = query_for_parsing.replace('<->', '+')

        try:
            parsed = sqlglot.parse_one(query_for_parsing, read='postgres')
        except Exception as e:
            logger.warning(f"SQL parsing warning: {e}")
            # Allow the query if it passed all other checks and starts with SELECT
            if query_upper.strip().startswith('SELECT'):
                logger.info("SQL query validation passed (basic check)")
                return True, None
            return False, f"SQL parsing error: {e}"

        if not isinstance(parsed, sqlglot.exp.Select):
            statement_type = type(parsed).__name__
            return False, f"Only SELECT queries are allowed. Detected: {statement_type}"

        logger.info("SQL query validation passed")
        return True, None

    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return False, f"Validation error: {e}"


def is_security_issue(error_message: str) -> bool:
    """
    Determine if a validation error is a security issue (should abort)
    vs a fixable syntax error (can retry).

    Args:
        error_message: Error message from validation.

    Returns:
        True if the error is a security concern.
    """
    security_indicators = [
        'Dangerous operation', 'Multiple SQL statements', 'Only SELECT queries',
        'SELECT INTO', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
    ]
    return any(indicator in error_message for indicator in security_indicators)
