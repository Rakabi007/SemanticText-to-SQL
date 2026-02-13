#!/usr/bin/env python3
"""
Retry prompt for the Text-to-SQL agent.
Provides error feedback to the LLM for SQL query regeneration.
"""


def create_sql_retry_prompt(user_request: str, error_history_text: str) -> str:
    """
    Create the user message for SQL query regeneration after failures.
    
    Args:
        user_request: Original user request
        error_history_text: Formatted text with history of all failed attempts
        
    Returns:
        str: User message with error context for regeneration
    """
    return f"""Original request: {user_request}

ALL PREVIOUS ATTEMPTS HAVE FAILED. Here is the complete history:
{error_history_text}

CRITICAL INSTRUCTIONS:
1. Analyze ALL previous attempts and their specific errors
2. DO NOT repeat the same mistakes from previous attempts
3. If multiple attempts failed with the same type of error, try a completely different approach
4. Generate a CORRECTED SQL query that addresses ALL the errors seen so far

Common issues to check:
- Syntax errors (check PostgreSQL syntax carefully)
- Missing or incorrect table/column names (verify against schema)
- Incorrect JOINs (ensure proper relationships)
- Type mismatches (especially with vector types)
- Missing WHERE clauses for NULL checks on embedding fields
- Placeholder count mismatch (ensure embedding_params matches %s count in query)
- If you need to reuse the same embedding multiple times in a query, you MUST list it multiple times in embedding_params
- Vector columns in GROUP BY: You CANNOT include vector columns (ending in _embed) in GROUP BY

Learn from previous failures and generate a query that will execute successfully."""
