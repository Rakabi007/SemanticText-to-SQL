#!/usr/bin/env python3
"""
Retry logic for the Text-to-SQL agent.
Handles regenerating SQL queries with comprehensive error feedback.
"""

import json
import logging
from typing import Dict, Any, List

from prompts.retry_prompt import create_sql_retry_prompt

logger = logging.getLogger(__name__)


def build_error_history_text(attempt_history: List[Dict[str, str]]) -> str:
    """
    Format previous attempt history into a readable text block for the LLM.

    Args:
        attempt_history: List of dicts with 'sql' and 'error' keys.

    Returns:
        Formatted string with all previous attempts.
    """
    lines = []
    for i, prev_attempt in enumerate(attempt_history, 1):
        lines.append(f"""
ATTEMPT {i}:
SQL Query: {prev_attempt['sql']}
Error: {prev_attempt['error']}
---""")
    return "".join(lines)


def regenerate_sql_with_error_feedback(
    client,
    model: str,
    temperature: float,
    system_prompt: str,
    user_request: str,
    attempt_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Regenerate SQL query by providing comprehensive error feedback to the LLM.

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        temperature: Model temperature.
        system_prompt: The system prompt (with schema).
        user_request: Original user request.
        attempt_history: List of previous attempts with their SQL and errors.

    Returns:
        Dict with regenerated 'sql_query', 'need_embedding', 'embedding_params'.
    """
    logger.info(f"Regenerating SQL with error feedback from {len(attempt_history)} previous attempt(s)...")

    error_history_text = build_error_history_text(attempt_history)
    user_message = create_sql_retry_prompt(user_request, error_history_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content.strip()
        result = json.loads(response_text)

        # Validate response structure
        for field in ["sql_query", "need_embedding", "embedding_params"]:
            if field not in result:
                raise ValueError(f"Missing required field '{field}' in LLM response")

        return result

    except Exception as e:
        logger.error(f"Error regenerating SQL: {e}")
        raise
