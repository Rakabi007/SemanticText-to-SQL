#!/usr/bin/env python3
"""
Answer generation prompts for the Text-to-SQL agent.
Handles creating natural language answers from query results.
"""


def create_final_answer_prompt() -> str:
    """
    Create the system prompt for generating final natural language answers.
    
    Returns:
        str: System prompt for answer generation
    """
    return """You are a helpful assistant that translates database query results into clear, natural language answers.

Your task is to:
1. Understand the user's original question
2. Analyze the query results
3. Provide a clear, concise, and accurate answer in natural language

Guidelines:
- Be direct and answer the question specifically
- Use natural, conversational language
- If there are multiple results, summarize them clearly
- If there are no results, explain what that means
- Don't mention technical details like SQL or database operations unless relevant
- Focus on the information the user wants to know"""


def create_final_answer_user_message(user_request: str, results_text: str, sql_query: str = None) -> str:
    """
    Create the user message for final answer generation.
    
    Args:
        user_request: Original user request
        results_text: Formatted query results text
        sql_query: The SQL query that was executed (optional)
        
    Returns:
        str: User message for answer generation
    """
    message = f"""User's Question: {user_request}

Query Results:
{results_text}

Please provide a clear, natural language answer to the user's question based on these results."""
    
    # Add SQL context if provided
    if sql_query:
        message += f"\n\nFor context, the SQL query used was:\n{sql_query}"
    
    return message
