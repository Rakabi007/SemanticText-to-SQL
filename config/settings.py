#!/usr/bin/env python3
"""
Centralized configuration for SemanticText2SQL.
Single source of truth for database, OpenAI, and model settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'books_db'),
    'user': os.getenv('DB_USER', 'bookadmin'),
    'password': os.getenv('DB_PASSWORD', 'bookpass123'),
}

# -------------------------------------------------------------------
# OpenAI Configuration
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# LLM model
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4.1')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))

# -------------------------------------------------------------------
# Agent Configuration
# -------------------------------------------------------------------
MAX_RETRIES = 4
MAX_RESULT_ROWS = 100
MAX_TOKENS_SQL = 1000
MAX_TOKENS_ANSWER = 1000
