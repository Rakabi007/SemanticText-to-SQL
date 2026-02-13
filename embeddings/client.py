#!/usr/bin/env python3
"""
Shared OpenAI embedding client for SemanticText2SQL.
Used by both the agent (real-time) and the batch generator.
"""

import logging
from typing import List

from openai import OpenAI
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Shared OpenAI embedding client.

    Generates vector embeddings for text using OpenAI's embedding models.
    Provides both single-text and batch embedding generation.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the embedding client.

        Args:
            api_key: OpenAI API key. Falls back to config if not provided.
        """
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please add your OpenAI API key to the .env file."
            )
        self.client = OpenAI(api_key=key)
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_for_params(self, embedding_params: List[dict]) -> List[str]:
        """
        Generate embeddings for a list of embedding parameter dicts.

        Each param dict must have a 'text_to_embed' key.

        Args:
            embedding_params: List of dicts with 'text_to_embed' values.

        Returns:
            List of PostgreSQL-formatted vector strings.
        """
        embeddings = []
        for param in embedding_params:
            text = param['text_to_embed']
            vector = self.generate(text)
            # Format as PostgreSQL vector literal
            embedding_str = '[' + ','.join(map(str, vector)) + ']'
            embeddings.append(embedding_str)
        return embeddings

    def estimate_cost(self, text: str) -> float:
        """
        Estimate the cost for embedding a text.

        Args:
            text: Input text.

        Returns:
            Estimated cost in USD.
        """
        # text-embedding-3-small: $0.020 per 1M tokens
        # Rough estimate: ~1 token per 4 characters
        tokens = len(text) / 4
        return (tokens / 1_000_000) * 0.020
