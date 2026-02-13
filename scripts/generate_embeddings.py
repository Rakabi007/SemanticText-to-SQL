#!/usr/bin/env python3
"""
CLI entry point for batch embedding generation.
Scans the database for text fields with corresponding _embed columns
and generates embeddings using OpenAI.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.generator import EmbeddingGenerator


def main():
    print("=" * 80)
    print("EMBEDDING GENERATOR")
    print("=" * 80)

    try:
        generator = EmbeddingGenerator()
        generator.process_all_tables()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        if 'generator' in locals():
            generator.close()


if __name__ == "__main__":
    main()
