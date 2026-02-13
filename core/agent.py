#!/usr/bin/env python3
"""
Text-to-SQL Agent — the slim orchestrator.
Delegates SQL validation, retry logic, embeddings, and DB connections to dedicated modules.
"""

import json
import logging
from typing import Dict, Any, List

from openai import OpenAI
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES
from database.connection import DatabaseConnection
from database.schema import generate_db_schema
from embeddings.client import EmbeddingClient
from core.sql_validator import validate_sql_query, is_security_issue
from core.retry import regenerate_sql_with_error_feedback
from prompts.sql_generation import create_text_to_sql_prompt
from prompts.answer_generation import create_final_answer_prompt, create_final_answer_user_message

import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentTextToSql:
    """
    Professional Text-to-SQL Agent that converts natural language queries to SQL.
    
    This agent uses OpenAI's GPT models to understand user intent and generate
    accurate SQL queries based on the database schema. It supports vector embeddings
    for semantic search capabilities.
    """

    def __init__(self, db_config: Dict[str, Any] = None, model: str = None, temperature: float = None):
        """
        Initialize the Text-to-SQL Agent.
        
        Args:
            db_config: Database configuration dictionary (optional, uses settings default)
            model: OpenAI model to use (optional, uses settings default)
            temperature: Model temperature for response generation (optional, uses settings default)
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.db_config = db_config
        self.database_schema = None

        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please add your OpenAI API key to the .env file."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized with model: {self.model}")

        # Initialize embedding client
        self.embedding_client = EmbeddingClient()

        # Load database schema
        self._load_database_schema()

    def _load_database_schema(self) -> None:
        """Generate database schema directly from database."""
        try:
            with DatabaseConnection(self.db_config) as connection:
                formatted_text, _ = generate_db_schema(connection)
                self.database_schema = formatted_text
            logger.info("Database schema generated successfully")
        except Exception as e:
            logger.error(f"Error loading database schema: {e}")
            raise

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Text-to-SQL agent."""
        return create_text_to_sql_prompt(self.database_schema)

    def generate_sql(self, user_request: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language request.
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict with 'sql_query', 'need_embedding', and 'embedding_params' fields
        """
        try:
            logger.info(f"Processing user request: {user_request}")
            system_prompt = self._create_system_prompt()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request},
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)

            # Validate response structure
            for field in ["sql_query", "need_embedding", "embedding_params"]:
                if field not in result:
                    raise ValueError(f"Missing required field '{field}' in LLM response")

            if not isinstance(result["embedding_params"], list):
                raise ValueError("embedding_params must be a list")

            if result["need_embedding"] and not result["embedding_params"]:
                raise ValueError("need_embedding is true but embedding_params is empty")

            if not result["need_embedding"] and result["embedding_params"]:
                raise ValueError("need_embedding is false but embedding_params is not empty")

            # Validate placeholder count
            if result["need_embedding"]:
                placeholder_count = result["sql_query"].count('%s')
                params_count = len(result["embedding_params"])
                if placeholder_count != params_count:
                    logger.warning(
                        f"Placeholder mismatch: SQL has {placeholder_count} %s placeholders "
                        f"but embedding_params has {params_count} entries"
                    )

            logger.info("SQL query generated successfully")
            logger.info(f"Generated SQL: {result['sql_query']}")
            logger.info(f"Needs Embedding: {result['need_embedding']}")
            if result['need_embedding']:
                logger.info(f"Embedding Parameters: {len(result['embedding_params'])} parameter(s)")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise

    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request and return structured response (without execution).
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict containing the generated SQL, need_embedding flag, embedding_params, and metadata
        """
        try:
            result = self.generate_sql(user_request)
            return {
                "success": True,
                "user_request": user_request,
                "sql_query": result["sql_query"],
                "need_embedding": result["need_embedding"],
                "embedding_params": result["embedding_params"],
                "model_used": self.model,
            }
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "user_request": user_request,
                "error": str(e),
                "model_used": self.model,
                "need_embedding": None,
                "embedding_params": [],
            }

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the loaded database schema."""
        return {
            "schema_loaded": self.database_schema is not None,
            "schema_length": len(self.database_schema) if self.database_schema else 0,
            "model": self.model,
            "temperature": self.temperature,
        }

    def execute_sql(self, sql_query: str, need_embedding: bool = False,
                    embedding_params: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute SQL query against the database with validation.
        
        Args:
            sql_query: SQL query to execute
            need_embedding: Whether the query needs embedding parameters
            embedding_params: List of embedding parameter dictionaries
            
        Returns:
            Dict containing query results and metadata
        """
        try:
            # SECURITY: Validate SQL query before execution
            logger.info("Validating SQL query for security...")
            is_valid, error_message = validate_sql_query(sql_query)

            if not is_valid:
                logger.error(f"SQL validation failed: {error_message}")
                return {
                    "success": False,
                    "error": f"Query validation failed: {error_message}",
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "validation_failed": True,
                    "is_security_issue": is_security_issue(error_message),
                }

            logger.info("Connecting to database for query execution...")
            connection = psycopg2.connect(**(self.db_config or {})) if self.db_config else None
            if connection is None:
                from config.settings import DB_CONFIG
                connection = psycopg2.connect(**DB_CONFIG)
            cursor = connection.cursor()

            # Generate embeddings if needed
            query_params = []
            if need_embedding:
                if not embedding_params:
                    raise ValueError("embedding_params required when need_embedding is True")

                logger.info(f"Generating {len(embedding_params)} embedding(s) for query...")
                embeddings_generated = self.embedding_client.generate_for_params(embedding_params)

                placeholder_count = sql_query.count('%s')
                if placeholder_count > len(embeddings_generated):
                    logger.info(f"Replicating {len(embeddings_generated)} embeddings to match {placeholder_count} placeholders")
                    query_params = []
                    for i in range(placeholder_count):
                        query_params.append(embeddings_generated[i % len(embeddings_generated)])
                else:
                    query_params = embeddings_generated

            # Execute query
            placeholder_count = sql_query.count('%s')
            logger.info("Executing SQL query...")

            if query_params and len(query_params) != placeholder_count:
                error_msg = f"Parameter count mismatch: query expects {placeholder_count} but got {len(query_params)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if query_params:
                try:
                    query_with_params = sql_query
                    for param in query_params:
                        query_with_params = query_with_params.replace('%s::vector', f"'{param}'::vector", 1)
                    cursor.execute(query_with_params)
                except psycopg2.Error as e:
                    logger.error(f"PostgreSQL execution error: {e}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {e}")
            else:
                try:
                    cursor.execute(sql_query)
                except psycopg2.Error as e:
                    logger.error(f"PostgreSQL execution error: {e}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {e}")

            # Fetch results
            try:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []

                results_list = []
                for row in results:
                    row_dict = {}
                    for i, col_name in enumerate(column_names):
                        row_dict[col_name] = row[i]
                    results_list.append(row_dict)

                logger.info(f"Query executed successfully. Retrieved {len(results_list)} row(s)")
                cursor.close()
                connection.close()

                return {
                    "success": True,
                    "results": results_list,
                    "column_names": column_names,
                    "row_count": len(results_list),
                }

            except psycopg2.ProgrammingError:
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                connection.close()
                logger.info(f"Query executed successfully. {affected_rows} row(s) affected")
                return {
                    "success": True,
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "affected_rows": affected_rows,
                }

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            if 'connection' in locals() and connection:
                connection.rollback()
                connection.close()
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "column_names": [],
                "row_count": 0,
            }

    def generate_final_answer(self, user_request: str, query_results: Dict[str, Any],
                              sql_query: str = None) -> str:
        """
        Generate a natural language answer based on the user request and query results.
        """
        try:
            logger.info("Generating final natural language answer...")

            if not query_results.get('success', False):
                results_text = f"Error executing query: {query_results.get('error', 'Unknown error')}"
            elif query_results['row_count'] == 0:
                results_text = "No results found."
            else:
                results_text = f"Found {query_results['row_count']} result(s):\n\n"
                for i, row in enumerate(query_results['results'][:20], 1):
                    results_text += f"Result {i}:\n"
                    for key, value in row.items():
                        if not key.endswith('_embed') and key != 'similarity' and key != 'combined_similarity':
                            results_text += f"  - {key}: {value}\n"
                    results_text += "\n"
                if query_results['row_count'] > 20:
                    results_text += f"... and {query_results['row_count'] - 20} more results\n"

            system_prompt = create_final_answer_prompt()
            user_message = create_final_answer_user_message(user_request, results_text, sql_query)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {e}"

    def process_request_with_execution(self, user_request: str, max_retries: int = None) -> Dict[str, Any]:
        """
        Complete pipeline with retry mechanism: Generate SQL, execute, and generate final answer.
        """
        max_retries = max_retries or MAX_RETRIES
        attempt = 0
        last_error = None
        sql_result = None
        attempt_history = []

        while attempt < max_retries:
            try:
                attempt += 1
                logger.info("=" * 80)
                logger.info(f"ATTEMPT {attempt}/{max_retries}")
                logger.info("=" * 80)

                # Step 1: Generate SQL query
                if attempt == 1:
                    logger.info("STEP 1: GENERATING SQL QUERY")
                    sql_result = self.generate_sql(user_request)
                else:
                    logger.info(f"STEP 1: REGENERATING SQL QUERY (Attempt {attempt})")
                    sql_result = regenerate_sql_with_error_feedback(
                        client=self.client,
                        model=self.model,
                        temperature=self.temperature,
                        system_prompt=self._create_system_prompt(),
                        user_request=user_request,
                        attempt_history=attempt_history,
                    )

                # Step 2: Execute SQL query
                logger.info("STEP 2: EXECUTING SQL QUERY")
                query_results = self.execute_sql(
                    sql_query=sql_result['sql_query'],
                    need_embedding=sql_result['need_embedding'],
                    embedding_params=sql_result['embedding_params'],
                )

                if not query_results.get('success', False):
                    last_error = query_results.get('error', 'Unknown execution error')
                    logger.warning(f"Attempt {attempt} failed: {last_error}")
                    attempt_history.append({
                        'sql': sql_result['sql_query'],
                        'error': last_error,
                    })

                    if query_results.get('validation_failed', False):
                        if query_results.get('is_security_issue', True):
                            logger.error("SECURITY ISSUE detected - aborting retries")
                            break
                        logger.warning("Validation failed but error is fixable - will retry")
                    continue

                # Step 3: Generate final answer
                logger.info("STEP 3: GENERATING FINAL ANSWER")
                final_answer = self.generate_final_answer(
                    user_request=user_request,
                    query_results=query_results,
                    sql_query=sql_result['sql_query'],
                )

                logger.info(f"PIPELINE COMPLETED SUCCESSFULLY (Attempt {attempt})")

                return {
                    "success": True,
                    "user_request": user_request,
                    "sql_query": sql_result['sql_query'],
                    "need_embedding": sql_result['need_embedding'],
                    "embedding_params": sql_result['embedding_params'],
                    "query_results": query_results,
                    "final_answer": final_answer,
                    "model_used": self.model,
                    "attempts": attempt,
                    "failed_attempts": attempt_history,
                }

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt} error: {last_error}")
                if sql_result:
                    attempt_history.append({
                        'sql': sql_result.get('sql_query', 'Query generation failed'),
                        'error': last_error,
                    })
                if attempt < max_retries:
                    continue
                break

        # All attempts failed
        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        return {
            "success": False,
            "user_request": user_request,
            "error": f"Failed after {attempt} attempts. Last error: {last_error}",
            "model_used": self.model,
            "attempts": attempt,
            "last_sql_query": sql_result['sql_query'] if sql_result else None,
            "failed_attempts": attempt_history,
        }
