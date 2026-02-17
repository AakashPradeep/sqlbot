"""
core/prompt.py

Prompt construction for Text-to-SQL generation.

Why this module exists:
- Prompts are "policy": they define what the LLM is allowed to do.
- Keeping them in a dedicated module makes them easy to tune and test.
- We separate prompt-building from UI so Streamlit/Gradio/FastAPI can reuse it.

Key rules enforced in the system prompt:
- Return ONLY SQL (no markdown fences, no extra commentary)
- Produce a single SELECT query (no DDL/DML)
- Use only schema-provided tables/columns
- Always include LIMIT if not specified
"""

from __future__ import annotations


# System-level rules sent to the LLM for consistent behavior.
# You will STILL enforce these with validators (defense in depth).
SYSTEM_RULES = """You are a Text-to-SQL assistant and follow these Rules:

- Only generate a single SELECT query and always use column names in SELECT statement (no SELECT *)
- Use only tables/columns from the schema.
- Always include a LIMIT if not provided.
- Prefer simple joins and explicit column names.
- instead of multiple sub select query use CTE 
- always add where clause with relevant filter to avoid full table scan and high latency query
- before responding reflect on the generated sql and check if it follows the above rules, if not fix it yourself and return the fixed version.
"""


def build_sql_prompt(question: str, schema_text: str, default_limit: int) -> str:
    """
    Build the SQL-generation prompt.

    Args:
      question:
        The user's natural-language query.
      schema_text:
        Prompt-friendly schema text from schema_to_text().
      default_limit:
        The LIMIT to add if user didn't request one.

    Returns:
      A prompt string for LLM SQL generation.

    Prompt structure:
      - System rules
      - Default limit value
      - Schema block
      - User question
      - "SQL:" completion cue

    Why a "SQL:" suffix:
      It helps steer the model to produce only SQL as the completion.
    """
    return f"""{SYSTEM_RULES}

DEFAULT LIMIT: {default_limit}

SCHEMA:
{schema_text}

USER QUESTION:
{question}

Return ONLY SQL. No markdown fences. No commentary.

SQL:
"""