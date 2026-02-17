from __future__ import annotations

FIX_PROMPT = """You are fixing a SQL query for a database.
Return ONLY corrected SQL (single SELECT). No markdown.

SCHEMA:
{schema}

POLICY:
- SELECT only
- Use only allowed tables/columns
- Keep intent the same
- Add/keep LIMIT

USER QUESTION:
{question}

BROKEN SQL:
{sql}

ERROR:
{error}

CORRECTED SQL:
"""

def fix_sql(llm, schema_text: str, question: str, sql: str, error: str) -> str:
    prompt = FIX_PROMPT.format(schema=schema_text, question=question, sql=sql, error=error)
    return llm.invoke(prompt).content.strip()
