"""
core/schema.py

Database schema introspection utilities.

Why this module exists:
- Text-to-SQL needs grounding: the model must know what tables/columns exist.
- We use SQLAlchemy's Inspector API so this works across many databases:
  SQLite, Postgres, MySQL, etc.

What it provides:
- load_schema(engine, allow_tables=...): fetch table + column metadata
- schema_to_text(schema): convert schema into a compact, prompt-friendly string

Notes:
- For large databases, you may not want to expose the full schema to the model.
  Use policy allow-lists / regex to restrict or sample tables.
- This module is intentionally minimal and UI-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.core.policies import CompiledPolicy


@dataclass(frozen=True)
class TableInfo:
    """
    Simple immutable container for table metadata.

    Fields:
      name:
        Table name as reported by SQLAlchemy inspector.

      columns:
        List of (column_name, column_type_string).
        We store types as strings because:
          - they're easy to display in prompts
          - they vary per DB dialect
    """
    name: str
    columns: List[Tuple[str, str]]


def load_schema(engine: Engine, raw_policy: CompiledPolicy) -> List[TableInfo]:
    """
    Introspect a database and return table/column metadata.

    Args:
      engine:
        SQLAlchemy Engine connected to the target database.
      allow_tables:
        Optional list of allowed tables. If provided, only these tables
        are included in the output. This is useful to limit prompt size.

    Returns:
      A list of TableInfo entries for tables in the database.

    Implementation details:
      - Uses SQLAlchemy inspector to get table names and columns.
      - Column types are converted to strings for readability.
    """
    insp = inspect(engine)
    tables = insp.get_table_names()

    # If an allowlist is provided, filter tables accordingly.
    filtered_tables =  filter_table_by_policy(tables, raw_policy)

    out: List[TableInfo] = []
    for t in filtered_tables:
        cols = [(c["name"], str(c["type"])) for c in insp.get_columns(t)]
        out.append(TableInfo(name=t, columns=cols))

    return out


def schema_to_text(schema: List[TableInfo]) -> str:
    """
    Convert TableInfo objects into a compact schema string suitable for LLM prompts.

    Format example:
      - customers(customer_id:INTEGER, first_name:VARCHAR, ...)
      - invoices(invoice_id:INTEGER, customer_id:INTEGER, ...)

    Args:
      schema: list of TableInfo

    Returns:
      A newline-separated schema string.
    """
    lines = []
    for t in schema:
        cols = ", ".join([f"{c}:{typ}" for c, typ in t.columns])
        lines.append(f"- {t.name}({cols})")
    return "\n".join(lines)
  

def filter_table_by_policy(tables: List[str], policy: CompiledPolicy) -> List[str]:
    """
    Filter a list of table names according to the allow/deny rules in the policy.

    This is a helper function that can be used by load_schema() if you want to
    filter tables at the schema loading stage. Alternatively, you can load the
    full schema and let the model see it, while enforcing allow/deny rules only
    at validation time.

    Args:
      tables: list of table names to filter
      policy: CompiledPolicy with allow/deny rules

    Returns:
      A filtered list of table names that are allowed by the policy.
    """
    # This function is not implemented here because we enforce policies at validation time.
    # You could implement it if you want to pre-filter tables at load time.
    filtered_tables = []
    for t in tables:
        norm = t.lower()
        # Apply deny rules first (deny overrides allow)
        if norm in policy.deny_tables or any(rx.search(norm) for rx in policy.deny_table_regex):
            continue
        # Apply allow rules
        if norm in policy.allow_tables or any(rx.search(norm) for rx in policy.allow_table_regex):
            filtered_tables.append(t)
    return filtered_tables