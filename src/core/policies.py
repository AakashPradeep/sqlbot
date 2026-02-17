"""
core/policies.py

Policy loading + compilation for Safe Text-to-SQL.

Why this exists:
- YAML is user-friendly, but you want fast, consistent checks at runtime.
- We compile regex strings once into Pattern objects (avoid recompiling every query).
- We keep the "policy model" (CompiledPolicy) separate from UI code so it can be reused
  across Streamlit/Gradio/FastAPI, tests, and batch jobs.

Policy features supported:
- allow_tables (exact allowlist, optional)
- allow_table_regex (regex allowlist, primary for large DBs)
- deny_tables (exact denylist, optional)
- deny_table_regex (regex denylist; deny ALWAYS wins)
- required_filters (multi-tenant safety; enforced in SQL WHERE clause)
- max_limit/default_limit and read_only mode
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern, List, Dict, Any

import yaml


@dataclass(frozen=True)
class CompiledPolicy:
    """
    Immutable, runtime-friendly representation of the policy.

    Fields:
      mode:
        - "read_only" means block DDL/DML (DROP/DELETE/UPDATE/etc.)
      max_limit:
        - maximum allowed LIMIT (we cap LIMIT to this)
      default_limit:
        - added if user SQL lacks LIMIT

      allow_tables:
        - optional exact allowlist (table or "db.table")
      allow_table_regex:
        - optional regex allow rules (supports patterns like "db1\\..*")

      deny_tables:
        - optional exact denylist
      deny_table_regex:
        - optional regex deny rules (deny ALWAYS overrides allow)

      required_filters:
        - optional list of required filter expressions (e.g. "org_id = :org_id")
        - enforced at AST level by checking required column(s) appear in WHERE.
    """
    mode: str
    max_limit: int
    default_limit: int

    allow_tables: List[str]
    allow_table_regex: List[Pattern]

    deny_tables: List[str]
    deny_table_regex: List[Pattern]

    required_filters: List[str]


def load_policy(path: str) -> Dict[str, Any]:
    """
    Load policy YAML from disk into a plain Python dict.

    We keep this separate from compile_policy() so:
    - you can inspect raw YAML values if needed
    - unit tests can feed in dicts directly
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compile_policy(raw: Dict[str, Any]) -> CompiledPolicy:
    """
    Compile a raw policy dict into CompiledPolicy.

    This:
    - fills defaults
    - ensures required keys have the correct types
    - compiles regex strings into compiled Pattern objects (case-insensitive)

    NOTE: For safety, you typically want at least one allow mechanism:
    - allow_tables OR allow_table_regex should be non-empty
    If both are empty, validation should fail closed (done in validate.py).
    """
    allow_tables = raw.get("allow_tables") or []
    deny_tables = raw.get("deny_tables") or []
    required_filters = raw.get("required_filters") or []

    allow_regex_strs = raw.get("allow_table_regex") or []
    deny_regex_strs = raw.get("deny_table_regex") or []

    allow_rx = [re.compile(s, re.IGNORECASE) for s in allow_regex_strs]
    deny_rx = [re.compile(s, re.IGNORECASE) for s in deny_regex_strs]

    return CompiledPolicy(
        mode=str(raw.get("mode", "read_only")),
        max_limit=int(raw.get("max_limit", 200)),
        default_limit=int(raw.get("default_limit", 50)),
        allow_tables=list(allow_tables),
        allow_table_regex=list(allow_rx),
        deny_tables=list(deny_tables),
        deny_table_regex=list(deny_rx),
        required_filters=list(required_filters),
    )