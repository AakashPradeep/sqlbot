"""
core/validate.py

SQL validation & guardrails for Safe Text-to-SQL.

Adds robust allow/deny matching:
- allow_tables (exact) + allow_table_regex (regex)
- deny_tables (exact) + deny_table_regex (regex)
- deny always overrides allow

Identifier normalization:
- We attempt to build a normalized name in the following forms:
  - "table"
  - "db.table"          (db can be schema/database depending on dialect)
  - "catalog.db.table"  (3-part name if available)
We test allow/deny rules against ALL candidate forms so policies are flexible.

Other guardrails:
- read-only keyword blocking
- SELECT-only enforcement
- required_filters enforcement
- LIMIT enforcement/capping
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional, Set

import sqlglot
from sqlglot import exp


FORBIDDEN = re.compile(r"\b(drop|delete|update|insert|alter|create|truncate|merge)\b", re.I)


class ValidationError(Exception):
    """Raised when SQL violates policy, safety rules, or cannot be parsed."""


def parse_sql(sql: str) -> exp.Expression:
    """Parse SQL into a sqlglot AST."""
    try:
        return sqlglot.parse_one(sql)
    except Exception as e:
        raise ValidationError(f"SQL parse failed: {e}") from e


def enforce_read_only(sql: str) -> None:
    """Block common DDL/DML keywords (first-line read-only safety check)."""
    if FORBIDDEN.search(sql):
        raise ValidationError("Forbidden statement detected (read-only mode).")


def enforce_select_only(tree: exp.Expression) -> None:
    """Ensure the query contains a SELECT (accept WITH ... SELECT as well)."""
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        raise ValidationError("Only SELECT queries are allowed.")


# ----------------------------
# Table policy enforcement
# ----------------------------

def _table_identifiers(tree: exp.Expression) -> List[exp.Table]:
    """Return all Table nodes in the AST."""
    return list(tree.find_all(exp.Table))


def _candidate_table_names(t: exp.Table) -> List[str]:
    """
    Return candidate normalized identifiers for matching policy rules.

    We return multiple forms so policies can match whatever the user writes:
      - table
      - db.table
      - catalog.db.table   (if catalog exists)

    Notes:
    - In sqlglot, for many dialects:
        t.name   -> table
        t.db     -> schema/database (varies)
        t.catalog -> catalog (varies)
    """
    tbl = (t.name or "").strip()
    db = (getattr(t, "db", None) or "").strip()
    catalog = (getattr(t, "catalog", None) or "").strip()

    out: List[str] = []
    if tbl:
        out.append(tbl)
    if db and tbl:
        out.append(f"{db}.{tbl}")
    if catalog and db and tbl:
        out.append(f"{catalog}.{db}.{tbl}")

    # De-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        lx = x.lower()
        if lx not in seen:
            uniq.append(x)
            seen.add(lx)
    return uniq


def _matches_exact(name_candidates: List[str], exact_list: List[str]) -> bool:
    """
    Exact match against allow_tables / deny_tables.

    We compare case-insensitively.
    """
    if not exact_list:
        return False
    exact_set = {x.lower() for x in exact_list}
    return any(n.lower() in exact_set for n in name_candidates)


def _matches_regex(name_candidates: List[str], patterns) -> bool:
    """
    Regex match against allow_table_regex / deny_table_regex.

    Patterns are compiled regex objects (from policies.compile_policy).
    """
    if not patterns:
        return False
    for n in name_candidates:
        for rx in patterns:
            if rx.search(n):
                return True
    return False


def enforce_allowed_tables(tree: exp.Expression, policy) -> None:
    """
    Enforce allow/deny rules for table usage.

    Rules:
      1) DENY wins:
         - deny_tables exact or deny_table_regex match => blocked
      2) ALLOW must exist (fail-closed):
         - If both allow_tables and allow_table_regex are empty => block
      3) Otherwise each table must match ALLOW:
         - allow_tables exact OR allow_table_regex match

    Raises:
      ValidationError listing blocked/disallowed tables.
    """
    tables = _table_identifiers(tree)
    if not tables:
        return

    has_allow = bool(getattr(policy, "allow_tables", [])) or bool(getattr(policy, "allow_table_regex", []))
    if not has_allow:
        raise ValidationError("No allow rules configured (allow_tables / allow_table_regex). Blocking for safety.")

    deny_hits: List[str] = []
    allow_misses: List[str] = []

    for t in tables:
        candidates = _candidate_table_names(t)
        if not candidates:
            continue

        # Prefer the most specific representation for display
        display = candidates[-1]  # usually catalog.db.table if present else db.table else table

        # 1) Deny checks (exact + regex). Deny always overrides allow.
        if _matches_exact(candidates, getattr(policy, "deny_tables", [])) or _matches_regex(
            candidates, getattr(policy, "deny_table_regex", [])
        ):
            deny_hits.append(display)
            continue

        # 2) Allow checks (exact + regex)
        allowed = _matches_exact(candidates, getattr(policy, "allow_tables", [])) or _matches_regex(
            candidates, getattr(policy, "allow_table_regex", [])
        )
        if not allowed:
            allow_misses.append(display)

    if deny_hits:
        raise ValidationError(f"Access denied by deny rules for tables: {sorted(set(deny_hits))}")

    if allow_misses:
        raise ValidationError(f"Disallowed tables (not in allow rules): {sorted(set(allow_misses))}")


# ----------------------------
# Required filters (tenant)
# ----------------------------

def _extract_column_names(expr: exp.Expression) -> Set[str]:
    """Extract unqualified column names (lowercased) used anywhere inside an expression tree."""
    cols: Set[str] = set()
    for c in expr.find_all(exp.Column):
        if c.name:
            cols.add(c.name.lower())
    return cols


def enforce_required_filters(tree: exp.Expression, policy) -> None:
    """
    Ensure required filter columns appear in WHERE clause.

    required_filters example:
      - "org_id = :org_id"
      - "account_id = :account_id"

    MVP enforcement:
      - Extract LHS column and require it appears in WHERE.
    """
    required = getattr(policy, "required_filters", []) or []
    if not required:
        return

    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        raise ValidationError("No SELECT found for required filter enforcement.")

    where = sel.args.get("where")
    if where is None:
        raise ValidationError(f"Missing WHERE clause; required_filters={required}")

    where_cols = _extract_column_names(where)

    missing: List[str] = []
    for rf in required:
        lhs = rf.split("=")[0].strip()
        col = lhs.split(".")[-1].strip().lower()
        if col not in where_cols:
            missing.append(col)

    if missing:
        raise ValidationError(f"Missing required filter(s) in WHERE: {missing}")


# ----------------------------
# LIMIT enforcement
# ----------------------------

def ensure_limit(sql: str, default_limit: int, max_limit: int) -> str:
    """
    Ensure the SQL has a LIMIT and cap it to max_limit.

    - If LIMIT missing => add default_limit
    - If LIMIT is a numeric literal and > max_limit => cap it
    """
    tree = parse_sql(sql)
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        raise ValidationError("No SELECT found.")

    limit = sel.args.get("limit")
    if limit is None:
        sel.set("limit", exp.Limit(this=exp.Literal.number(default_limit)))
    else:
        try:
            n = int(limit.this.name)
            if n > max_limit:
                sel.set("limit", exp.Limit(this=exp.Literal.number(max_limit)))
        except Exception:
            # e.g. LIMIT :n or LIMIT (SELECT ...)
            pass

    return sel.sql()