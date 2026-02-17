"""
core/config.py

Central configuration loader for the project.

What it does:
- Loads environment variables from .env (via python-dotenv)
- Builds DB_URL from DB_* fields if DB_URL is not provided
- Provides a Settings object to be reused by Streamlit/Gradio/tests/CLI

Why:
- Avoid duplicating env parsing logic across apps
- Keep credentials out of code
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings used across the project."""
    openai_api_key: str
    llm_model: str
    policy_path: str
    db_url: str
    org_id: str | None = None  # optional tenant id


def _build_db_url_from_parts() -> str:
    """
    Build a SQLAlchemy DB URL from DB_* environment variables.

    Supported:
      - sqlite (DB_NAME is file path)
      - postgresql+psycopg2 (host/port/user/password/name required)
      - mysql+pymysql (host/port/user/password/name required)

    DB_URL (if provided) should override this.
    """
    dialect = (os.getenv("DB_DIALECT") or "sqlite").strip()

    if dialect == "sqlite":
        # For sqlite, DB_NAME is the path to the .db file
        db_name = (os.getenv("DB_NAME") or "examples/chinook.db").strip()
        # Ensure format sqlite:///path
        if db_name.startswith("sqlite:"):
            return db_name
        # If user passed an absolute path, sqlite:////abs/path works; here we keep simple.
        return f"sqlite:///{db_name}"

    host = (os.getenv("DB_HOST") or "").strip()
    port = (os.getenv("DB_PORT") or "").strip()
    name = (os.getenv("DB_NAME") or "").strip()
    user = (os.getenv("DB_USER") or "").strip()
    password = (os.getenv("DB_PASSWORD") or "").strip()

    missing = [k for k, v in [
        ("DB_HOST", host),
        ("DB_PORT", port),
        ("DB_NAME", name),
        ("DB_USER", user),
        ("DB_PASSWORD", password),
    ] if not v]

    if missing:
        raise ValueError(
            f"Missing DB settings for dialect '{dialect}': {missing}. "
            f"Either set DB_URL directly or fill DB_* variables."
        )

    # URL-encode username/password to handle special characters safely
    user_enc = quote_plus(user)
    pwd_enc = quote_plus(password)

    return f"{dialect}://{user_enc}:{pwd_enc}@{host}:{port}/{name}"


def get_settings() -> Settings:
    """
    Load .env + environment variables and return a Settings object.

    Precedence:
      1) Environment variables from OS
      2) Values from .env loaded by python-dotenv
    """
    load_dotenv()  # loads .env into environment (does nothing if .env missing)

    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required (set it in .env or environment).")

    llm_model = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    policy_path = (os.getenv("POLICY_PATH") or "policies/default_policy.yaml").strip()

    # If DB_URL explicitly provided, use it; else build from DB_* parts.
    db_url = (os.getenv("DB_URL") or "").strip()
    if not db_url:
        db_url = _build_db_url_from_parts()

    org_id = (os.getenv("ORG_ID") or "").strip() or None

    return Settings(
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        policy_path=policy_path,
        db_url=db_url,
        org_id=org_id,
    )