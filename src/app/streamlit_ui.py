import os
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

from src.core.schema import load_schema, schema_to_text
from src.core.policies import (load_policy, compile_policy)
from src.core.prompt import build_sql_prompt
from src.core.generate import make_llm, generate_sql
from src.core.validate import (
    enforce_read_only, enforce_select_only, enforce_allowed_tables,
    parse_sql, ensure_limit, ValidationError
)
from src.core.execute import run_query
from src.core.retry_fix import fix_sql
from src.core.config import get_settings

load_dotenv()

st.set_page_config(page_title="Safe Text-to-SQL", layout="wide")
st.title("Safe Text-to-SQL Bot (SELECT-only + guardrails + self-heal)")

s = get_settings()

db_url = st.sidebar.text_input("DB URL", value=s.db_url)
policy_path = st.sidebar.text_input("Policy file", value=s.policy_path)
model = st.sidebar.text_input("LLM model", value=s.llm_model)

policy = load_policy(policy_path)
compiled_policy = compile_policy(policy)  # For validation, with compiled regex
engine = create_engine(db_url)

schema = load_schema(engine, compiled_policy)
schema_text = schema_to_text(schema)

with st.expander("Schema (allowed)", expanded=False):
    st.code(schema_text)

question = st.text_input("Ask a question (natural language)", value="Top 10 customers by total spend")

if st.button("Run"):
    llm = make_llm(model=model)
    prompt = build_sql_prompt(question, schema_text, policy["default_limit"])
    sql = generate_sql(llm, prompt)

    st.subheader("Generated SQL (raw)")
    st.code(sql, language="sql")

    try:
        enforce_read_only(sql)
        tree = parse_sql(sql)
        enforce_select_only(tree)
        enforce_allowed_tables(tree, compiled_policy)
        sql2 = ensure_limit(sql, compiled_policy.default_limit, compiled_policy.max_limit)

        st.subheader("Validated SQL (final)")
        st.code(sql2, language="sql")

        df = run_query(engine, sql2)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.warning(f"First attempt failed: {e}")

        # self-heal once
        try:
            fixed = fix_sql(llm, schema_text, question, sql, str(e))
            st.subheader("Fixed SQL (retry)")
            st.code(fixed, language="sql")

            enforce_read_only(fixed)
            tree = parse_sql(fixed)
            enforce_select_only(tree)
            enforce_allowed_tables(tree, policy["allow_tables"])
            fixed2 = ensure_limit(fixed, policy["default_limit"], policy["max_limit"])

            df = run_query(engine, fixed2)
            st.subheader("Results (after fix)")
            st.dataframe(df, use_container_width=True)

        except Exception as e2:
            st.error(f"Retry failed: {e2}")
