from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

def run_query(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)
