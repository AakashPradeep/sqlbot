"""
app/gradio_ui.py

Gradio chatbot UI for Safe Text-to-SQL.

What this file does:
- Loads policy and compiles it (regex compiled once)
- Introspects DB schema and shows it (optional)
- Provides a chat UI:
    user question -> LLM generates SQL -> validate guardrails -> run query
- If the first SQL fails, does one self-heal retry.
- Adds intelligent plotting:
    - Shows results in a table by default
    - If the user asks for a chart (or selects a chart mode), renders a SINGLE plot area
      that can display line / bar / scatter / area / box.

Design notes:
- We keep the UI thin; most logic lives in core/*.
- We use gr.Plot (matplotlib) for maximum compatibility and to avoid multiple plot boxes.
- Plot column selection:
    - Auto-picks reasonable x/y columns (datetime + numeric => line; category + numeric => bar; numeric + numeric => scatter)
    - Users can override X/Y via dropdowns without changing SQL.
"""

import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

from core.schema import load_schema, schema_to_text
from core.policies import load_policy, compile_policy
from core.prompt import build_sql_prompt
from core.generate import make_llm, generate_sql
from core.validate import (
    enforce_read_only,
    enforce_select_only,
    enforce_allowed_tables,
    enforce_required_filters,
    parse_sql,
    ensure_limit,
)
from core.execute import run_query
from core.retry_fix import fix_sql
from core.config import get_settings


# ----------------------------
# Plot intelligence helpers
# ----------------------------

def wants_plot(user_msg: str, plot_mode_value: str) -> bool:
    """
    Decide whether to produce a chart.

    Rules:
    - If user selects explicit plot mode (line/bar/scatter/area/box) => plot
    - If table_only => don't plot
    - If auto => plot only if the user's message asks for a chart
    """
    if plot_mode_value and plot_mode_value != "auto":
        return plot_mode_value != "table_only"

    msg = (user_msg or "").lower()
    keywords = [
        "plot", "graph", "chart", "visualize",
        "line graph", "line chart", "bar chart", "bar graph",
        "scatter", "trend", "over time", "time series",
        "area", "box plot", "boxplot",
    ]
    return any(k in msg for k in keywords)


def _is_datetime_like(s: pd.Series) -> bool:
    """Heuristic: treat a column as datetime-like if most sampled values parse to datetime."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    try:
        sample = s.dropna().astype(str).head(20)
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.7
    except Exception:
        return False


def _to_datetime_series(s: pd.Series) -> pd.Series:
    """Convert to datetime when possible."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def _numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _datetime_cols(df: pd.DataFrame):
    return [c for c in df.columns if _is_datetime_like(df[c])]


def _categorical_cols(df: pd.DataFrame):
    out = []
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            out.append(c)
    return out


def infer_requested_chart(user_msg: str) -> str | None:
    """
    Infer chart type from user message.
    Returns: "line" | "bar" | "scatter" | "area" | "box" | None
    """
    msg = (user_msg or "").lower()
    if "box" in msg:
        return "box"
    if "area" in msg:
        return "area"
    if "bar" in msg:
        return "bar"
    if "scatter" in msg:
        return "scatter"
    if "line" in msg or "trend" in msg or "over time" in msg or "time series" in msg:
        return "line"
    return None


def pick_xy(
    df: pd.DataFrame,
    user_msg: str,
    chart_mode: str,
    x_override: str | None,
    y_override: str | None,
) -> tuple[str, str, str]:
    """
    Decide chart type and x/y columns.

    Returns (chart_type, x_col, y_col).

    chart_mode:
      - "auto" means infer from user message + data types
      - otherwise user forced: "line"|"bar"|"scatter"|"area"|"box"
    """
    cols = list(df.columns)
    dt = _datetime_cols(df)
    nums = _numeric_cols(df)
    cats = _categorical_cols(df)

    requested = infer_requested_chart(user_msg)
    chart = chart_mode if chart_mode != "auto" else (requested or None)

    x = x_override if x_override in cols else None
    y = y_override if y_override in cols else None

    # For box plot, we only need Y; X is ignored.
    if chart == "box":
        if y is None:
            y = nums[0] if nums else (cols[1] if len(cols) > 1 else cols[0])
        x = x or (cols[0] if cols else "x")
        return "box", x, y

    # Auto-pick based on chart type
    if chart in (None, "line", "area"):
        if (x is None or y is None) and dt and nums:
            x = x or dt[0]
            y = y or nums[0]
            chart = chart or "line"
        elif (x is None or y is None) and len(cols) >= 2 and pd.api.types.is_numeric_dtype(df[cols[1]]):
            x = x or cols[0]
            y = y or cols[1]
            chart = chart or "line"

    if chart in (None, "bar"):
        if (x is None or y is None) and cats and nums:
            x = x or cats[0]
            y = y or nums[0]
            chart = chart or "bar"

    if chart in (None, "scatter"):
        if (x is None or y is None) and len(nums) >= 2:
            x = x or nums[0]
            y = y or nums[1]
            chart = chart or "scatter"

    # Final fallback if still unknown
    if chart is None or x is None or y is None:
        if dt and nums:
            chart = "line"
            x = dt[0]
            y = nums[0]
        elif cats and nums:
            chart = "bar"
            x = cats[0]
            y = nums[0]
        elif len(nums) >= 2:
            chart = "scatter"
            x = nums[0]
            y = nums[1]
        else:
            # worst case: first 2 columns
            chart = "line"
            x = cols[0]
            y = cols[1] if len(cols) > 1 else cols[0]

    # Normalize area: if user chose "area" keep it, otherwise keep inferred chart
    if chart_mode == "area":
        chart = "area"

    return chart, x, y


def render_plot(
    df: pd.DataFrame,
    user_msg: str,
    chart_mode: str,
    x_override: str | None = None,
    y_override: str | None = None,
):
    """
    Render a SINGLE matplotlib figure for gr.Plot.

    - chart_mode: "auto"|"line"|"bar"|"scatter"|"area"|"box"
    - x_override/y_override: optional column names from dropdown
    """
    fig, ax = plt.subplots()

    if df is None or df.empty or df.shape[1] < 1:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        fig.tight_layout()
        return fig

    chart, x_col, y_col = pick_xy(df, user_msg, chart_mode, x_override, y_override)

    # Extract data
    if y_col not in df.columns:
        ax.text(0.5, 0.5, f"Column not found: {y_col}", ha="center", va="center")
        fig.tight_layout()
        return fig

    # Box plot uses only y
    if chart == "box":
        y = df[y_col].dropna()
        if y.empty:
            ax.text(0.5, 0.5, "No numeric data for box plot", ha="center", va="center")
        else:
            ax.boxplot(y.values, vert=True)
            ax.set_xticks([1])
            ax.set_xticklabels([y_col])
        ax.set_title("Box plot")
        fig.tight_layout()
        return fig

    if x_col not in df.columns:
        ax.text(0.5, 0.5, f"Column not found: {x_col}", ha="center", va="center")
        fig.tight_layout()
        return fig

    x = df[x_col]
    y = df[y_col]

    # Convert datetime-like x to datetime + sort for line/area
    if _is_datetime_like(x):
        x = _to_datetime_series(x)
        if chart in ("line", "area"):
            tmp = pd.DataFrame({"x": x, "y": y}).dropna()
            tmp = tmp.sort_values("x")
            x, y = tmp["x"], tmp["y"]

    # Render
    if chart == "bar":
        # If x has many unique values, bar can be crowded; keep as-is for now.
        ax.bar(x.astype(str), y)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Bar chart")
    elif chart == "scatter":
        ax.scatter(x, y)
        ax.set_title("Scatter plot")
    elif chart == "area":
        # For non-datetime x, fill_between needs numeric x indices
        if _is_datetime_like(x):
            ax.fill_between(x, y, alpha=0.3)
            ax.plot(x, y)
        else:
            idx = range(len(y))
            ax.fill_between(idx, y, alpha=0.3)
            ax.plot(idx, y)
            ax.set_xticks(list(idx))
            ax.set_xticklabels([str(v) for v in x], rotation=45, ha="right")
        ax.set_title("Area chart")
    else:
        ax.plot(x, y)
        ax.set_title("Line chart")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return fig


# ----------------------------
# App
# ----------------------------

def make_app(db_url: str, policy_path: str, model: str) -> gr.Blocks:
    """
    Build and return the Gradio app.
    """
    raw_policy = load_policy(policy_path)
    policy = compile_policy(raw_policy)

    engine = create_engine(db_url)

    # NOTE: schema filtering applies only to exact allow_tables (not regex).
    schema = load_schema(engine, policy)
    schema_text = schema_to_text(schema)

    llm = make_llm(model=model)

    def validate_and_run(sql_text: str):
        """
        Validate SQL and run it, returning (final_sql, df).
        """
        if policy.mode == "read_only":
            enforce_read_only(sql_text)

        tree = parse_sql(sql_text)
        enforce_select_only(tree)
        enforce_allowed_tables(tree, policy)
        enforce_required_filters(tree, policy)

        final_sql = ensure_limit(sql_text, policy.default_limit, policy.max_limit)
        df = run_query(engine, final_sql)
        return final_sql, df

    def chat_fn(user_message: str, history):
        """
        Main bot handler.
        """
        question = (user_message or "").strip()
        prompt = build_sql_prompt(question, schema_text, policy.default_limit)
        sql = generate_sql(llm, prompt)

        try:
            final_sql, df = validate_and_run(sql)
            status = f"SQL (validated):\n{final_sql}"
            return status, df
        except Exception as e:
            try:
                fixed_sql = fix_sql(llm, schema_text, question, sql, str(e))
                final_sql, df = validate_and_run(fixed_sql)
                status = f"SQL (fixed + validated):\n{final_sql}"
                return status, df
            except Exception as e2:
                status = (
                    "❌ Query failed.\n\n"
                    f"First error: {e}\n\n"
                    f"Fix attempt error: {e2}\n\n"
                    f"Original SQL:\n{sql}"
                )
                return status, pd.DataFrame()

    # ---- UI layout ----
    with gr.Blocks() as demo:
        gr.Markdown("# Safe Text-to-SQL Bot (Gradio)")
        gr.Markdown("Chat → SQL generation → guardrails → execution → results.")

        with gr.Accordion("Schema (introspected)", open=False):
            gr.Textbox(value=schema_text, label="Schema", lines=20, interactive=False)

        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(label="Ask a question", placeholder="e.g., Top 10 customers by total spend")

        # Visualization controls
        with gr.Row():
            plot_mode = gr.Dropdown(
                ["auto", "table_only", "chart"],
                value="auto",
                label="Output mode",
                info="auto = chart only when you ask; chart = always show chart; table_only = never chart",
            )

            chart_type = gr.Dropdown(
                ["auto", "line", "bar", "scatter", "area", "box"],
                value="auto",
                label="Chart type",
            )

            x_col = gr.Dropdown(choices=[], value=None, label="X column (optional)")
            y_col = gr.Dropdown(choices=[], value=None, label="Y column (optional)")

        out_sql = gr.Textbox(label="SQL / Status", lines=10)

        with gr.Row():
            out_df = gr.Dataframe(label="Results", interactive=False)
            out_plot = gr.Plot(label="Chart")

        def on_submit(user_msg, chat_hist, plot_mode_value, chart_type_value, x_override, y_override):
            chat_hist = chat_hist or []
            chat_hist.append({"role": "user", "content": user_msg})

            sql_status, df = chat_fn(user_msg, chat_hist)

            chat_hist.append({"role": "assistant", "content": sql_status})

            # Update X/Y dropdown choices based on df columns
            col_choices = list(df.columns) if hasattr(df, "columns") else []
            x_update = gr.Dropdown(choices=col_choices, value=x_override if x_override in col_choices else None)
            y_update = gr.Dropdown(choices=col_choices, value=y_override if y_override in col_choices else None)

            # Decide whether to plot
            mode_for_wants_plot = "table_only" if plot_mode_value == "table_only" else (
                "line" if plot_mode_value == "chart" else "auto"
            )

            if plot_mode_value == "chart":
                # Always show chart
                fig = render_plot(df, user_msg, chart_type_value, x_override, y_override)
            else:
                # auto/table_only behavior
                fig = render_plot(df, user_msg, chart_type_value, x_override, y_override) if wants_plot(
                    user_msg, mode_for_wants_plot
                ) else render_plot(pd.DataFrame(), user_msg, chart_type_value)

            return "", chat_hist, sql_status, df, fig, x_update, y_update

        msg.submit(
            on_submit,
            inputs=[msg, chatbot, plot_mode, chart_type, x_col, y_col],
            outputs=[msg, chatbot, out_sql, out_df, out_plot, x_col, y_col],
        )

    return demo


if __name__ == "__main__":
    s = get_settings()
    app = make_app(db_url=s.db_url, policy_path=s.policy_path, model=s.llm_model)
    app.queue()
    app.launch()