"""
core/generate.py

LLM interaction layer for generating SQL from a prompt.

Why this module exists:
- Encapsulates the LLM client initialization and call pattern.
- Keeps the rest of the codebase independent of a specific LLM provider.
- Makes it easy to replace OpenAI with Gemini/Vertex/etc. later.

Current implementation:
- Uses LangChain ChatOpenAI as the LLM client.
- Returns SQL as a plain string.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI


def make_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create an LLM client.

    Args:
      model:
        Model name (OpenAI). Ex: "gpt-4o-mini"
      temperature:
        Sampling temperature. 0 => more deterministic SQL output.

    Returns:
      A LangChain ChatOpenAI instance.

    Notes:
      - Requires OPENAI_API_KEY in environment for OpenAI.
      - For production, you might also configure:
          - request timeout
          - max retries
          - tracing / logging
    """
    return ChatOpenAI(model=model, temperature=temperature)


def generate_sql(llm: ChatOpenAI, prompt: str) -> str:
    """
    Call the LLM and return the SQL text.

    Args:
      llm:
        LLM client created by make_llm()
      prompt:
        Full prompt string (rules + schema + question)

    Returns:
      SQL query string.

    LangChain return type:
      llm.invoke(...) returns an AIMessage-like object with `.content`.
    """
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()