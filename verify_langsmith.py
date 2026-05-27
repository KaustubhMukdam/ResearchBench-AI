"""
Phase 0 — Step 1: Verify LangSmith tracing is working.

Run this BEFORE writing any agent code:
    python verify_langsmith.py

Expected output:
    LangSmith tracing is active. Run ID: <uuid>
    Project: researchbench-ai

If you see an error, fix your .env before proceeding.
"""

import sys
from app.utils.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    GROQ_API_KEY,
    GROQ_MODEL,
)


def verify_langsmith() -> None:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from langsmith import Client
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    print("Making a test LLM call (will appear in LangSmith)...")

    response = llm.invoke(
        [HumanMessage(content="Reply with exactly: LANGSMITH_TRACING_OK")]
    )

    assert "LANGSMITH_TRACING_OK" in response.content, (
        f"Unexpected LLM response: {response.content}"
    )

    client = Client(api_key=LANGSMITH_API_KEY)
    runs = list(
        client.list_runs(
            project_name=LANGSMITH_PROJECT,
            limit=1,
        )
    )

    if not runs:
        print("LLM call succeeded but no runs found in LangSmith yet.")
        print("Wait ~10 seconds and check: https://smith.langchain.com")
        print(f"Project: {LANGSMITH_PROJECT}")
    else:
        run = runs[0]
        print(f"LangSmith tracing is active.")
        print(f"Run ID   : {run.id}")
        print(f"Run name : {run.name}")
        print(f"Project  : {LANGSMITH_PROJECT}")
        print(f"Dashboard: https://smith.langchain.com/projects/{LANGSMITH_PROJECT}")


if __name__ == "__main__":
    verify_langsmith()