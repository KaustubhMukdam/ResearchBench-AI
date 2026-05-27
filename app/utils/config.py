"""
Central config loader. Reads .env and exposes typed constants.
Import this at the top of every file that needs API keys or settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Raise a clear error if a required env var is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your keys."
        )
    return value


# LangSmith
LANGSMITH_API_KEY: str = _require("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "researchbench-ai")

# LLM
GROQ_API_KEY: str = _require("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Search
TAVILY_API_KEY: str = _require("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # optional until Phase 1

# ChromaDB
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./memory/.chromadb")
CHROMA_COLLECTION_NAME: str = "researchbench_papers"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Eval
EVAL_TEST_QUERIES_PATH: str = os.getenv("EVAL_TEST_QUERIES_PATH", "./eval/test_queries.json")
EVAL_RESULTS_PATH: str = os.getenv("EVAL_RESULTS_PATH", "./eval/results_log.jsonl")

# Retrieval
MAX_PAPERS_PER_QUERY: int = int(os.getenv("MAX_PAPERS_PER_QUERY", "10"))
SEMANTIC_SCHOLAR_RPS: float = float(os.getenv("SEMANTIC_SCHOLAR_RPS", "0.9"))  # stay under 1 RPS limit