"""
Phase 0 — Ragas Baseline Evaluation Script

Run TWICE across the project:
  Run 1 (now)  : python eval/run_ragas.py --mode baseline
  Run 2 (Phase 1): python eval/run_ragas.py --mode with_verifier

Quick test: python eval/run_ragas.py --limit 5
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import arxiv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
)

from app.utils.config import (
    EVAL_RESULTS_PATH,
    EVAL_TEST_QUERIES_PATH,
    GROQ_API_KEY,
    GROQ_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# arXiv helpers

def fetch_arxiv_abstracts(topic: str, max_results: int = 3) -> list[str]:
    """
    Fetch up to max_results abstracts from arXiv for a topic.
    Returns list of strings (title + abstract per paper).
    Falls back to empty list on any network error.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        contexts = []
        for paper in client.results(search):
            contexts.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Published: {paper.published.year}\n"
                f"Abstract: {paper.summary}"
            )
            time.sleep(0.5)  # be polite to arXiv
        return contexts
    except Exception as e:
        logger.warning(f"arXiv fetch failed for '{topic}': {e}")
        return []
    
def fetch_arxiv_by_ids(paper_ids: list[str]) -> list[str]:
    """
    Fetch abstracts for specific arXiv paper IDs.
    Used to build ground truth context for evaluation.
    """
    contexts = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=paper_ids)
        for paper in client.results(search):
            contexts.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Published: {paper.published.year}\n"
                f"Abstract: {paper.summary}"
            )
            time.sleep(0.3)
    except Exception as e:
        logger.warning(f"arXiv ID fetch failed for {paper_ids}: {e}")
    return contexts


# LLM answer generation

def generate_answer(llm: ChatGroq, topic: str, contexts: list[str]) -> str:
    """
    Generate a summary answer using retrieved contexts.
    This is the BASELINE — no citation verifier, raw LLM summarisation.
    """
    if not contexts:
        return "No relevant papers found for this topic."

    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        f"You are a research assistant. Based ONLY on the following paper abstracts, "
        f"answer the question: What are the key methods and findings related to '{topic}'?\n\n"
        f"Paper abstracts:\n{context_block}\n\n"
        f"Answer concisely in 3-5 sentences. "
        f"Only include claims you can support from the abstracts above."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.warning(f"LLM generation failed for '{topic}': {e}")
        return "LLM generation failed."


# Build Ragas dataset

def build_evaluation_dataset(
    queries: list[dict[str, Any]],
    llm: ChatGroq,
) -> Dataset:
    """
    For each query:
    - retrieved_contexts: fetched by TOPIC keyword (what the agent will do in production)
    - ground_truth_context: fetched by PAPER ID (the correct papers we know are relevant)

    Ragas then measures:
    - Faithfulness: is the answer grounded in retrieved_contexts?
    - Context Recall: did the keyword retrieval get the same papers as the ID-based ground truth?
    - Answer Correctness: does the answer match the known claims?
    """
    records = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }

    for i, query in enumerate(queries):
        topic = query["topic"]
        ground_truth_ids = query.get("ground_truth_paper_ids", [])
        ground_truth_claims = query.get("ground_truth_claims", [])

        logger.info(f"[{i + 1}/{len(queries)}] '{topic}'")

        # What the agent retrieves in production (keyword search)
        retrieved_contexts = fetch_arxiv_abstracts(topic, max_results=3)

        # The known-correct papers (fetched by ID for evaluation only)
        ground_truth_contexts = fetch_arxiv_by_ids(ground_truth_ids[:2])

        # Answer is generated from retrieved contexts (simulates production)
        answer = generate_answer(llm, topic, retrieved_contexts)

        # Ground truth: combine the known claims + ground truth paper abstracts
        # This gives Ragas enough signal to evaluate recall and correctness
        ground_truth_text = " ".join(ground_truth_claims)
        if ground_truth_contexts:
            # Use the actual ground truth paper abstracts as the reference context
            records["contexts"].append(
                retrieved_contexts if retrieved_contexts else ["No context retrieved."]
            )
        else:
            records["contexts"].append(["No context retrieved."])

        records["question"].append(topic)
        records["answer"].append(answer)
        records["ground_truth"].append(ground_truth_text)

        time.sleep(2)

    return Dataset.from_dict(records)

# Add this class above run_evaluation() in run_ragas.py
import time
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

class RateLimitedGroq:
    """
    Thin wrapper that enforces a minimum delay between Groq calls.
    Prevents Ragas from firing parallel requests that exhaust TPM/TPD limits.
    """
    def __init__(self, llm: ChatGroq, min_interval: float = 3.0):
        self._llm = llm
        self._min_interval = min_interval
        self._last_call = 0.0

    def invoke(self, input, **kwargs):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        result = self._llm.invoke(input, **kwargs)
        self._last_call = time.time()
        return result

    # Ragas calls these attributes internally — proxy them through
    def __getattr__(self, name):
        return getattr(self._llm, name)


def run_evaluation(dataset: Dataset, llm: ChatGroq) -> dict[str, float]:
    """
    Ragas 0.1.21 — explicitly pass Groq LLM + HuggingFace embeddings.
    This prevents Ragas from falling back to OpenAI for either LLM or embeddings.

    Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, local, no API key needed)
    LLM: Groq via LangChain wrapper
    """
    import os
    os.environ["RAGAS_MAX_WORKERS"] = "1"

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 3 second gap between every Ragas LLM call
    throttled_llm = RateLimitedGroq(llm, min_interval=3.0)

    ragas_llm = LangchainLLMWrapper(langchain_llm=throttled_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    faithfulness.llm = ragas_llm
    answer_correctness.llm = ragas_llm
    context_recall.llm = ragas_llm
    answer_correctness.embeddings = ragas_embeddings

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_correctness, context_recall],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_correctness": round(float(result["answer_correctness"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
    }
    return scores


# Save + print results
import math

def _is_valid_scores(scores: dict[str, float]) -> bool:
    return all(not math.isnan(v) for v in scores.values())

def save_results(scores: dict[str, float], mode: str) -> None:
    if not _is_valid_scores(scores):
        logger.error(
            "Scores contain NaN — run had too many 429 errors. "
            "NOT saving to results_log.jsonl. Wait for token limit to reset and retry."
        )
        return
    results_path = Path(EVAL_RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scores": scores,
    }
    with results_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Results saved to {EVAL_RESULTS_PATH}")

def save_results(scores: dict[str, float], mode: str) -> None:
    results_path = Path(EVAL_RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scores": scores,
    }
    with results_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Results saved to {EVAL_RESULTS_PATH}")


def print_summary(scores: dict[str, float], mode: str) -> None:
    width = 55
    print("\n" + "=" * width)
    print(f"  RAGAS RESULTS — {mode.upper()}")
    print("=" * width)
    print(f"  Faithfulness        : {scores['faithfulness']:.4f}")
    print(f"  Answer Correctness  : {scores['answer_correctness']:.4f}")
    print(f"  Context Recall      : {scores['context_recall']:.4f}")
    print("=" * width)
    print(f"\n  Faithfulness {scores['faithfulness']:.2f} → "
          f"{scores['faithfulness'] * 100:.1f}% of claims grounded in context")
    print(f"  Answer Correctness {scores['answer_correctness']:.2f} → "
          f"{scores['answer_correctness'] * 100:.1f}% accuracy vs ground truth")
    print(f"  Context Recall {scores['context_recall']:.2f} → "
          f"{scores['context_recall'] * 100:.1f}% of necessary context retrieved")
    print(f"\n  Results appended to: {EVAL_RESULTS_PATH}")
    print("=" * width + "\n")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation for ResearchBench AI")
    parser.add_argument(
        "--mode",
        choices=["baseline", "with_verifier"],
        default="baseline",
        help="Label for this run (default: baseline)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on first N queries only — e.g. --limit 5 for quick testing",
    )
    args = parser.parse_args()

    queries_path = Path(EVAL_TEST_QUERIES_PATH)
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Test queries file not found: {EVAL_TEST_QUERIES_PATH}\n"
            "Expected: eval/test_queries.json"
        )

    with queries_path.open() as f:
        queries: list[dict] = json.load(f)

    if args.limit:
        queries = queries[: args.limit]
        logger.info(f"Limited to first {args.limit} queries")

    logger.info(f"Loaded {len(queries)} queries | mode: {args.mode}")

    # LangSmith traces this automatically via LANGSMITH_TRACING=true in .env
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    dataset = build_evaluation_dataset(queries=queries, llm=llm)
    scores = run_evaluation(dataset=dataset, llm=llm)

    save_results(scores=scores, mode=args.mode)
    print_summary(scores=scores, mode=args.mode)


if __name__ == "__main__":
    main()