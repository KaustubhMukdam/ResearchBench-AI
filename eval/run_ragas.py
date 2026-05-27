"""
Phase 0 — Ragas Baseline Evaluation Script

Run TWICE across the project:
  Run 1 (now)  : python eval/run_ragas.py --mode baseline
  Run 2 (Phase 1): python eval/run_ragas.py --mode with_verifier

Quick test: python eval/run_ragas.py --limit 5
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arxiv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
from ragas.llms import LangchainLLMWrapper

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


# ── arXiv helpers ──────────────────────────────────────────────────────────

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


# ── LLM answer generation ──────────────────────────────────────────────────

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


# ── Build Ragas dataset ────────────────────────────────────────────────────

def build_evaluation_dataset(
    queries: list[dict[str, Any]],
    llm: ChatGroq,
) -> EvaluationDataset:
    """
    For each query: retrieve abstracts → generate answer → pack into SingleTurnSample.

    SingleTurnSample fields:
      user_input         : the research topic
      retrieved_contexts : list of paper abstract strings
      response           : the LLM-generated answer
      reference          : ground truth claims joined into one string
    """
    samples = []

    for i, query in enumerate(queries):
        topic = query["topic"]
        reference = " ".join(query.get("ground_truth_claims", []))

        logger.info(f"[{i + 1}/{len(queries)}] '{topic}'")

        contexts = fetch_arxiv_abstracts(topic, max_results=3)
        answer = generate_answer(llm, topic, contexts)

        samples.append(
            SingleTurnSample(
                user_input=topic,
                retrieved_contexts=contexts if contexts else ["No context retrieved."],
                response=answer,
                reference=reference,
            )
        )

        time.sleep(2)  # pace for Groq free tier (~30 req/min)

    return EvaluationDataset(samples=samples)


# ── Run Ragas metrics ──────────────────────────────────────────────────────

def run_evaluation(dataset: EvaluationDataset, llm: ChatGroq) -> dict[str, float]:
    """
    Metrics:
      Faithfulness        — are all claims grounded in retrieved_contexts?
      FactualCorrectness  — are claims accurate vs reference ground truth?
      LLMContextRecall    — did retrieval fetch the right context for the reference?
    """
    ragas_llm = LangchainLLMWrapper(llm)

    metrics = [
        Faithfulness(llm=ragas_llm),
        FactualCorrectness(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]

    logger.info("Running Ragas evaluation (makes LLM calls per sample, ~10-20 min for 30 queries)...")
    results = evaluate(dataset=dataset, metrics=metrics)

    return {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "factual_correctness": round(float(results["factual_correctness"]), 4),
        "llm_context_recall": round(float(results["llm_context_recall"]), 4),
    }


# ── Save + print results ───────────────────────────────────────────────────

def save_results(scores: dict[str, float], mode: str) -> None:
    """Append one timestamped result entry to eval/results_log.jsonl."""
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
    print(f"  Factual Correctness : {scores['factual_correctness']:.4f}")
    print(f"  LLM Context Recall  : {scores['llm_context_recall']:.4f}")
    print("=" * width)
    print(f"\n  Faithfulness {scores['faithfulness']:.2f} → "
          f"{scores['faithfulness'] * 100:.1f}% of claims grounded in context")
    print(f"  Factual Correctness {scores['factual_correctness']:.2f} → "
          f"{scores['factual_correctness'] * 100:.1f}% of claims match ground truth")
    print(f"  Context Recall {scores['llm_context_recall']:.2f} → "
          f"{scores['llm_context_recall'] * 100:.1f}% of necessary context retrieved")
    print(f"\n  Results appended to: {EVAL_RESULTS_PATH}")
    print("=" * width + "\n")


# ── Main ───────────────────────────────────────────────────────────────────

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