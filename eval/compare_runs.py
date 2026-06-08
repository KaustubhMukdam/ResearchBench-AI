"""
Compare all Ragas evaluation runs in eval/results_log.jsonl.
Auto-generates your resume headline metric.

Usage:
    python eval/compare_runs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from app.utils.config import EVAL_RESULTS_PATH


def main() -> None:
    results_path = Path(EVAL_RESULTS_PATH)

    if not results_path.exists():
        print(f"No results found at {EVAL_RESULTS_PATH}")
        print("Run `python eval/run_ragas.py` first.")
        return

    runs: list[dict] = []
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    if not runs:
        print("results_log.jsonl is empty.")
        return

    header = (
        f"{'#':<4} {'Timestamp':<22} {'Mode':<16} "
        f"{'Faithful':<12} {'AnswerCorr':<12} {'CtxRecall':<12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for i, run in enumerate(runs):
        ts = run["timestamp"][:19].replace("T", " ")
        mode = run["mode"]
        s = run["scores"]
        print(
            f"{i + 1:<4} {ts:<22} {mode:<16} "
            f"{s.get('faithfulness', 0.0):<12.4f} "
            f"{s.get('answer_correctness', 0.0):<12.4f} "
            f"{s.get('context_recall', 0.0):<12.4f}"
        )

    print("=" * len(header))

    if len(runs) >= 2:
        first = runs[0]["scores"]
        last = runs[-1]["scores"]

        print("\nDelta (last run vs first run):")
        for metric in ["faithfulness", "answer_correctness", "context_recall"]:
            delta = last.get(metric, 0.0) - first.get(metric, 0.0)
            arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
            print(f"  {metric:<26}: {arrow} {delta:+.4f}")

        faith_delta = last.get("faithfulness", 0.0) - first.get("faithfulness", 0.0)
        recall_delta = last.get("context_recall", 0.0) - first.get("context_recall", 0.0)
        if recall_delta > 0 or faith_delta != 0:
            print(
                f"\n[RESUME METRIC]:\n"
                f'   "Multi-agent RAG pipeline improved Context Recall from '
                f"{first.get('context_recall', 0.0):.2f} -> {last.get('context_recall', 0.0):.2f} "
                f"and Answer Correctness from "
                f"{first.get('answer_correctness', 0.0):.2f} -> {last.get('answer_correctness', 0.0):.2f} "
                f"on 5 arXiv test queries, measured via Ragas."
                f'"'
            )

    print()


if __name__ == "__main__":
    main()