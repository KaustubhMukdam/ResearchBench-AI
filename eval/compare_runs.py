"""
Compare all Ragas evaluation runs in eval/results_log.jsonl.
Auto-generates your resume headline metric.

Usage:
    python eval/compare_runs.py
"""

import json
from pathlib import Path

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
        f"{'Faithful':<12} {'FactCorr':<12} {'CtxRecall':<12}"
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
            f"{s.get('factual_correctness', 0.0):<12.4f} "
            f"{s.get('llm_context_recall', 0.0):<12.4f}"
        )

    print("=" * len(header))

    if len(runs) >= 2:
        first = runs[0]["scores"]
        last = runs[-1]["scores"]

        print("\nDelta (last run vs first run):")
        for metric in ["faithfulness", "factual_correctness", "llm_context_recall"]:
            delta = last.get(metric, 0.0) - first.get(metric, 0.0)
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
            print(f"  {metric:<26}: {arrow} {delta:+.4f}")

        faith_delta = last.get("faithfulness", 0.0) - first.get("faithfulness", 0.0)
        if faith_delta > 0:
            print(
                f"\n📌 Resume metric:\n"
                f'   "Cross-agent citation verification improved Faithfulness score from '
                f"{first['faithfulness']:.2f} → {last['faithfulness']:.2f} "
                f"on 30 arXiv test queries "
                f'(+{faith_delta * 100:.1f} points), measured via Ragas."'
            )

    print()


if __name__ == "__main__":
    main()