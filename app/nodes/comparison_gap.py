"""
Comparison & Gap Analysis Node.
Builds a benchmark comparison table and generates a research gap summary.
Final node in the research pipeline.
"""

from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.schemas import ResearchState
from app.utils.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

_GAP_PROMPT = """\
You are a research synthesis expert. Based on the extracted methods and verified findings below, identify:
1. Key patterns across papers (what approaches dominate?)
2. Clear research gaps (what has NOT been tried or is underexplored?)
3. Open questions for future work

Topic: {topic}

Extracted methods and findings:
{findings_text}

Write a concise research gap analysis (4-6 sentences). Be specific — mention actual method names and dataset names from the data above.
"""


def _build_benchmark_table(state: ResearchState) -> list[dict]:
    """
    Build a flat benchmark table from extractions.
    Each row = one method from one paper.
    """
    rows = []
    for extraction in state.extractions:
        for method in extraction.methods:
            rows.append({
                "paper": extraction.paper_title[:60],
                "arxiv_id": extraction.arxiv_id,
                "method": method.name,
                "dataset": method.dataset,
                "metric": method.metric,
                "score": method.score,
            })
    return rows


def _build_findings_text(state: ResearchState) -> str:
    """Flatten all extractions into a readable text block for the LLM prompt."""
    lines = []
    for ext in state.extractions:
        lines.append(f"\nPaper: {ext.paper_title}")
        for m in ext.methods:
            lines.append(f"  Method: {m.name} — {m.description}")
            if m.dataset:
                lines.append(f"    Dataset: {m.dataset}, Metric: {m.metric}, Score: {m.score}")
        for f in ext.key_findings:
            lines.append(f"  Finding: {f}")
    return "\n".join(lines) if lines else "No extractions available."


def run_comparison_gap(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Gap Analysis
    Input:  state.extractions, state.topic
    Output: state.benchmark_table, state.gap_summary
    """
    logger.info("[Gap] building benchmark table")
    state.benchmark_table = _build_benchmark_table(state)

    findings_text = _build_findings_text(state)

    logger.info("[Gap] generating research gap summary")
    try:
        prompt = _GAP_PROMPT.format(topic=state.topic, findings_text=findings_text)
        response = _llm.invoke([HumanMessage(content=prompt)])
        state.gap_summary = response.content.strip()
    except Exception as e:
        logger.error(f"Gap analysis LLM call failed: {e}")
        state.gap_summary = "Gap analysis unavailable."

    logger.info(f"[Gap] done — {len(state.benchmark_table)} rows in benchmark table")
    return state