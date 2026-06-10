"""Research service — wraps the LangGraph research pipeline for FastAPI."""
import logging

from app.research_graph import run_research_pipeline
from app.schemas import ResearchResponse

logger = logging.getLogger(__name__)


def run_research(topic: str) -> ResearchResponse:
    state = run_research_pipeline(topic)
    return ResearchResponse(
        topic=state.topic,
        gap_summary=state.gap_summary,
        papers=state.papers,
        extractions=state.extractions,
        verified_claims=state.verified_claims,
        benchmark_table=state.benchmark_table,
        error=state.error,
    )
