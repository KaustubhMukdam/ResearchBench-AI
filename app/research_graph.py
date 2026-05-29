"""
Research LangGraph StateGraph.
Wires all Phase 1 nodes into a linear pipeline:
  retrieval → extraction → verification → gap_analysis
"""

from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from app.schemas import ResearchState
from app.nodes.paper_retrieval import run_paper_retrieval
from app.nodes.method_extraction import run_method_extraction
from app.nodes.citation_verifier import run_citation_verifier
from app.nodes.comparison_gap import run_comparison_gap

logger = logging.getLogger(__name__)


def build_research_graph() -> StateGraph:
    """
    Build and compile the research pipeline graph.
    All nodes take ResearchState and return ResearchState.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("retrieval", run_paper_retrieval)
    graph.add_node("extraction", run_method_extraction)
    graph.add_node("verification", run_citation_verifier)
    graph.add_node("gap_analysis", run_comparison_gap)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "extraction")
    graph.add_edge("extraction", "verification")
    graph.add_edge("verification", "gap_analysis")
    graph.add_edge("gap_analysis", END)

    return graph.compile()


def run_research_pipeline(topic: str) -> ResearchState:
    """
    Run the full research pipeline for a given topic.
    Returns the final ResearchState with papers, extractions,
    verified_claims, benchmark_table, and gap_summary populated.
    """
    logger.info(f"Starting research pipeline for: '{topic}'")
    graph = build_research_graph()

    initial_state = ResearchState(topic=topic)
    final_state = graph.invoke(initial_state)

    logger.info(
        f"Pipeline complete — "
        f"{len(final_state['papers'])} papers, "
        f"{len(final_state['extractions'])} extractions, "
        f"{len(final_state['verified_claims'])} claims verified"
    )
    return ResearchState(**final_state)