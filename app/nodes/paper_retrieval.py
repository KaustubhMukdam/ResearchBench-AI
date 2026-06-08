"""
Paper Retrieval Node.
Combines arXiv + Semantic Scholar, deduplicates, ranks by recency + citations,
and stores all retrieved papers in ChromaDB.
"""

from __future__ import annotations
import logging

from app.schemas import PaperMetadata, ResearchState
from app.tools.arxiv_tool import search_arxiv
from app.tools.semantic_scholar_tool import enrich_with_citations
from app.memory.cache import store_papers
from app.utils.config import MAX_PAPERS_PER_QUERY

logger = logging.getLogger(__name__)


def _deduplicate(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """Remove duplicate papers by arXiv ID, keeping the first occurrence."""
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p.arxiv_id or p.title.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _rank(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    Rank papers by a composite score:
      score = 0.6 * normalised_citations + 0.4 * normalised_recency
    More recent and more cited papers rank higher.
    """
    if not papers:
        return []

    max_cites = max((p.citation_count for p in papers), default=1) or 1
    min_year = min((p.year for p in papers if p.year), default=2000)
    max_year = max((p.year for p in papers if p.year), default=2024)
    year_range = max(max_year - min_year, 1)

    def score(p: PaperMetadata) -> float:
        cite_score = p.citation_count / max_cites
        year_score = ((p.year or min_year) - min_year) / year_range
        return 0.6 * cite_score + 0.4 * year_score

    return sorted(papers, key=score, reverse=True)


def run_paper_retrieval(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Retrieval
    Input:  state.topic
    Output: state.papers (deduplicated, ranked, cached in ChromaDB)
    """
    topic = state.topic
    logger.info(f"[Retrieval] topic='{topic}'")

    # 1. arXiv keyword search
    arxiv_papers = search_arxiv(topic, max_results=MAX_PAPERS_PER_QUERY)

    # 2. Enrich with citation counts from Semantic Scholar
    enriched = enrich_with_citations(arxiv_papers)

    # 3. Deduplicate + rank
    unique = _deduplicate(enriched)
    ranked = _rank(unique)
    top_papers = ranked[:MAX_PAPERS_PER_QUERY]

    # 4. Cache in ChromaDB for downstream RAG
    store_papers(top_papers)

    logger.info(f"[Retrieval] returning {len(top_papers)} papers")
    state.papers = top_papers
    return state