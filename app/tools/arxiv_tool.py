"""
arXiv retrieval tool.
Queries arXiv by topic and returns structured PaperMetadata list.
Also supports fetching by exact arXiv ID.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import PaperMetadata
from app.utils.config import MAX_PAPERS_PER_QUERY

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_results(search: arxiv.Search) -> list[arxiv.Result]:
    client = arxiv.Client()
    results = list(client.results(search))
    return results


def search_arxiv(topic: str, max_results: int = MAX_PAPERS_PER_QUERY) -> list[PaperMetadata]:
    """
    Search arXiv by topic keyword.
    Returns up to max_results papers ranked by relevance.
    """
    logger.info(f"arXiv search: '{topic}' (max={max_results})")
    try:
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = _fetch_results(search)
        papers = [_to_metadata(r) for r in results]
        logger.info(f"arXiv returned {len(papers)} papers for '{topic}'")
        return papers
    except Exception as e:
        logger.error(f"arXiv search failed for '{topic}': {e}")
        return []


def fetch_arxiv_by_ids(arxiv_ids: list[str]) -> list[PaperMetadata]:
    """
    Fetch specific papers by their arXiv IDs.
    Used by the evaluation harness to retrieve ground truth papers.
    """
    if not arxiv_ids:
        return []
    logger.info(f"arXiv ID fetch: {arxiv_ids}")
    try:
        search = arxiv.Search(id_list=arxiv_ids)
        results = _fetch_results(search)
        papers = [_to_metadata(r) for r in results]
        time.sleep(0.5)
        return papers
    except Exception as e:
        logger.error(f"arXiv ID fetch failed for {arxiv_ids}: {e}")
        return []


def _to_metadata(result: arxiv.Result) -> PaperMetadata:
    """Convert an arxiv.Result to PaperMetadata."""
    arxiv_id = result.entry_id.split("/abs/")[-1].split("v")[0]
    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[a.name for a in result.authors[:5]],
        year=result.published.year if result.published else None,
        abstract=result.summary,
        citation_count=0,  # arXiv doesn't provide this; enriched by Semantic Scholar
        url=result.entry_id,
        source="arxiv",
    )