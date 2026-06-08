"""
Semantic Scholar API tool.
Enriches papers with citation counts, fetches references and citing papers.
Rate limit: 1 RPS (with API key in x-api-key header).
"""

from __future__ import annotations
import logging
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas import PaperMetadata
from app.utils.config import SEMANTIC_SCHOLAR_API_KEY, SEMANTIC_SCHOLAR_RPS

logger = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org/graph/v1"
_HEADERS = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
_MIN_INTERVAL = 1.0 / SEMANTIC_SCHOLAR_RPS  # seconds between requests


def _wait() -> None:
    """Enforce rate limit between API calls."""
    time.sleep(_MIN_INTERVAL)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=False,
)
def _get(url: str, params: dict) -> Optional[dict]:
    """Make a single GET request with retry and rate limiting."""
    _wait()
    resp = requests.get(url, headers=_HEADERS, params=params, timeout=10)
    if resp.status_code == 429:
        logger.warning("Semantic Scholar 429 — backing off")
        time.sleep(15)
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def enrich_with_citations(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    Look up each paper on Semantic Scholar by arXiv ID.
    Enriches: citation_count, semantic_scholar_id.
    Papers not found on S2 are returned unchanged.
    """
    enriched = []
    for paper in papers:
        if not paper.arxiv_id:
            enriched.append(paper)
            continue
        try:
            data = _get(
                f"{_BASE}/paper/arXiv:{paper.arxiv_id}",
                params={"fields": "paperId,citationCount,externalIds"},
            )
            if data:
                paper.citation_count = data.get("citationCount", 0) or 0
                paper.semantic_scholar_id = data.get("paperId", "")
            enriched.append(paper)
        except Exception as e:
            logger.warning(f"S2 enrichment failed for {paper.arxiv_id}: {e}")
            enriched.append(paper)
    return enriched


def fetch_references(semantic_scholar_id: str, limit: int = 10) -> list[PaperMetadata]:
    """Fetch papers that this paper cites."""
    if not semantic_scholar_id:
        return []
    try:
        data = _get(
            f"{_BASE}/paper/{semantic_scholar_id}/references",
            params={"fields": "title,authors,year,abstract,citationCount,externalIds", "limit": limit},
        )
        if not data:
            return []
        return [
            _ref_to_metadata(item.get("citedPaper", {}))
            for item in data.get("data", [])
            if item.get("citedPaper")
        ]
    except Exception as e:
        logger.warning(f"S2 references failed for {semantic_scholar_id}: {e}")
        return []


def fetch_citations(semantic_scholar_id: str, limit: int = 10) -> list[PaperMetadata]:
    """Fetch papers that cite this paper."""
    if not semantic_scholar_id:
        return []
    try:
        data = _get(
            f"{_BASE}/paper/{semantic_scholar_id}/citations",
            params={"fields": "title,authors,year,abstract,citationCount,externalIds", "limit": limit},
        )
        if not data:
            return []
        return [
            _ref_to_metadata(item.get("citingPaper", {}))
            for item in data.get("data", [])
            if item.get("citingPaper")
        ]
    except Exception as e:
        logger.warning(f"S2 citations failed for {semantic_scholar_id}: {e}")
        return []


def _ref_to_metadata(data: dict) -> PaperMetadata:
    """Convert a Semantic Scholar paper dict to PaperMetadata."""
    external = data.get("externalIds") or {}
    arxiv_id = external.get("ArXiv", "")
    authors = [a.get("name", "") for a in data.get("authors", [])[:5]]
    return PaperMetadata(
        arxiv_id=arxiv_id,
        semantic_scholar_id=data.get("paperId", ""),
        title=data.get("title", "Unknown"),
        authors=authors,
        year=data.get("year"),
        abstract=data.get("abstract") or "",
        citation_count=data.get("citationCount", 0) or 0,
        url=f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}",
        source="semantic_scholar",
    )