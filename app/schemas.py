"""
Shared Pydantic schemas used across all Phase 1 nodes.
Single source of truth for data shapes — import from here everywhere.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Structured metadata for a single paper."""
    arxiv_id: str = ""
    semantic_scholar_id: str = ""
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    source: str = "arxiv"  # "arxiv" | "semantic_scholar"


class ExtractedMethod(BaseModel):
    """A single model/method extracted from a paper."""
    name: str
    description: str
    dataset: str = ""
    metric: str = ""
    score: str = ""


class PaperExtraction(BaseModel):
    """Structured extraction result for one paper."""
    paper_title: str
    arxiv_id: str = ""
    methods: list[ExtractedMethod] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class VerifiedClaim(BaseModel):
    """A single claim with its verification result."""
    claim: str
    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk: str = ""
    paper_title: str = ""


class ResearchState(BaseModel):
    """
    LangGraph state object — passed between all nodes in research_graph.py.
    Each node reads what it needs and writes its outputs back here.
    """
    topic: str
    papers: list[PaperMetadata] = Field(default_factory=list)
    extractions: list[PaperExtraction] = Field(default_factory=list)
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    gap_summary: str = ""
    benchmark_table: list[dict] = Field(default_factory=list)
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True