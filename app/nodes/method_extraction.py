"""
Method Extraction Node.
Uses Groq + Pydantic structured output to extract methods, findings,
benchmarks and limitations from each retrieved paper.
"""

from __future__ import annotations
import json
import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.schemas import ExtractedMethod, PaperExtraction, ResearchState
from app.utils.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

_EXTRACTION_PROMPT = """\
You are a research paper analyst. Extract structured information from the paper abstract below.

Paper Title: {title}
Abstract: {abstract}

Return a JSON object with EXACTLY this structure (no extra keys):
{{
  "methods": [
    {{
      "name": "method name",
      "description": "one sentence description",
      "dataset": "dataset used or empty string",
      "metric": "evaluation metric or empty string",
      "score": "numeric score or empty string"
    }}
  ],
  "key_findings": ["finding 1", "finding 2"],
  "benchmarks": ["benchmark name or result 1"],
  "limitations": ["limitation 1"]
}}

Rules:
- If information is not mentioned in the abstract, use an empty list or empty string.
- Do NOT invent information not present in the abstract.
- Return valid JSON only. No markdown fences, no explanation.
"""


def _extract_single(paper_title: str, abstract: str, arxiv_id: str) -> PaperExtraction:
    """Run extraction on one paper abstract."""
    if not abstract.strip():
        return PaperExtraction(paper_title=paper_title, arxiv_id=arxiv_id)

    prompt = _EXTRACTION_PROMPT.format(title=paper_title, abstract=abstract)
    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        methods = [ExtractedMethod(**m) for m in data.get("methods", [])]
        return PaperExtraction(
            paper_title=paper_title,
            arxiv_id=arxiv_id,
            methods=methods,
            key_findings=data.get("key_findings", []),
            benchmarks=data.get("benchmarks", []),
            limitations=data.get("limitations", []),
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Extraction failed for '{paper_title}': {e}")
        return PaperExtraction(paper_title=paper_title, arxiv_id=arxiv_id)


def run_method_extraction(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Extraction
    Input:  state.papers
    Output: state.extractions
    """
    logger.info(f"[Extraction] processing {len(state.papers)} papers")
    extractions = []

    for paper in state.papers:
        extraction = _extract_single(
            paper_title=paper.title,
            abstract=paper.abstract,
            arxiv_id=paper.arxiv_id,
        )
        extractions.append(extraction)
        logger.debug(f"  Extracted {len(extraction.methods)} methods from '{paper.title[:50]}'")

    state.extractions = extractions
    logger.info(f"[Extraction] done — {len(extractions)} extractions")
    return state