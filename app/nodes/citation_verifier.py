"""
Citation Verifier Node.
The core accuracy component — uses RAG over ChromaDB to verify
every key finding against the source paper text.
This is what improves the Faithfulness score in the Ragas eval.
"""

from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.memory.cache import retrieve_relevant_chunks
from app.schemas import ResearchState, VerifiedClaim
from app.utils.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

_VERIFY_PROMPT = """\
You are a fact-checking assistant. Your job is to determine whether a claim is supported by the provided source text.

Claim: {claim}

Source text from paper "{paper_title}":
{source_text}

Answer with exactly one word: SUPPORTED or UNSUPPORTED
Then on the next line, give a confidence score from 0.0 to 1.0.

Format:
SUPPORTED
0.92

or

UNSUPPORTED
0.85
"""


def _verify_claim(claim: str) -> VerifiedClaim:
    """
    Verify a single claim using RAG:
    1. Retrieve the most relevant chunk from ChromaDB
    2. Ask the LLM whether the chunk supports the claim
    """
    chunks = retrieve_relevant_chunks(query=claim, n_results=3)

    if not chunks:
        return VerifiedClaim(
            claim=claim,
            supported=False,
            confidence=0.0,
            source_chunk="No source text found in cache.",
            paper_title="",
        )

    # Use the most relevant chunk (lowest cosine distance)
    best = min(chunks, key=lambda c: c["distance"])
    source_text = best["text"]
    paper_title = best["title"]

    prompt = _VERIFY_PROMPT.format(
        claim=claim,
        paper_title=paper_title,
        source_text=source_text,
    )

    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().splitlines()
        verdict = lines[0].strip().upper() if lines else "UNSUPPORTED"
        confidence = float(lines[1].strip()) if len(lines) > 1 else 0.5
        confidence = max(0.0, min(1.0, confidence))
        supported = verdict == "SUPPORTED"
    except Exception as e:
        logger.warning(f"Verification LLM call failed: {e}")
        supported = False
        confidence = 0.0

    return VerifiedClaim(
        claim=claim,
        supported=supported,
        confidence=confidence,
        source_chunk=source_text[:500],
        paper_title=paper_title,
    )


def run_citation_verifier(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Verification
    Input:  state.extractions
    Output: state.verified_claims

    Verifies every key_finding from every extraction.
    Flags low-confidence claims for downstream filtering.
    """
    claims_to_verify: list[str] = []
    for extraction in state.extractions:
        claims_to_verify.extend(extraction.key_findings)

    logger.info(f"[Verifier] verifying {len(claims_to_verify)} claims")
    verified: list[VerifiedClaim] = []

    for claim in claims_to_verify:
        result = _verify_claim(claim)
        verified.append(result)
        status = "[OK]" if result.supported else "[NO]"
        logger.debug(f"  {status} [{result.confidence:.2f}] {claim[:70]}")

    supported = sum(1 for v in verified if v.supported)
    logger.info(
        f"[Verifier] {supported}/{len(verified)} claims supported "
        f"(faithfulness={supported/len(verified):.2f})" if verified else "[Verifier] no claims to verify"
    )

    state.verified_claims = verified
    return state