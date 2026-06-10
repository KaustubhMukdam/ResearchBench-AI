"""
ChromaDB cache layer.
Stores paper abstracts as embeddings for fast semantic retrieval.
Used by citation_verifier.py to do RAG over stored papers.
"""

from __future__ import annotations
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from app.schemas import PaperMetadata
from app.utils.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _get_collection() -> chromadb.Collection:
    """Get or create the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Try the full SentenceTransformer model first.
    # Falls back to ChromaDB's built-in default embeddings if the model
    # isn't cached locally yet (avoids HuggingFace download failures at runtime).
    try:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        logger.debug(f"Using SentenceTransformer embedding: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(
            f"SentenceTransformer load failed ({e}). "
            f"Falling back to ChromaDB default embeddings — "
            f"run 'python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer(\\'{EMBEDDING_MODEL}\\')\"' to pre-cache the model."
        )
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def store_papers(papers: list[PaperMetadata]) -> int:
    """
    Store papers in ChromaDB. Skips papers already cached (by arxiv_id).
    Returns the number of newly stored papers.
    """
    if not papers:
        return 0

    collection = _get_collection()
    stored = 0

    for paper in papers:
        doc_id = paper.arxiv_id or paper.semantic_scholar_id or paper.title[:50]
        if not doc_id:
            continue

        # Skip if already cached
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        doc_text = f"Title: {paper.title}\nAbstract: {paper.abstract}"
        metadata = {
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
            "year": str(paper.year or ""),
            "citation_count": str(paper.citation_count),
            "authors": ", ".join(paper.authors[:3]),
        }

        collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        stored += 1

    logger.info(f"ChromaDB: stored {stored} new papers (skipped {len(papers) - stored} cached)")
    return stored


def retrieve_relevant_chunks(query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search over stored papers.
    Returns list of dicts with 'text', 'title', 'arxiv_id', 'distance'.
    """
    collection = _get_collection()

    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty — no papers cached yet")
        return []

    n_results = min(n_results, count)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta.get("title", ""),
            "arxiv_id": meta.get("arxiv_id", ""),
            "distance": dist,
        })

    return chunks


def get_cache_size() -> int:
    """Return number of papers currently in the cache."""
    return _get_collection().count()