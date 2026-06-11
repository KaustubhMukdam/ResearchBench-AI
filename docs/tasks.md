# Tasks — ResearchBench AI

## Phase 0 — Evaluation harness (FIRST, before any agent code)

- [ x ] Set up LangSmith account + get API key
- [ x ] Set LANGCHAIN_TRACING_V2=true in .env — verify it traces a simple LangChain call
- [ x ] Create eval/test_queries.json — 30 arXiv topic queries with known ground-truth papers and at least 1 checkable claim per paper
- [ x ] Install Ragas + write eval/run_ragas.py skeleton
- [ x ] Run baseline Ragas eval with no verifier (just retrieval → LLM summary) → record score in eval.md

## Phase 1 — Research Module (Weeks 1–2)

- [ x ] Register for Semantic Scholar API key (apply at semanticscholar.org/product/api)
- [ x ] Build arxiv_tool.py — query by topic, return structured metadata list
- [ x ] Build semantic_scholar_tool.py — fetch paper details, citations, references
- [ x ] Build paper_retrieval.py node — combines both tools, deduplicates, ranks by recency + citations
- [ x ] Build ChromaDB cache layer (cache.py) — store/retrieve paper embeddings
- [ x ] Build method_extraction.py node — Groq call + Pydantic PaperExtraction schema
- [ x ] Build citation_verifier.py node — RAG over ChromaDB, flag low-confidence claims
- [ x ] Wire research_graph.py StateGraph (retrieval → extraction → verification → gap)
- [ x ] Build comparison_gap.py node — benchmark table + research gap summary
- [ x ] Run Ragas eval WITH verifier → compare to baseline → log in eval.md

## Phase 2 — DS Module (Week 3)

- [ x ] Build eda_agent.py — pandas profiling on uploaded CSV
- [ x ] Build feature_agent.py — LLM suggestions with structured Pydantic output
- [ x ] Build modeling_agent.py — XGBoost + LightGBM + RF with 5-fold CV
- [ x ] Build validation_agent.py — leakage + overfit checks
- [ x ] Build explainability_agent.py — SHAP values for best model
- [ x ] Wire ds_graph.py StateGraph
- [ x ] Test on 3 Kaggle datasets — record baseline vs. agent model scores in experiment_log.md

## Phase 3 — Integration + Frontend (Week 4)

- [ x ] FastAPI routers (research.py, dataset.py)
- [ x ] Streamlit app — topic input + CSV upload + results display (tables + SHAP chart)
- [ x ] Connect Streamlit → FastAPI → LangGraph end-to-end (run test commands below)
- [ x ] Verify LangSmith traces appear for a full run

## Phase 4 — Benchmarking + Polish (Weeks 5–6)

- [ x ] Run full Ragas eval suite — record final Faithfulness score delta (eval.md updated with honest narrative)
- [x] Test DS module on 5 datasets — all beat baseline (documented in experiment_log.md)
- [ x ] Write README with: what it does, how to run it, benchmark results (the specific numbers)
- [x] Add token cost display per run in Streamlit (TokenCounter callback tracks Groq token usage live)

## Done

- [x] Project ideation and architecture design
- [x] All documentation files created

## Blocked

- (none yet)

## Backlog / nice-to-have

- BibTeX export
- Citation graph visualisation
- Novelty scorer
- Benchmark trend charts over time
