# PRD — ResearchBench AI

## Problem statement
AIML students, researchers, and Kaggle practitioners waste significant time on two disconnected workflows:
1. Manually reading and comparing papers to find relevant methods and benchmarks
2. Manually writing EDA, feature engineering, and model evaluation code for new datasets

Existing tools solve these in isolation — ResearchRabbit/Elicit help with papers; AutoGluon/H2O handle AutoML. No tool chains both: "What does the literature say I should try? → Run it → Measure the gap." Current AI research assistants also hallucinate citations frequently — a core trust problem.

## Target users
- AIML/Data Science students working on Kaggle competitions who want to know what methods SOTA literature recommends for their problem type
- Researchers doing literature surveys who need structured benchmark comparisons across papers
- Students preparing internship portfolios who need a technically deep, measurable project

## Core features (MVP)

### Module 1 — Research Agent Pipeline
- [ ] **Paper Retrieval Agent** — searches arXiv + Semantic Scholar by topic/keyword; ranks papers by relevance and recency
- [ ] **Method Extraction Agent** — extracts architecture, dataset, metrics, training method as structured Pydantic schema from each paper
- [ ] **Citation Verifier Agent** — cross-checks every extracted claim against actual paper text using RAG; this is the hallucination reducer
- [ ] **Comparison + Gap Agent** — generates benchmark comparison table; identifies limitations and unexplored research directions

### Module 2 — Data Science Agent Pipeline
- [ ] **EDA Agent** — runs on uploaded CSV; finds missing values, correlations, class imbalance, skew
- [ ] **Feature Engineering Agent** — suggests encoding strategies, transformations, derived features
- [ ] **Modeling Agent** — tests XGBoost, LightGBM, RandomForest with cross-validation; reports metrics
- [ ] **Validation Agent** — checks for data leakage, improper splits, overfitting signals
- [ ] **Explainability Agent** — generates SHAP values and feature importance charts

### Infrastructure (non-negotiable, built first)
- [ ] **LangSmith tracing** — every agent call traced; latency, token cost, error rates visible
- [ ] **Ragas evaluation harness** — Faithfulness + Factual Correctness measured on a fixed 30-query test set
- [ ] **ChromaDB memory layer** — past research sessions cached; avoids re-fetching known papers
- [ ] **Streamlit frontend** — topic input, CSV upload, results display with benchmark tables and SHAP charts

## Nice-to-have (post-MVP)
- [ ] BibTeX export for literature survey
- [ ] Citation graph visualisation (which papers cite which)
- [ ] Benchmark trend charts over time (metric improvements year-over-year)
- [ ] Novelty scorer — how unique is a given paper idea vs. existing corpus?
- [ ] Token cost display per run in Streamlit UI

## Non-goals
- No multi-user login or auth system
- No real-time monitoring or paper alerts
- No PDF parsing for full-text extraction (use arXiv structured metadata + abstracts first)
- No deployment to production cloud in MVP phase

## Success metrics
- Ragas Faithfulness score improves from baseline (no verifier) to post-verifier by at least 25 points
- EDA Agent correctly identifies at least 3 of 5 known issues in a labelled test CSV
- Modeling Agent beats a naive majority-class baseline on at least 4 of 5 Kaggle test datasets
- All agent runs produce a LangSmith trace — 100% observability coverage from day 1

## Constraints
- Time: 5–6 weeks
- Cost: Free tier only (Groq free tier, Tavily free tier, Semantic Scholar free API key, LangSmith free tier)
- Tech: Python-only stack (no Node.js, no JavaScript frontend beyond Streamlit)
