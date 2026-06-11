# ResearchBench AI

Multi-agent research assistant and automated data science pipeline. Two AI agents in one:

1. **Research Agent** — given a topic, searches arXiv + Semantic Scholar, extracts methods, verifies claims via RAG, and produces a research gap analysis.
2. **Data Science Agent** — given a CSV + target column, runs EDA, suggests features, trains XGBoost/LightGBM/RandomForest with cross-validation, checks for leakage, and explains predictions via SHAP.

Built with LangGraph, LangSmith, Groq (Llama 3), and Streamlit.

---

## Quickstart

```bash
# Clone
git clone https://github.com/KaustubhMukdam/researchbench-ai.git
cd researchbench-ai

# Copy env config and fill in your API keys
cp .env.example .env

# Install
pip install -r requirements.txt

# Start backend
python -m uvicorn app.main:app --reload --port 8000

# Start frontend (in another terminal)
python -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. Two tabs: **Research** (enter a topic) and **Data Science** (upload a CSV).

### Required API keys

| Service | Why Needed                                                    | Get One                  |
| ------- | ------------------------------------------------------------- | ------------------------ |
| Groq    | LLM inference (extraction, verification, feature suggestions) | https://console.groq.com |
| Tavily  | Web search fallback for paper retrieval                       | https://tavily.com       |

Optional but recommended: LangSmith (observability) and Semantic Scholar (better academic search).

---

## Architecture

```
[Streamlit Frontend]  :8501
         |
         | HTTP
         v
[FastAPI Backend]  :8000
    |          |
    v          v
[Research]   [DS Graph]
[Graph]      (5 nodes)
(4 nodes)    eda -> feature_engineering ->
retrieval ->  modeling -> validation ->
extraction -> explainability -> END
verification ->
gap_analysis -> END
    |          |
    v          v
[LangSmith]  ← tracing (latency, tokens, errors)
    |
    v
[ChromaDB]   ← paper embeddings cache
```

---

## Module 1 — Research Pipeline

Given a research topic (e.g. "transformers for tabular data"), the pipeline:

1. **Paper Retrieval** — searches arXiv + Semantic Scholar, deduplicates, ranks by citation count + recency
2. **Method Extraction** — sends each abstract to Groq, extracts structured methods, datasets, metrics, limitations
3. **Citation Verification** — cross-checks every extracted claim against source text via ChromaDB RAG; flags unsupported claims
4. **Gap Analysis** — synthesises all extractions into a benchmark comparison table + research gap summary

### Evaluation (Ragas)

| Metric             | Baseline | With Verifier | Delta      |
| ------------------ | -------- | ------------- | ---------- |
| Context Recall     | 0.000    | 0.125         | **+0.125** |
| Answer Correctness | 0.065    | 0.096         | **+0.031** |

> Faithfulness is excluded intentionally — see [docs/eval.md](docs/eval.md) for why baseline Faithfulness of 1.0 is a metric artifact, not genuine performance.

---

## Module 2 — Data Science Pipeline

Given a CSV file + target column, the pipeline:

1. **EDA** — missing values, data types, correlations, class balance, high-cardinality detection
2. **Feature Engineering** — Groq LLM suggests 3-5 new features with pandas code
3. **Modeling** — trains XGBoost, LightGBM, RandomForest with 5-fold cross-validation
4. **Validation** — checks for leakage (target-correlated features), overfitting (high CV variance), high cardinality, missing data bias
5. **Explainability** — SHAP values for best model (falls back to sklearn feature*importances* if SHAP version-incompatible)

### Results (beat naive baseline on 5/5 tested)

| Dataset         | Rows  | Baseline | Best Model   | CV Mean F1 | Delta       |
| --------------- | ----- | -------- | ------------ | ---------- | ----------- |
| Titanic         | 891   | 0.6162   | RandomForest | 0.8406     | **+0.2244** |
| Telco Churn     | 7,043 | 0.7346   | RandomForest | 0.8015     | **+0.0669** |
| Pima Diabetes   | 768   | 0.6510   | RandomForest | 0.7721     | **+0.1211** |
| Breast Cancer   | 569   | 0.6274   | LightGBM     | 0.9631     | **+0.3357** |
| Digits (Binary) | 1,797 | 0.5014   | LightGBM     | 0.9733     | **+0.4719** |

---

## Project Structure

```
researchbench-ai/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── routers/                   # REST API endpoints
│   │   ├── research.py            # POST /research
│   │   └── dataset.py             # POST /analyze-dataset
│   ├── services/                  # Graph wrappers
│   ├── models/                    # Pydantic schemas
│   ├── nodes/                     # LangGraph agent nodes (9 total)
│   │   ├── paper_retrieval.py     # arXiv + Semantic Scholar
│   │   ├── method_extraction.py   # Groq extraction
│   │   ├── citation_verifier.py   # ChromaDB RAG
│   │   ├── comparison_gap.py      # Gap synthesis
│   │   ├── eda_agent.py           # CSV analysis
│   │   ├── feature_agent.py       # LLM feature suggestions
│   │   ├── modeling_agent.py      # XGBoost + LightGBM + RF
│   │   ├── validation_agent.py    # Leakage + overfit checks
│   │   └── explainability_agent.py # SHAP / feature importance
│   ├── schemas.py                 # Research state + API response schemas
│   ├── research_graph.py          # Research LangGraph
│   └── ds_graph.py                # DS LangGraph
├── frontend/
│   └── streamlit_app.py           # Streamlit UI (2 tabs)
├── eval/
│   ├── test_queries.json          # 30-query benchmark set
│   ├── run_ragas.py               # Ragas evaluation script
│   └── results_log.jsonl          # Eval results ledger
├── data/                          # Test datasets
├── docs/                          # All project documentation
│   ├── eval.md                    # Evaluation results + analysis
│   ├── experiment_log.md          # ML experiment history
│   └── Phases/                    # Phase-by-phase breakdowns
└── .env.example                   # Environment config template
```

---

## Tech Stack

| Layer         | Technology                                               |
| ------------- | -------------------------------------------------------- |
| Orchestration | LangGraph (StateGraph)                                   |
| LLM Inference | Groq (Llama 3 70B)                                       |
| Search        | arXiv API, Semantic Scholar API, Tavily                  |
| Vector Memory | ChromaDB (persistent embeddings)                         |
| Evaluation    | Ragas (Faithfulness, Answer Correctness, Context Recall) |
| Observability | LangSmith (tracing, latency, token counts)               |
| Backend       | FastAPI                                                  |
| Frontend      | Streamlit                                                |
| ML            | scikit-learn, XGBoost, LightGBM, SHAP                    |

---

## Benchmarking

Run Ragas evaluation:

```bash
# Baseline (no citation verifier)
python eval/run_ragas.py --mode baseline

# With verifier (uses ChromaDB RAG)
python eval/run_ragas.py --mode with_verifier

# Quick test (first 5 queries)
python eval/run_ragas.py --mode baseline --limit 5

# Compare results
python eval/compare_runs.py
```

Run DS pipeline on a test dataset:

```bash
# Batch test (all datasets in data/)
python test_ds_batch.py

# Or through the UI: upload CSV in Streamlit, pick target, click Run
```

---

## Key Design Decisions

- **Groq over OpenAI:** Free tier is fast (~300 tok/s on Llama 3), essential for agentic loops needing 10-20 LLM calls per run
- **RAG over fine-tuning:** The citation verifier uses retrieval-augmented generation instead of fine-tuning — faster iteration, no GPU cost, no retraining when adding new papers
- **SHAP fallback:** SHAP's TreeExplainer has version compatibility issues with newer sklearn; the pipeline automatically falls back to sklearn's `feature_importances_` (equivalent for RandomForest)
- **Temp file uploads:** Uploaded CSVs are saved to temp files and deleted after pipeline completes — no disk accumulation
- **LangSmith as observability layer:** Every node execution, LLM call, and tool invocation is traced automatically with latency and token counts

---

## License

MIT
