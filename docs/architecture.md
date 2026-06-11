# Architecture — ResearchBench AI

## System overview
ResearchBench AI is a two-module multi-agent system orchestrated by LangGraph. The Research Module takes a topic query, retrieves relevant arXiv/Semantic Scholar papers, extracts structured benchmark data, verifies claims via RAG, and produces a comparison + research gap report. The DS Module takes a CSV file, runs automated EDA, feature engineering suggestions, multi-model training, leakage validation, and SHAP explainability. Both modules are observable via LangSmith and evaluated via Ragas. A Streamlit frontend connects to a FastAPI backend that triggers the agent graphs.

## Component diagram (ASCII)

```
[Streamlit Frontend]
    |
    | HTTP POST /research  OR  /analyze-dataset
    v
[FastAPI Backend]
    |
    |—————————————————————————————————————|
    |                                     |
    v                                     v
[Research Agent Graph]           [DS Agent Graph]
(LangGraph StateGraph)           (LangGraph StateGraph)
    |                                     |
    |— Paper Retrieval Agent              |— EDA Agent
    |     arXiv API                       |     pandas + ydata-profiling
    |     Semantic Scholar API            |
    |     Tavily (web fallback)           |— Feature Engineering Agent
    |                                     |     LLM suggestions + schema
    |— Method Extraction Agent            |
    |     Groq (Llama 3)                  |— Modeling Agent
    |     Pydantic schema validation      |     XGBoost, LightGBM, RF
    |                                     |     scikit-learn cross-val
    |— Citation Verifier Agent            |
    |     ChromaDB (paper embeddings)     |— Validation Agent
    |     Ragas Faithfulness check        |     leakage detection
    |                                     |     overfitting signals
    |— Comparison + Gap Agent             |
          LLM synthesis                   |— Explainability Agent
          benchmark table output               SHAP values
                                               feature importance chart
    |
    v
[ChromaDB] ←—— cached paper embeddings (persist to disk)

    |
    v
[LangSmith] ←—— ALL agent calls traced automatically (via LangChain callbacks)

    |
    v
[Ragas Evaluation] ←—— runs on fixed 30-query test set during benchmarking
    Faithfulness score
    Factual Correctness score
    Context Recall score
```

## Data flow — Research Module
1. User enters a research topic in Streamlit (e.g., "tabular fraud detection transformers")
2. Streamlit calls FastAPI `POST /research` with the topic string
3. FastAPI triggers the Research LangGraph StateGraph
4. Paper Retrieval Agent queries arXiv + Semantic Scholar; returns top-N papers as metadata dicts
5. ChromaDB is checked first — papers already cached are skipped (avoids re-hitting API rate limits)
6. New papers are embedded and stored in ChromaDB
7. Method Extraction Agent sends each abstract + metadata to Groq; returns a Pydantic `PaperExtraction` object (architecture, dataset, metrics, training method)
8. Citation Verifier Agent uses ChromaDB RAG to check extracted claims against source text; flags low-confidence claims
9. Comparison + Gap Agent synthesises all extractions into a benchmark table and a research gap summary
10. Final JSON response returned to FastAPI → Streamlit renders the table and report

## Data flow — DS Module
1. User uploads a CSV file in Streamlit
2. Streamlit calls FastAPI `POST /analyze-dataset`
3. FastAPI triggers the DS LangGraph StateGraph
4. EDA Agent runs pandas profiling; returns a structured `EDAReport` Pydantic object
5. Feature Engineering Agent sends EDA report + column metadata to Groq; returns feature suggestions
6. Modeling Agent trains XGBoost, LightGBM, RandomForest with 5-fold CV; returns `ModelResults` object
7. Validation Agent checks for leakage signals (e.g., target-correlated features that shouldn't exist at inference time)
8. Explainability Agent computes SHAP values for the best-performing model
9. Full `DSReport` returned to FastAPI → Streamlit renders metric table + SHAP bar chart

## Key interfaces
- Frontend ↔ Backend: REST API (FastAPI auto-generates `/docs` OpenAPI spec)
- Backend ↔ LangGraph: Python function calls (LangGraph is a Python library, not a separate service)
- Backend ↔ ChromaDB: ChromaDB Python client (`chromadb.PersistentClient`)
- All LLM calls: LangChain `ChatGroq` with LangSmith callbacks enabled via `LANGCHAIN_TRACING_V2=true`

## LangSmith integration (how it works)
Set two environment variables — LangSmith hooks in automatically via LangChain:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
```
Every LLM call, every tool call, every agent step is logged with token count, latency, and input/output. No extra code required.

## Ragas evaluation (how it works)
Ragas is NOT run on every live query. It runs only during benchmarking:
1. Maintain a fixed `eval/test_queries.json` — 30 arXiv queries with known ground-truth papers and claims
2. After any significant agent change, run `python eval/run_ragas.py`
3. Script passes: question, retrieved context, LLM-generated answer, and ground truth to Ragas
4. Ragas returns Faithfulness (0–1), Factual Correctness (0–1), Context Recall (0–1)
5. Log results in `docs/eval.md` experiment table

## Security considerations
- [ ] All API keys in `.env` file — never hardcoded, never committed to Git
- [ ] `.env` added to `.gitignore` before first commit
- [ ] Input validation on all CSV uploads (file size limit, column count limit)
- [ ] Semantic Scholar and arXiv requests go through a caching layer — never hit API directly without checking ChromaDB first
